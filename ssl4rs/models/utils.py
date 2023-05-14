"""Contains utility functions and a base interface for LightningModules-derived objects."""
import abc
import copy
import os
import typing

import cv2 as cv
import hydra
import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch.utilities.types as pl_types
import numpy as np
import omegaconf
import torch
import torch.utils.data
import torchmetrics

import ssl4rs

logger = ssl4rs.utils.logging.get_logger(__name__)
PLCallback = pl.callbacks.callback.Callback


class BaseModel(pl.LightningModule):
    """Base LightningModule (model) interface.

    Using this interface is not mandatory for experiments in this framework, but it will help log
    and debug basic batch-related issues. It also exposes a few of the useful (but rarely remembered)
    features that the base LightningModule implementation supports, such as metrics that can be
    stored/reloaded inside checkpoints.

    The main (not-mandatory-but-still-useful) feature offered here is the ability to 'render' the
    same batches (with predictions) every epoch based on their IDs. The way this is done is by first
    trying to find out the data loader batch size and the expected batch count. Given this, we pick
    a random set samples based on the total expected number of samples (batch size x data loader
    length) we will see. Each time we see one of these picked samples for the first time, we
    memorize its real batch identifier, and render it. In subsequent epochs, we check for the batch
    identifier directly, and re-render the same samples we saw in the past. This might break with
    variable-length or iterator-based data loaders, and should be disabled in that case (by setting
    `sample_count_to_render=0` in the constructor). With combined data loaders or data loaders that
    work with varying batch lengths, it might not always be able to find/render as many samples
    as the requested number, so see the `sample_count_to_render` as an optimistic upper bound.

    Note that regarding the usage of torchmetrics, there are some pitfalls to avoid e.g. when
    using multiple data loaders; refer to the following link for more information:
        https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#common-pitfalls

    For more information on the role of the base LightningModule interface, see:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(
        self,
        optimization: typing.Optional[ssl4rs.utils.DictConfig],
        log_train_metrics_each_step: bool = False,
        sample_count_to_render: int = 10,
        log_metrics_in_loop_types: typing.Sequence[str] = ("train", "valid", "test"),
    ):
        """Initializes the base model interface and its attributes.

        Note: we do NOT call `self.save_hyperparameters` in this base class constructor, but it
        should be called in the derived classes in order to make checkpoint reloading work. See
        these links for more information:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
            https://github.com/Lightning-AI/lightning/issues/16206

        Args:
            optimization: optimizer+scheduler configuration dictionary used during training. If
                training is never expected, can be `None`.
            log_train_metrics_each_step: toggles whether the metrics should be computed and logged
                each step in the training loop (true), or whether we should accumulate predictions
                and targets and only compute/log the metrics at the end of the epoch (false) as in
                the validation and testing loops. Defaults to false. When the predictions take a lot
                of memory (e.g. when doing semantic image segmentation with many classes), it might
                be best to turn this off to avoid out-of-memory errors when running long epochs, or
                drastic slowdowns when complex metrics are used.
            sample_count_to_render: number of samples that should (ideally) be rendered using the
                internal rendering function (if any is defined). Note that if the dataset is too
                small or if we do not have a reliable way to get good persistent IDs for data
                samples, we might be forced to render fewer samples than the number specified here.
            log_metrics_in_loop_types: defines the sequence of loop types (e.g. "train", "valid",
                and "test") for which we should compute metrics. The default covers all loop types
                where we can expect to have target labels.
        """
        super().__init__()
        logger.debug("Instantiating LightningModule base class...")
        self.log_train_metrics_each_step = log_train_metrics_each_step
        assert sample_count_to_render >= 0, "invalid sample count to render (should be >= 0)"
        self.sample_count_to_render = sample_count_to_render
        # remember to override the `_create_example_input_array` function if you ever want the base
        # class to know how to e.g. convert the model to onnx/torchscript, trace it, or to give
        # users an idea of the input tensors that are typically used in the `forward(...)` function!
        self.example_input_array: typing.Optional[typing.Any] = None
        self.metrics = self._instantiate_metrics(log_metrics_in_loop_types)  # auto-updated + reset
        self._ids_to_render: typing.Dict[str, typing.List[typing.Hashable]] = {}
        assert optimization is None or isinstance(
            optimization, (typing.Dict, omegaconf.DictConfig)
        ), f"incompatible optimization config type: {type(optimization)}"
        self.optimization = optimization  # this will be instantiated later, if we actually need it

    def _instantiate_metrics(
        self,
        log_metrics_in_loop_types: typing.Sequence[str],
    ) -> torch.nn.ModuleDict:
        """Instantiates and returns the metrics collections to use for all loop types.

        By default, this function will call the `configure_metrics` function in order to instantiate
        the actual metric objects, and it will clone those objects for each of the loop types that
        require metrics to be computed independently.

        Note that we prefix the metric collections with the `metrics` keyword in order to cleanly
        separate them from other weights inside the saved checkpoints.
        """
        assert not any(["/" in loop_type for loop_type in log_metrics_in_loop_types])
        logger.debug(f"Instantiating metrics collections for loops: {log_metrics_in_loop_types}")
        metrics = self.configure_metrics()
        metrics.persistent(True)  # by default, all metrics WILL be saved to checkpoints with this
        metrics = {  # below, we refer to the "metrics/loop_type" key as the metric group name
            f"metrics/{loop_type}": metrics.clone(prefix=(loop_type + "/")) for loop_type in log_metrics_in_loop_types
        }
        return torch.nn.ModuleDict(metrics)

    def has_metric(self, metric_name: typing.AnyStr) -> bool:
        """Returns whether this model possesses a metric with a specific name.

        The metric name is expected to be in `<loop_type>/<metric_name>` format. For example, it
        might be `valid/accuracy`. This metric name will be prefixed with `metric` internally.
        """
        loop_type, metric_name = metric_name.split("/", maxsplit=1)
        metric_group_name = f"metrics/{loop_type}"
        return metric_group_name in self.metrics and metric_name in self.metrics[metric_group_name]

    def compute_metric(self, metric_name: typing.AnyStr) -> typing.Any:
        """Returns the current value of a metric with a specific name.

        The metric name is expected to be in `<loop_type>/<metric_name>` format. For example, it
        might be `valid/accuracy`. This metric name will be prefixed with `metric` internally.
        """
        loop_type, metric_name = metric_name.split("/", maxsplit=1)
        metric_group_name = f"metrics/{loop_type}"
        assert metric_group_name in self.metrics
        return self.metrics[metric_group_name][metric_name].compute()

    @abc.abstractmethod
    def configure_metrics(self) -> torchmetrics.MetricCollection:
        """Configures and returns the metric objects to update when given predictions + labels.

        All metrics returned here should be train/valid/test loop agnostic and (likely) derived from
        the `torchmetrics.Metric` interface. We will clone the returned output for each of the
        train/valid/test loop types so that all metrics can be independently reset and updated at
        different frequencies (if needed).

        In order to NOT use a particular metric in a loop type, or in order to NOT compute metrics
        in a certain loop type at all, override the `_instantiate_metrics` function.
        """
        raise NotImplementedError

    def configure_callbacks(self) -> typing.Union[typing.Sequence[PLCallback], PLCallback]:
        """Configures and returns model-specific callbacks.

        No callbacks are used by default. When the model gets attached to a trainer, i.e. when the
        trainer's ``.fit()`` or ``.test()`` gets called with thismodel, the list of callbacks
        returned here will be merged with the list of callbacks passed to the Trainer's
        ``callbacks`` argument.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-callbacks
        """
        return []

    def configure_optimizers(self) -> typing.Any:
        """Configures and returns model-specific optimizers and schedulers to use during training.

        This function can return a pretty wild number of object combinations; refer to the docs
        for the full list and a bunch of examples:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers

        For this particular framework, we favor the use of a dictionary that contains an ``optimizer``
        key and a ``lr_scheduler`` key. For example, the implementation of this function could be:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, ...),
                    "monitor": "valid/loss",
                    "frequency": self.trainer.check_val_every_n_epoch * 4,
                },
            }

        If you need to use the estimated number of stepping batches during training (e.g. when using
        the `OneCycleLR` scheduler), use the `self.trainer.estimated_stepping_batches` value.
        """
        logger.debug("Configuring module optimizer and scheduler...")
        assert self.optimization is not None, "we're about to train, we need an optimization cfg!"
        optimization = copy.deepcopy(self.optimization)  # we'll fully resolve + convert it below
        omegaconf.OmegaConf.resolve(optimization)
        assert "optimizer" in optimization, "missing mandatory 'optimizer' field in optim cfg!"
        assert isinstance(optimization.optimizer, (dict, omegaconf.DictConfig))
        if optimization.get("freeze_no_grad_params", True):
            model_params = [p for p in self.parameters() if p.requires_grad]
            assert len(model_params) > 0, "no model parameters left to train??"
        else:
            model_params = self.parameters()
        optimizer = hydra.utils.instantiate(optimization.optimizer, model_params)
        scheduler = None
        if "lr_scheduler" in optimization:
            assert isinstance(optimization.lr_scheduler, (dict, omegaconf.DictConfig))
            assert "scheduler" in optimization.lr_scheduler, "missing mandatory 'scheduler' field!"
            scheduler = hydra.utils.instantiate(
                config=optimization.lr_scheduler.scheduler,
                optimizer=optimizer,
            )
        output = omegaconf.OmegaConf.to_container(
            cfg=optimization,
            resolve=True,
            throw_on_missing=True,
        )
        output["optimizer"] = optimizer
        if scheduler is not None:
            output["lr_scheduler"]["scheduler"] = scheduler
        if "freeze_no_grad_params" in output:
            del output["freeze_no_grad_params"]
        return output

    def _create_example_input_array(self, **kwargs) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Wraps the given kwargs inside a fake batch dict to be used as the example input.

        The `self.example_input_array` attribute is actually used by Lightning to offer lots of
        small debugging/logging features, but remains optional.
        """
        batch_data = dict(**kwargs)
        self.example_input_array = dict(batch=batch_data)
        return self.example_input_array

    @abc.abstractmethod
    def forward(self, batch: ssl4rs.data.BatchDictType) -> typing.Any:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`.

        This function is meant to be used mostly for inference purposes, e.g. when this model is
        reloaded from a checkpoint and used in a downstream application.

        With this interface, we always expect that the inputs will be provided under a DICTIONARY
        format which can be used to fetch/store various tensors used as input, output, or for
        debugging/visualization. This means that if the `example_input_array` is ever set, it should
        correspond to a dictionary itself, such as:
            model = SomeClassDerivedFromBaseModel(...)
            model.example_input_array = {"batch": {"tensor_A": ...}}

        The output of this function should only be the "prediction" of the model, i.e. what it would
        provide given only input data in a production setting.

        Args:
            batch: a dictionary of batch data loaded by a data loader object. This dictionary may
                be modified and new attributes may be added into it by the LightningModule.

        Returns:
            The model's prediction result.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _generic_step(
        self,
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
    ) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Runs a generic version of the forward + evaluation step for the train/valid/test loops.

        In comparison with the regular `forward()` function, this function will compute the loss
        and return multiple outputs used to update the metrics based on the assumption that the
        batch dictionary also contains info about the target labels. This means that it should never
        be used in production, as we would then try to access labels that do not exist.

        Args:
            batch: a dictionary of batch data loaded by a data loader object. This dictionary may
                be modified and new attributes may be added into it by the LightningModule.
            batch_idx: the index of the provided batch in the data loader's current loop.

        Returns:
            A dictionary of outputs (likely tensors) indexed using names. Typically, this would
            contain at least `loss`, `preds`, and `target` tensors so that we can easily update
            the metrics, render predictions/errors images, and log the results (as needed).
        """
        raise NotImplementedError

    def on_train_epoch_start(self) -> None:
        """Resets the metrics + render IDs before the start of a new epoch."""
        self._reset_metrics("train")
        if "train" not in self._ids_to_render:
            ids = self._pick_ids_to_render("train")
            logger.debug(f"Will try to render {len(ids)} training samples")
            self._ids_to_render["train"] = ids

    def training_step(
        self,
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
    ) -> pl_types.STEP_OUTPUT:
        """Runs a forward + evaluation step for the training loop.

        Note that this step may be happening across multiple devices/nodes.

        Args:
            batch: a dictionary of batch data loaded by a data loader object.
            batch_idx: the index of the provided batch in the data loader's current loop.

        Returns:
            The full outputs dictionary that should be reassembled across all potential devices
            and nodes, and that might be further used in the `on_train_batch_end` function.
        """
        outputs = self._generic_step(batch, batch_idx)
        assert "loss" in outputs, "loss tensor is NOT optional in training step implementation (needed for backprop!)"
        return outputs

    def on_train_batch_end(
        self,
        outputs: pl_types.STEP_OUTPUT,
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
    ) -> None:
        """Completes the forward + evaluation step for the training loop.

        Args:
            outputs: the outputs of the `training_step` method that was just completed.
            batch: a dictionary of batch data loaded by a data loader object.
            batch_idx: the index of the provided batch in the data loader's current loop.

        Returns:
            Nothing.
        """
        assert (
            "loss" in outputs
        ), "loss tensor is NOT optional in training step end implementation (needed for backprop!)"
        loss = outputs["loss"]
        batch_size = outputs.get(ssl4rs.data.batch_size_key, None)
        # todo: figure out if we need to add sync_dist arg to self.log calls below?
        self.log("train/loss", loss.item(), prog_bar=True, batch_size=batch_size)
        self.log("train/epoch", float(self.current_epoch), batch_size=batch_size)
        metrics_val = self._update_metrics(
            loop_type="train",
            outputs=outputs,
            return_vals=self.log_train_metrics_each_step,
        )
        if self.log_train_metrics_each_step:
            assert metrics_val is not None and isinstance(metrics_val, dict)
            self.log_dict(metrics_val, batch_size=batch_size)
        self._check_and_render_batch(
            loop_type="train",
            batch=batch,
            batch_idx=batch_idx,
            outputs=outputs,
        )

    def on_train_epoch_end(self) -> None:
        """Computes and logs the training metrics (if not always done at the step level)."""
        if not self.log_train_metrics_each_step:
            metrics_val = self.compute_metrics(loop_type="train")
            self.log_dict(metrics_val)

    def on_validation_epoch_start(self) -> None:
        """Resets the metrics + render IDs before the start of a new epoch."""
        self._reset_metrics("valid")
        if "valid" not in self._ids_to_render:
            ids = self._pick_ids_to_render("valid")
            logger.debug(f"Will try to render {len(ids)} validation samples")
            self._ids_to_render["valid"] = ids

    def validation_step(
        self,
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> typing.Optional[pl_types.STEP_OUTPUT]:
        """Runs a forward + evaluation step for the validation loop.

        Note that this step may be happening across multiple devices/nodes. It is recommended to
        evaluate on a single device to ensure each sample/batch gets evaluated exactly once. This
        is helpful to make sure benchmarking for research papers is done the right way. Otherwise,
        in a multi-device setting, samples could occur duplicated when DistributedSampler is used,
        e.g. with strategy="ddp". It replicates some samples on some devices to make sure all
        devices have the same batch size in case of uneven inputs.

        Args:
            batch: a dictionary of batch data loaded by a data loader object.
            batch_idx: the index of the provided batch in the data loader's current loop.
            dataloader_idx: the index of the dataloader that's being used (if more than one).

        Returns:
            The full outputs dictionary that should be reassembled across all potential devices
            and nodes, and that might be further used in the `on_validation_batch_end` function.
        """
        return self._generic_step(batch, batch_idx)

    def on_validation_batch_end(
        self,
        outputs: typing.Optional[pl_types.STEP_OUTPUT],
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Completes the forward + evaluation step for the validation loop.

        Args:
            outputs: the outputs of the `validation_step` method that was just completed.
            batch: a dictionary of batch data loaded by a data loader object.
            batch_idx: the index of the provided batch in the data loader's current loop.
            dataloader_idx: the index of the dataloader that's being used (if more than one).

        Returns:
            Nothing.
        """
        if "loss" in outputs:
            loss = outputs["loss"]
            batch_size = outputs.get(ssl4rs.data.batch_size_key, None)
            # todo: figure out if we need to add sync_dist arg to self.log calls below?
            self.log("valid/loss", loss.item(), prog_bar=True, batch_size=batch_size)
        self._update_metrics(loop_type="valid", outputs=outputs, return_vals=False)
        self._check_and_render_batch(
            loop_type="valid",
            batch=batch,
            batch_idx=batch_idx,
            outputs=outputs,
            dataloader_idx=dataloader_idx,
        )

    def on_validation_epoch_end(self) -> None:
        """Completes the epoch by asking the evaluator to summarize its results."""
        metrics_val = self.compute_metrics(loop_type="valid")
        self.log_dict(metrics_val)
        fit_state_fn = pl.trainer.trainer.TrainerFn.FITTING
        if self.trainer is not None and self.trainer.state.fn == fit_state_fn:
            for metric_name, metric_val in metrics_val.items():
                logger.debug(f"epoch#{self.current_epoch:03d} {metric_name}: {metric_val}")

    def on_test_epoch_start(self) -> None:
        """Resets the metrics + render IDs before the start of a new epoch."""
        self._reset_metrics("test")
        if "test" not in self._ids_to_render:
            ids = self._pick_ids_to_render("test")
            logger.debug(f"Will try to render {len(ids)} testing samples")
            self._ids_to_render["test"] = ids

    def test_step(
        self,
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> typing.Optional[pl_types.STEP_OUTPUT]:
        """Runs a forward + evaluation step for the testing loop.

        Note that this step may be happening across multiple devices/nodes. It is recommended to
        evaluate on a single device to ensure each sample/batch gets evaluated exactly once. This
        is helpful to make sure benchmarking for research papers is done the right way. Otherwise,
        in a multi-device setting, samples could occur duplicated when DistributedSampler is used,
        e.g. with strategy="ddp". It replicates some samples on some devices to make sure all
        devices have the same batch size in case of uneven inputs.

        Args:
            batch: a dictionary of batch data loaded by a data loader object.
            batch_idx: the index of the provided batch in the data loader's current loop.
            dataloader_idx: the index of the dataloader that's being used (if more than one).

        Returns:
            The full outputs dictionary that should be reassembled across all potential devices
            and nodes, and that might be further used in the `on_test_batch_end` function.
        """
        return self._generic_step(batch, batch_idx)

    def on_test_batch_end(
        self,
        outputs: typing.Optional[pl_types.STEP_OUTPUT],
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Completes the forward + evaluation step for the testing loop.

        Args:
            outputs: the outputs of the `test_step` method that was just completed.
            batch: a dictionary of batch data loaded by a data loader object.
            batch_idx: the index of the provided batch in the data loader's current loop.
            dataloader_idx: the index of the dataloader that's being used (if more than one).

        Returns:
            Nothing.
        """
        if "loss" in outputs:
            loss = outputs["loss"]
            batch_size = outputs.get(ssl4rs.data.batch_size_key, None)
            # todo: figure out if we need to add sync_dist arg to self.log calls below?
            self.log("test/loss", loss.item(), prog_bar=True, batch_size=batch_size)
        self._update_metrics(loop_type="test", outputs=outputs, return_vals=False)
        self._check_and_render_batch(
            loop_type="test",
            batch=batch,
            batch_idx=batch_idx,
            outputs=outputs,
            dataloader_idx=dataloader_idx,
        )

    def on_test_epoch_end(self) -> None:
        """Completes the epoch by asking the evaluator to summarize its results."""
        metrics_val = self.compute_metrics(loop_type="test")
        self.log_dict(metrics_val)

    def predict_step(
        self,
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> typing.Any:
        """Runs a prediction step on new data, returning only the predictions.

        In comparison with the model's `forward()` implementation, this hook may be used to scale
        inference on multi-node/device setups.

        Note: if you are interested in logging the predictions of the model to disk while computing
        them to avoid out-of-memory issues, refer to `pl.callbacks.BasePredictionWriter`:
            https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html
        """
        return self(batch)

    @staticmethod
    def _get_batch_size_from_data_loader(data_loader: typing.Any) -> int:
        """Returns the batch size that will be (usually) used by a given data loader.

        This batch size should correspond to the 'max' batch size typically seen in all batches. It
        is an optimistic prediction of the real batch size that will probably be seen at runtime.
        """
        # note: with an extra flag, we could try to load the 1st batch from the loader and check it...
        if isinstance(data_loader, pl.utilities.combined_loader.CombinedLoader):
            potential_batch_sizes = [
                bs.batch_size if bs is not None else getattr(data_loader, "batch_size", None)
                for bs in data_loader.batch_sampler
            ]
            assert any(
                [bs is not None for bs in potential_batch_sizes]
            ), "could not find any batch size hint in combined data loader!"
            expected_batch_size = max(bs for bs in potential_batch_sizes if bs is not None)
        elif hasattr(data_loader, "batch_sampler"):
            assert hasattr(data_loader.batch_sampler, "batch_size")
            # noinspection PyUnresolvedReferences
            expected_batch_size = data_loader.batch_sampler.batch_size
        else:
            assert hasattr(data_loader, "batch_size"), "missing batch size hint!"
            expected_batch_size = data_loader.batch_size
        assert expected_batch_size > 0, "bad expected batch size found!"
        return expected_batch_size

    def _get_data_id(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        batch: typing.Optional[ssl4rs.data.BatchDictType],  # null when not in loop
        batch_idx: int,  # index of the batch itself inside the dataloader loop
        sample_idx: int,  # index of the data sample itself that should be ID'd inside the batch
        dataloader_idx: int = 0,  # index of the dataloader that the batch was loaded from
    ) -> typing.Hashable:
        """Returns a unique 'identifier' for a particular data sample in a specific batch.

        By default, the approach to tag samples without batch data that is provided below is not
        robust to dataloader shuffling, meaning that derived classes should implement one if they
        want always-persistent IDs (even before train/valid/test loops are ever invoked). If batch
        data is available, we will assume that we can use an attribute called 'batch_id' directly
        as the identifier for each batch sample. To generate such an attribute in all your batches,
        refer to the `_get_batch_id_for_index` function of the `DataParser` base class.
        """
        assert batch is None or isinstance(batch, dict)
        assert batch_idx >= 0 and sample_idx >= 0 and dataloader_idx >= 0
        if batch is not None:
            assert (
                "batch_id" in batch
            ), "missing mandatory 'batch_id' field required to generate persistent data sample IDs!"
            assert (
                ssl4rs.data.batch_size_key in batch
            ), "missing mandatory 'batch_size' field required to validate persistent data sample IDs!"
            batch_size, batch_ids = ssl4rs.data.get_batch_size(batch), batch["batch_id"]
            assert len(batch_ids) == batch_size, "unexpected batch id/size mismatch?"
            if sample_idx < batch_size:
                batch_id = batch_ids[sample_idx]
                assert isinstance(batch_id, typing.Hashable), f"bad batch id type: {type(batch_id)}"
                return batch_id
        # if we don't have batch ids or we requested an out-of-batch sample idx, return a generic id
        return f"{loop_type}_loader{dataloader_idx:02d}_batch{batch_idx:05d}_sample{sample_idx:05d}"

    def _get_data_ids_for_batch(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        batch: typing.Optional[ssl4rs.data.BatchDictType],  # null when not in loop
        batch_idx: int,  # index of the batch itself inside the dataloader loop
        dataloader_idx: int = 0,  # index of the dataloader that the batch was loaded from
    ) -> typing.List[typing.Hashable]:
        """Returns a list of identifiers used to uniquely tag all data sample in a given batch.

        This function  returns potentially temporary IDs provided by the `_get_data_id` function.
        """
        assert loop_type in ["train", "valid", "test"]
        if batch is not None and ssl4rs.data.batch_size_key in batch:
            batch_size = ssl4rs.data.get_batch_size(batch)
        else:
            if loop_type == "train":
                assert dataloader_idx == 0
                dataloader = self.trainer.train_dataloader
            elif loop_type == "valid":
                dataloader = self.trainer.val_dataloaders
            elif loop_type == "test":
                dataloader = self.trainer.test_dataloaders
            else:
                raise NotImplementedError
            batch_size = self._get_batch_size_from_data_loader(dataloader)
        assert batch_size > 0
        if loop_type == "train":
            assert dataloader_idx == 0
            assert 0 <= batch_idx < self.trainer.num_training_batches
            return [
                self._get_data_id(
                    loop_type=loop_type,
                    batch=batch,
                    batch_idx=batch_idx,
                    sample_idx=sample_idx,
                    dataloader_idx=dataloader_idx,
                )
                for sample_idx in range(batch_size)
            ]
        elif loop_type in ["valid", "test"]:
            if loop_type == "valid":
                num_batches = self.trainer.num_val_batches
            else:
                num_batches = self.trainer.num_test_batches
            assert 0 <= dataloader_idx < len(num_batches)
            assert 0 <= batch_idx < num_batches[dataloader_idx]
            return [
                self._get_data_id(
                    loop_type=loop_type,
                    batch=batch,
                    batch_idx=batch_idx,
                    sample_idx=sample_idx,
                    dataloader_idx=dataloader_idx,
                )
                for sample_idx in range(batch_size)
            ]
        else:
            raise NotImplementedError

    def _pick_ids_to_render(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        seed: typing.Optional[int] = 1000,  # if left as `None`, IDs will likely change each run
    ) -> typing.List[typing.Hashable]:
        """Returns the list of data sample ids that should be rendered/displayed/logged each epoch.

        This function will rely on internal batch count/size attributes in order to figure out
        how many of them should be rendered, and which. If the associated dataloader(s) contain(s)
        fewer samples than the requested count, this function will return the maximum number of
        sample IDs that can be rendered.

        Due to the naive approach used in this base class to identify/tag samples, it may also be
        possible that we return IDs that can never be seen (e.g. due to varying batch sizes). The
        rendering function will just have to ignore those IDs on its own. Also, if the derived
        class does not have a persistent way to get batch IDs without having access to batch data,
        using shuffling on the data loader may result in the re-shuffling of IDs each run as well.
        """
        assert loop_type in ["train", "valid", "test"]
        picked_ids = []
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", None)
        rng = np.random.default_rng(seed=seed)
        if loop_type == "train":
            batch_size = self._get_batch_size_from_data_loader(self.trainer.train_dataloader)
            selected_batch_idxs = rng.choice(
                self.trainer.num_training_batches,
                size=min(self.sample_count_to_render, self.trainer.num_training_batches),
                replace=False,
            )
            for batch_idx in selected_batch_idxs:
                picked_ids.append(
                    self._get_data_id(
                        loop_type=loop_type,
                        batch=None,  # we do not have the actual batch data yet!
                        batch_idx=batch_idx,
                        sample_idx=rng.choice(batch_size),
                        dataloader_idx=0,
                    )
                )
        elif loop_type in ["valid", "test"]:
            if loop_type == "valid":
                num_batches = self.trainer.num_val_batches
                dataloaders = self.trainer.val_dataloaders
            else:
                num_batches = self.trainer.num_test_batches
                dataloaders = self.trainer.test_dataloaders
            if len(num_batches) > 1:
                selected_dataloader_idxs = rng.choice(
                    len(num_batches),
                    size=self.sample_count_to_render,
                )
                for dataloader_idx in range(len(num_batches)):
                    curr_count = np.count_nonzero(selected_dataloader_idxs == dataloader_idx)
                    batch_size = self._get_batch_size_from_data_loader(dataloaders[dataloader_idx])
                    selected_batch_idxs = rng.choice(
                        num_batches[dataloader_idx],
                        size=min(curr_count, num_batches[dataloader_idx]),
                        replace=False,
                    )
                    for batch_idx in selected_batch_idxs:
                        picked_ids.append(
                            self._get_data_id(
                                loop_type=loop_type,
                                batch=None,  # we do not have the actual batch data yet!
                                batch_idx=batch_idx,
                                sample_idx=rng.choice(batch_size),
                                dataloader_idx=dataloader_idx,
                            )
                        )
            else:
                batch_size = self._get_batch_size_from_data_loader(dataloaders)
                selected_batch_idxs = rng.choice(
                    num_batches[0],
                    size=min(self.sample_count_to_render, num_batches[0]),
                    replace=False,
                )
                for batch_idx in selected_batch_idxs:
                    picked_ids.append(
                        self._get_data_id(
                            loop_type=loop_type,
                            batch=None,  # we do not have the actual batch data yet!
                            batch_idx=batch_idx,
                            sample_idx=rng.choice(batch_size),
                            dataloader_idx=0,
                        )
                    )
        else:
            raise NotImplementedError
        return picked_ids

    def _check_and_render_batch(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
        outputs: typing.Dict[typing.AnyStr, typing.Any],
        dataloader_idx: int = 0,
    ) -> typing.Any:
        """Extracts and renders data samples from the current batch (if any match is found).

        This function relies on the picked data sample IDs that are generated at the beginning of
        the 1st epoch of training/validation/testing. If a match is found for any picked id in the
        current batch, we will render the corresponding data, log it (if possible), and return the
        rendering result.
        """
        if loop_type not in self._ids_to_render or not self._ids_to_render[loop_type]:
            return  # quick exit if we don't actually want to render/log any predictions
        assert batch is not None, "it's rendering time, we need the batch data!"
        # first step is to check what sample IDs we have in front of us with the current batch
        # (we'll extract IDs with + without batch data, in case some need to be made persistent)
        persistent_ids, temporary_ids = (
            self._get_data_ids_for_batch(
                loop_type=loop_type,
                batch=_batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )
            for _batch in [batch, None]
        )
        assert len(persistent_ids) <= len(
            temporary_ids
        ), "it makes no sense to have more persistent than temporary IDs, ever?"
        batch_ids = persistent_ids + temporary_ids
        ids_to_render = self._ids_to_render[loop_type]
        assert isinstance(ids_to_render, list)
        if not any([sid in ids_to_render for sid in batch_ids]):
            return  # this batch contains nothing we need to render
        # before rendering, if we got hits from temporary (non-batch-data-based-) IDs, replace them
        matched_sample_idxs = []  # indices of samples-within-the-current-batch to be rendered
        matched_sample_ids = []  # in case the rendering function would also like to access those
        for persistent_id, temp_id in zip(persistent_ids, temporary_ids):
            if persistent_id != temp_id and temp_id in ids_to_render:
                assert persistent_id not in ids_to_render
                ids_to_render[ids_to_render.index(temp_id)] = persistent_id
            if persistent_id in ids_to_render:
                matched_sample_idxs.append(persistent_ids.index(persistent_id))
                matched_sample_ids.append(persistent_id)
        # now, time to go render+log the selected samples based on the found persistent ID matches
        result = self._render_and_log_samples(
            loop_type=loop_type,
            batch=batch,
            batch_idx=batch_idx,
            sample_idxs=matched_sample_idxs,
            sample_ids=matched_sample_ids,
            outputs=outputs,
            dataloader_idx=dataloader_idx,
        )
        return result

    @abc.abstractmethod
    def _render_and_log_samples(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
        sample_idxs: typing.List[int],
        sample_ids: typing.List[typing.Hashable],
        outputs: typing.Dict[typing.AnyStr, typing.Any],
        dataloader_idx: int = 0,
    ) -> typing.Any:
        """Renders and logs a specific data samples from the current batch using available loggers.

        Note: by default, the base class has no idea what to render and how to properly log it, so
        this implementation does nothing. Derived classes are strongly suggested to implement this
        properly, but it is not actually required in order to just train a model.

        If you want to log OpenCV-based (BGR) images using all available and compatible loggers,
        see the `_log_rendered_image` helper function.
        """
        return None

    def _log_rendered_image(
        self,
        image: np.ndarray,  # in H x W x C (OpenCV) 8-bit BGR format
        key: str,  # should be a filesystem-compatible string if using mlflow artifacts
    ) -> None:
        """Logs an already-rendered image in OpenCV BGR format to TBX/MLFlow/Wandb."""
        logger.debug(f"Will try to log rendered image with key '{key}'")
        assert image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # noqa
        assert len(key) > 0
        for loggr in self.loggers:
            if isinstance(loggr, pl_loggers.TensorBoardLogger):
                loggr.experiment.add_image(
                    tag=key,
                    img_tensor=image_rgb,
                    global_step=self.global_step,
                    dataformats="HWC",
                )
            elif isinstance(loggr, pl_loggers.MLFlowLogger):
                assert loggr.run_id is not None
                loggr.experiment.log_image(
                    run_id=loggr.run_id,
                    image=image_rgb,
                    artifact_file=f"renders/{key}.png",
                )
            elif isinstance(loggr, pl_loggers.CometLogger):
                loggr.experiment.log_image(
                    image_data=image_rgb,
                    name=key,
                )
            elif isinstance(loggr, pl_loggers.WandbLogger):
                loggr.log_image(
                    key=key,
                    images=[image_rgb],
                    step=self.global_step,
                )

    def _update_metrics(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        outputs: typing.Dict[typing.AnyStr, typing.Any],
        return_vals: bool = False,  # in case we want to log the metric for the current outputs
    ) -> typing.Optional[typing.Dict[typing.AnyStr, typing.Any]]:
        """Updates the metrics for a particular metric collection (based on loop type)."""
        metric_vals = None
        metric_group = f"metrics/{loop_type}"
        if metric_group in self.metrics:
            metrics = self.metrics[metric_group]
            target = outputs.get("target", None)
            preds = outputs.get("preds", None)
            assert (
                target is not None and preds is not None
            ), "missing `target` and/or `preds` field in batch outputs to auto-update metrics!"
            if return_vals:
                metric_vals = metrics(preds, target)  # is a bit slower due to output
            else:
                metrics.update(preds, target)  # will save some compute time!
        return metric_vals

    def compute_metrics(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
    ) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns the metric values for a particular metric collection (based on loop type)."""
        metric_group = f"metrics/{loop_type}"
        if metric_group in self.metrics:
            return self.metrics[metric_group].compute()
        return {}

    def _reset_metrics(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
    ) -> None:
        """Resets the metrics for a particular metric collection (based on loop type)."""
        metric_group = f"metrics/{loop_type}"
        if metric_group in self.metrics:
            self.metrics[metric_group].reset()
