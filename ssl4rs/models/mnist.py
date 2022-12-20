"""Implements a simple MNIST image classification module based on PyTorch-Lightning."""
import copy
import typing

import hydra
import omegaconf
import torch
import torch.nn.functional
import torchmetrics

import ssl4rs.data
import ssl4rs.models
import ssl4rs.utils

logger = ssl4rs.utils.logging.get_logger(__name__)


class MNISTClassifier(ssl4rs.models.BaseModel):
    """Example of LightningModule used for MNIST image classification.

    This class is derived from the framework's base model interface and it implements all the extra
    goodies required for automatic rendering/logging of predictions.

    For more information on the role and responsibilities of the LightningModule, see:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    For more information on the base class, see:
        `ssl4rs.models.utils.BaseModel`
    """

    def __init__(
        self,
        encoder_config: ssl4rs.utils.config.DictConfig,
        head_config: ssl4rs.utils.config.DictConfig,
        loss_config: ssl4rs.utils.config.DictConfig,
        optimization: typing.Optional[ssl4rs.utils.config.DictConfig] = None,
    ):
        """Initializes the LightningModule and its submodules, loss, metrics, and optimizer.

        Note: we pass everything in as configurations that can be used to instantiate modules directly
        as this seems to be the 'cleanest' way to log everything needed to reinstantiate the model
        from scratch without having to serialize the modules directly...
        """
        super().__init__()  # for the base class, default arguments are fine (nothing to forward)
        # this line allows us to access hparams with `self.hparams` + auto-stores them in checkpoints
        self.save_hyperparameters(logger=False)  # logger=False since we don't need duplicated logs
        self._create_example_input_array(  # for easier tracing/profiling; fake tensors for 'forward'
            data=torch.randn(4, 1, 28, 28),
            batch_size=4,
        )
        self.encoder = hydra.utils.instantiate(encoder_config)
        self.head = hydra.utils.instantiate(head_config)
        self.loss_fn = hydra.utils.instantiate(loss_config)
        assert optimization is None or isinstance(optimization, (dict, omegaconf.DictConfig))
        self.optim_config = optimization  # this will be instantiated later, if we actually need it

    def configure_metrics(self) -> torchmetrics.MetricCollection:
        """Configures and returns the metric objects to update when given predictions + labels."""
        return torchmetrics.MetricCollection(
            dict(
                accuracy=torchmetrics.classification.accuracy.Accuracy(num_classes=10),
            )
        )

    def configure_optimizers(self) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Configures and returns model-specific optimizers and schedulers to use during
        training."""
        logger.debug("Configuring MNIST module optimizer and scheduler...")
        assert self.optim_config is not None, "we're about to train, we need an optimization cfg!"
        optim_config = copy.deepcopy(self.optim_config)  # we'll fully resolve + convert it below
        omegaconf.OmegaConf.resolve(optim_config)
        assert "optimizer" in optim_config, "missing mandatory 'optimizer' field in optim cfg!"
        assert isinstance(optim_config.optimizer, (dict, omegaconf.DictConfig))
        if optim_config.get("freeze_no_grad_params", True):
            model_params = [p for p in self.parameters() if p.requires_grad]
            assert len(model_params) > 0, "no model parameters left to train??"
        else:
            model_params = self.parameters()
        optimizer = hydra.utils.instantiate(optim_config.optimizer, model_params)
        scheduler = None
        if "lr_scheduler" in optim_config:
            assert isinstance(optim_config.lr_scheduler, (dict, omegaconf.DictConfig))
            assert "scheduler" in optim_config.lr_scheduler, "missing mandatory 'scheduler' field!"
            scheduler = hydra.utils.instantiate(
                config=optim_config.lr_scheduler.scheduler,
                optimizer=optimizer,
            )
        output = omegaconf.OmegaConf.to_container(
            cfg=optim_config,
            resolve=True,
            throw_on_missing=True,
        )
        output["optimizer"] = optimizer
        if scheduler is not None:
            output["lr_scheduler"]["scheduler"] = scheduler
        if "freeze_no_grad_params" in output:
            del output["freeze_no_grad_params"]
        return output

    def forward(self, batch: typing.Dict[typing.AnyStr, typing.Any]) -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        # we just need one thing from the batch, i.e. the 'data' tensor (input mnist image array)
        assert "data" in batch, "missing mandatory 'data' field with input tensor data!"
        input_tensor = batch["data"]
        assert input_tensor.ndim == 4
        batch_size, img_shape = input_tensor.shape[0], input_tensor.shape[1:]
        assert img_shape == (1, 28, 28), "unexpected mnist input tensor image shape"
        assert batch_size == ssl4rs.data.get_batch_size(batch)
        embed = self.encoder(input_tensor)
        assert embed.ndim >= 2 and embed.shape[0] == batch_size
        embed = torch.flatten(embed, start_dim=1)
        logits = self.head(embed)
        assert logits.ndim == 2 and logits.shape == (batch_size, 10)
        return logits

    def _generic_step(
        self,
        batch: typing.Dict[typing.AnyStr, typing.Any],
        batch_idx: int,
    ) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Runs a generic version of the forward + evaluation step for the train/valid/test loops.

        In comparison with the regular `forward()` function, this function will compute the loss
        and return multiple outputs used to update the metrics based on the assumption that the
        batch dictionary also contains info about the target labels. This means that it should
        never be used in production, as we would then try to access labels that do not exist.
        """
        preds = self(batch)  # this will call the 'forward' implementation above and return preds
        assert "target" in batch, "missing mandatory 'target' field with target labels data!"
        target = batch["target"]
        loss = self.loss_fn(preds, target)
        return {
            "loss": loss,  # mandatory for training loop, optional for validation/testing
            "preds": preds,  # used in metrics, logging, and potentially even returned to user
            "target": target,  # so that metric update functions have access to the tensor itself
            "batch_size": ssl4rs.data.get_batch_size(batch),  # so that logging functions can use it
        }

    def _get_data_id(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        batch: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]],  # null when not in loop
        batch_idx: int,  # index of the batch itself inside the dataloader loop
        sample_idx: int,  # index of the data sample itself that should be ID'd inside the batch
        dataloader_idx: int = 0,  # index of the dataloader that the batch was loaded from
    ) -> typing.Hashable:
        """Returns a unique 'identifier' for a particular data sample in a specific batch."""
        # if the batch data is available, we'll return the original ID from the parser...
        if batch is not None:
            assert (
                "batch_id" in batch
            ), "missing mandatory 'batch_id' field required to generate persistent data sample IDs!"
            assert (
                "batch_size" in batch
            ), "missing mandatory 'batch_size' field required to validate persistent data sample IDs!"
            batch_size, batch_ids = ssl4rs.data.get_batch_size(batch), batch["batch_id"]
            assert len(batch_ids) == batch_size, "unexpected batch id/size mismatch?"
            assert 0 <= sample_idx < batch_size, "out-of-scope sample idx wrt batch size!"
            batch_id = batch_ids[sample_idx]
            assert isinstance(batch_id, typing.Hashable), f"bad batch id type: {type(batch_id)}"
            return batch_id
        # otherwise, we cannot do any better than to use the default impl as temporary IDs
        return super()._get_data_id(
            loop_type=loop_type,
            batch=None,
            batch_idx=batch_idx,
            sample_idx=sample_idx,
            dataloader_idx=dataloader_idx,
        )

    def _render_and_log_samples(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        batch: typing.Dict[typing.AnyStr, typing.Any],
        batch_idx: int,
        sample_idxs: typing.List[int],
        sample_ids: typing.List[typing.Hashable],
        outputs: typing.Dict[typing.AnyStr, typing.Any],
        dataloader_idx: int = 0,
    ) -> typing.Any:
        """Renders and logs a specific data samples from the current batch using available
        loggers."""
        assert len(sample_idxs) == len(sample_ids) and len(sample_idxs) > 0
        # we'll render the input tensors with their IDs, predicted, and target labels underneath
        pred_idxs, target_idxs = torch.argmax(outputs["preds"], dim=1), outputs["target"]
        outputs = []
        for sample_idx, sample_id in zip(sample_idxs, sample_ids):
            input_tensor = batch["data"][sample_idx][0].cpu().numpy()
            image = ssl4rs.utils.drawing.get_displayable_image(input_tensor)
            image = ssl4rs.utils.drawing.resize_nn(image, zoom_factor=10)
            image = ssl4rs.utils.drawing.add_subtitle_to_image(
                image=image, subtitle=str(sample_id), extra_border_size=10, scale=1.35
            )
            sstr = f"Predicted: {pred_idxs[sample_idx]}, Target: {target_idxs[sample_idx]}"
            image = ssl4rs.utils.drawing.add_subtitle_to_image(
                image=image,
                subtitle=sstr,
                extra_subtitle_padding=4,
                scale=1.5,
            )
            self._log_rendered_image(image, key=f"{loop_type}/{sample_id}")
            outputs.append(image)
        return outputs
