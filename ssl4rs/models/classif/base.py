"""Implements generic classifier modules based on Lightning."""

import typing

import cv2 as cv
import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional
import torchmetrics

import ssl4rs.data
import ssl4rs.utils
from ssl4rs.models.utils import BaseModel

logger = ssl4rs.utils.logging.get_logger(__name__)
TorchModuleOrDictConfig = typing.Union[torch.nn.Module, ssl4rs.utils.DictConfig]


class GenericClassifier(BaseModel):
    """Example of LightningModule used for image classification tasks.

    This class is derived from the framework's base model interface, and it implements all the
    extra goodies required for automatic rendering/logging of predictions. The input data and
    classification label attributes required to ingest and evaluate predictions are assumed to be
    specified via keys in the loaded batch dictionaries. The exact keys should be specified to the
    constructor of this class.

    This particular implementation expects to get a "backbone" encoder configuration alongside a
    "head" classification layer configuration. Embeddings generated in the forward pass are also
    flattened automatically, based on the assumption that we generate one embedding per image.

    For more information on the role and responsibilities of the LightningModule, see:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

    For more information on the base class, see:
        `ssl4rs.models.utils.BaseModel`
    """

    def __init__(
        self,
        encoder: TorchModuleOrDictConfig,
        head: typing.Optional[TorchModuleOrDictConfig],
        loss_fn: typing.Optional[TorchModuleOrDictConfig],
        metrics: typing.Optional[ssl4rs.utils.DictConfig],
        optimization: typing.Optional[ssl4rs.utils.DictConfig],
        num_output_classes: int,
        num_input_channels: int,
        freeze_encoder: bool = False,
        input_key: typing.AnyStr = "input",
        label_key: typing.AnyStr = "label",
        ignore_index: typing.Optional[int] = None,
        example_image_shape: typing.Optional[typing.Tuple[int, int]] = (224, 224),  # height, width
        save_hyperparams: bool = True,  # turn this off in derived classes
        **kwargs,
    ):
        """Initializes the LightningModule and its submodules, loss, metrics, and optimizer.

        Note: we favor passing everything in as dict configs that can be used to instantiate
        modules directly as this seems to be the 'cleanest' way to log everything needed to
        reinstantiate the model from scratch without having to serialize the modules directly...

        Args:
            encoder: dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the backbone encoder of the model. If a config is provided, it
                will be used to instantiate the backbone encoder via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object.
            head: optional dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the classification head of the model. If a config is provided, it
                will be used to instantiate the classifier via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object. If nothing is provided, we will
                assume that the backbone encoder already possesses a classifier, and will
                compute the loss directly on the backbone's output.
            loss_fn: optional dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the loss function of the model. If a config is provided, it
                will be used to instantiate the loss function via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object. If nothing is provided, we will
                assume that the model already implements its own loss, and a derived class will be
                computing it in its own override of the `generic_step` function.
            metrics: dict-based configuration that corresponds to the metrics to be instantiated
                during training/validation/testing. It must be possible to return the result
                of instantiating this config via `hydra.utils.instantiate` directly to a
                `torchmetrics.MetricCollection` object. If no config is provided, it will default
                to an accuracy metric only.
            optimization: dict-based configuration that can be used to instantiate the
                optimizer used by the trainer. This config is assumed to be formatted according
                to Lightning's format. See the base class's `configure_optimizers` for more info.
            num_output_classes: number of unique classes (categories) to be predicted.
            num_input_channels: number of channels in the images to be loaded.
            freeze_encoder: specifies whether to freeze (disable gradient computation) the encoder
                parameters.
            input_key: key used to fetch the input data tensor from the loaded batch dictionaries.
            label_key: key used to fetch the class label tensor from the loaded batch dictionaries.
            ignore_index: value used to indicate dontcare predictions. None = not used.
            example_image_shape: shape of the example image tensor to be created. Defaults to the
                commonly used imagenet image shape (224x224), but which might not always be OK.
            save_hyperparams: toggles whether hyperparameters should be saved in this class. This
                should be `False` when this class is derived, and the `save_hyperparameters`
                function should be called in the derived constructor.
        """
        assert num_output_classes >= 1, f"invalid number of output classes: {num_output_classes}"
        assert num_input_channels >= 1, f"invalid number of input channels: {num_input_channels}"
        self.num_output_classes = num_output_classes
        self.num_input_channels = num_input_channels
        self.input_key, self.label_key, self.ignore_index = input_key, label_key, ignore_index
        if metrics is None or not metrics:
            metrics = dict(  # default: add a simple classification metric
                accuracy=dict(
                    _target_="torchmetrics.classification.accuracy.Accuracy",
                    task="multiclass" if self.num_output_classes >= 2 else "binary",
                    num_classes=self.num_output_classes,
                    ignore_index=self.ignore_index,
                ),
            )
        assert isinstance(metrics, (dict, omegaconf.DictConfig)), f"incompatible metrics type: {type(metrics)}"
        self._metrics_config = metrics
        if save_hyperparams:
            # this line allows us to access hparams with `self.hparams` + auto-stores them in checkpoints
            self.save_hyperparameters(logger=False)  # logger=False since we don't need duplicated logs
        super().__init__(
            optimization=optimization,
            **kwargs,
        )
        if isinstance(encoder, (dict, omegaconf.DictConfig)):
            encoder = hydra.utils.instantiate(encoder)
        assert isinstance(encoder, torch.nn.Module), f"incompatible encoder type: {type(encoder)}"
        self.encoder = encoder
        if freeze_encoder:  # freeze all parameters in the encoder, after loading it
            for param in self.encoder.parameters():
                param.requires_grad = False
        if head is not None:  # if none, we will just not use it, and return encoder logits directly
            if isinstance(head, (dict, omegaconf.DictConfig)):
                head = hydra.utils.instantiate(head)
            assert isinstance(head, torch.nn.Module), f"incompatible head type: {type(head)}"
        self.head = head
        if loss_fn is not None:  # if none, user will have to override generic_step to provide their own
            if isinstance(loss_fn, (dict, omegaconf.DictConfig)):
                loss_fn = hydra.utils.instantiate(loss_fn)
            assert isinstance(loss_fn, torch.nn.Module), f"incompatible loss_fn type: {type(loss_fn)}"
        self.loss_fn = loss_fn
        self.example_input_array = None  # this is automatically used by pytorch lightning when not None
        if example_image_shape is not None and example_image_shape:
            self._create_example_input_array(  # for easier tracing/profiling; fake tensors for 'forward'
                **{
                    self.input_key: torch.randn(4, self.num_input_channels, *example_image_shape),
                    "batch_size": 4,
                },
            )

    def configure_metrics(self) -> torchmetrics.MetricCollection:
        """Configures and returns the metric objects to update when given predictions + labels."""
        metrics = hydra.utils.instantiate(self._metrics_config)
        assert isinstance(metrics, (dict, omegaconf.DictConfig)), f"invalid metric dict type: {type(metrics)}"
        if isinstance(metrics, omegaconf.DictConfig):
            metrics = omegaconf.OmegaConf.to_container(
                cfg=metrics,
                resolve=True,
                throw_on_missing=True,
            )
        assert all([isinstance(k, str) for k in metrics.keys()])
        assert all([isinstance(v, torchmetrics.Metric) for v in metrics.values()])
        return torchmetrics.MetricCollection(metrics)

    def forward(self, batch: ssl4rs.data.BatchDictType) -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        assert self.input_key in batch, f"missing mandatory '{self.input_key}' tensor from batch"
        input_tensor = batch[self.input_key]
        assert input_tensor.ndim >= 2
        batch_size, tensor_shape = input_tensor.shape[0], input_tensor.shape[1:]
        assert batch_size == ssl4rs.data.get_batch_size(batch)
        embed = self.encoder(input_tensor)
        assert embed.ndim >= 2 and embed.shape[0] == batch_size
        embed = torch.flatten(embed, start_dim=1)
        if self.head is not None:
            logits = self.head(embed)
        else:
            logits = embed
        assert logits.ndim == 2 and logits.shape == (batch_size, self.num_output_classes)
        return logits

    def _generic_step(
        self,
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
    ) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Runs a generic version of the forward + evaluation step for the train/valid/test loops.

        In comparison with the regular `forward()` function, this function will compute the loss and
        return multiple outputs used to update the metrics based on the assumption that the batch
        dictionary also contains info about the target labels. This means that it should never be
        used in production, as we would then try to access labels that do not exist.

        This generic+default implementation will break if the forward pass returns more than a
        single prediction tensor, or if the target labels need to be processed or transformed in any
        fashion before being sent to the loss.
        """
        assert self.loss_fn is not None, "missing impl in derived class, no loss function defined!"
        preds = self(batch)  # this will call the 'forward' implementation above and return preds
        assert self.label_key in batch, f"missing mandatory '{self.label_key}' tensor from batch"
        target = batch[self.label_key]
        loss = self.loss_fn(preds, target)
        return {
            "loss": loss,  # mandatory for training loop, optional for validation/testing
            "preds": preds.detach(),  # used in metrics, logging, and potentially even returned to user
            "targets": target,  # so that metric update functions have access to the tensor itself
            ssl4rs.data.batch_size_key: ssl4rs.data.get_batch_size(batch),  # so that logging can use it
        }

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
        """Renders and logs specific samples from the current batch using available loggers.

        Note: by default, the base class has no idea how to treat class labels other than by just
        displaying their index. Derived classes are strongly suggested to reimplement this
        function, but it is not actually required in order to just train a model.
        """
        assert len(sample_idxs) == len(sample_ids) and len(sample_idxs) > 0
        # we'll render the input tensors with their IDs, predicted, and target labels underneath
        pred_idxs, target_idxs = torch.argmax(outputs["preds"], dim=1), outputs["targets"]
        outputs = []
        for sample_idx, sample_id in zip(sample_idxs, sample_ids):
            input_tensor = batch[self.input_key][sample_idx].cpu()
            image = ssl4rs.utils.drawing.get_displayable_image(input_tensor)
            estimated_zoom_factor = 300 / image.shape[0]  # target size = 300 px height
            image = ssl4rs.utils.drawing.resize_nn(image, zoom_factor=estimated_zoom_factor)
            image = ssl4rs.utils.drawing.add_subtitle_to_image(
                image=image,
                subtitle=str(sample_id),
                extra_border_size=10,
                scale=1.35,  # this is a bit arbitrary, just derive the function if needs fixing
            )
            sstr = f"Predicted: {pred_idxs[sample_idx]}, Target: {target_idxs[sample_idx]}"
            image = ssl4rs.utils.drawing.add_subtitle_to_image(
                image=image,
                subtitle=sstr,
                extra_subtitle_padding=4,
                scale=1.5,  # this is a bit arbitrary, just derive the function if needs fixing
            )
            self._log_rendered_image(image, key=f"{loop_type}/{sample_id}")
            outputs.append(image)
        return outputs


class GenericSegmenter(GenericClassifier):
    """Example of LightningModule used for image segmentation tasks.

    This class is derived from the framework's base model interface, which is itself derived from
    the base image classifier interface. The input data and classification label attributes
    required to ingest and evaluate predictions are assumed to be specified via keys in the
    loaded batch dictionaries. The exact keys should be specified to the constructor of this class.

    Note that we do not constrain the type of model backbone that can be used here; any should
    work as long as they give an interface that is similar to the PyTorch standard, i.e. they are
    based on the `torch.nn.Module` interface. We also skip the `head` attribute of the base
    classifier interface, and just use the `encoder` attribute to store the entire encoder+decoder.

    For more information on the role and responsibilities of the LightningModule, see:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

    For more information on the base class, see:
        `ssl4rs.models.classif.base.GenericClassifier`
        `ssl4rs.models.utils.BaseModel`
    """

    @property
    def model(self):
        """Returns the entire (encoder+decoder) model.

        Note that the encoder attribute is identical to this `model` attribute; this is because
        we are using a common base class to avoid copy-pasting some identical code.
        """
        return self.encoder

    def __init__(
        self,
        model: TorchModuleOrDictConfig,
        loss_fn: typing.Optional[TorchModuleOrDictConfig],
        metrics: ssl4rs.utils.DictConfig,
        optimization: typing.Optional[ssl4rs.utils.DictConfig],
        num_output_classes: int,
        num_input_channels: int,
        input_key: typing.AnyStr = "input",
        label_key: typing.AnyStr = "label",
        ignore_index: typing.Optional[int] = None,
        example_image_shape: typing.Optional[typing.Tuple[int, int]] = (256, 256),  # height, width
        save_hyperparams: bool = True,  # turn this off in derived classes
        **kwargs,
    ):
        """Initializes the LightningModule and its submodules, loss, metrics, and optimizer.

        See the documentation of the base classes for more information on args:
            `ssl4rs.models.classif.base.GenericClassifier`
            `ssl4rs.models.utils.BaseModel`
        """
        if save_hyperparams:
            # this line allows us to access hparams with `self.hparams` + auto-stores them in checkpoints
            self.save_hyperparameters(logger=False)  # logger=False since we don't need duplicated logs
        super().__init__(
            encoder=model,
            head=None,
            loss_fn=loss_fn,
            metrics=metrics,
            optimization=optimization,
            num_output_classes=num_output_classes,
            num_input_channels=num_input_channels,
            input_key=input_key,
            label_key=label_key,
            ignore_index=ignore_index,
            example_image_shape=example_image_shape,
            save_hyperparams=False,
            **kwargs,
        )

    def forward(self, batch: ssl4rs.data.BatchDictType) -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        assert self.input_key in batch, f"missing mandatory '{self.input_key}' tensor from batch"
        input_tensor = batch[self.input_key]
        assert input_tensor.ndim == 4, "unexpected 2D image tensor shape (should be BxCxHxW)"
        batch_size, ch, h, w = input_tensor.shape
        assert batch_size == ssl4rs.data.get_batch_size(batch)
        assert ch == self.num_input_channels
        logits = self.model(input_tensor)
        assert isinstance(logits, torch.Tensor)
        assert logits.ndim == 4, "unexpected 2d pred shape (should be BxCxHxW)"
        assert logits.shape[0] == batch_size and logits.shape[1] == self.num_output_classes
        return logits

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
        """Renders and logs specific samples from the current batch using available loggers.

        Note: only available when using binary classification models/masks.
        """
        if self.num_output_classes != 2:
            return None  # not clear how to render here, let's let derived classes handle it
        assert len(sample_idxs) == len(sample_ids) and len(sample_idxs) > 0
        batch_size = ssl4rs.data.get_batch_size(batch)
        preds, targets = outputs["preds"], outputs["targets"]
        assert targets.ndim == 3 and targets.shape[0] == batch_size and targets.dtype == torch.long
        tensor_shape = targets.shape[1:]
        assert preds.ndim == 4 and preds.shape == (batch_size, self.num_output_classes, *tensor_shape)
        # we'll render the input tensors with a prediction mask and target mask side-by-side
        outputs = []
        for sample_idx, sample_id in zip(sample_idxs, sample_ids):
            input_tensor = batch[self.input_key][sample_idx].cpu()
            assert input_tensor.ndim == 3
            assert input_tensor.shape == (self.num_input_channels, *tensor_shape)
            input_image = ssl4rs.utils.drawing.get_displayable_image(input_tensor)
            # note: line below assumes that 2nd class (2nd channel) is the 'positive' class
            pred_prob = torch.softmax(preds[sample_idx], dim=0)[1].cpu().numpy()
            pred_image = cv.cvtColor((pred_prob * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
            target_mask = targets[sample_idx]
            dontcare_mask = torch.logical_and(target_mask != 0, target_mask != 1).cpu().numpy()
            target_mask = (target_mask == 1).cpu().numpy().astype(np.uint8) * 255
            target_mask[dontcare_mask] = 128
            target_image = cv.cvtColor(target_mask, cv.COLOR_GRAY2BGR)
            output_image = cv.hconcat([input_image, pred_image, target_image])
            self._log_rendered_image(output_image, key=f"{loop_type}/{sample_id}")
            outputs.append(output_image)
        return outputs
