"""Implements a generic classifier module based on Lightning."""

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
TorchModuleOrDictConfig = typing.Union[torch.nn.Module, ssl4rs.utils.DictConfig]


class GenericClassifier(ssl4rs.models.BaseModel):
    """Example of LightningModule used for classification tasks.

    This class is derived from the framework's base model interface, and it implements all the
    extra goodies required for automatic rendering/logging of predictions. The input data and
    classification label attributes required to ingest and evaluate predictions are assumed to be
    specified via keys in the loaded batch dictionaries. The exact keys should be specified to the
    constructor of this class.

    For more information on the role and responsibilities of the LightningModule, see:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

    For more information on the base class, see:
        `ssl4rs.models.utils.BaseModel`
    """

    def __init__(
        self,
        encoder: TorchModuleOrDictConfig,
        head: typing.Optional[TorchModuleOrDictConfig],
        loss_fn: TorchModuleOrDictConfig,
        num_classes: int,
        optimization: typing.Optional[ssl4rs.utils.DictConfig] = None,
        input_key: typing.AnyStr = "input",
        label_key: typing.AnyStr = "label",
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
                corresponds to the backbone encoder of the model. If a config is provided, it
                will be used to instantiate the classifier via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object. If nothing is provided, we will
                assume that the backbone encoder already possesses a classifier, and will
                compute the loss directly on the backbone's output.
            loss_fn: dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the loss function of the model. If a config is provided, it
                will be used to instantiate the loss function via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object.
            num_classes: number of unique classes (categories) to be predicted.
            optimization: dict-based configuration that can be used to instantiate the
                optimizer used by the trainer. This config is assumed to be formatted according
                to Lightning's format. See the base class's `configure_optimizers` for more info.
            input_key: key used to fetch the input data tensor from the loaded batch dictionaries.
            label_key: key used to fetch the class label tensor from the loaded batch dictionaries.
        """
        assert isinstance(num_classes, int) and num_classes > 0, f"invalid num_classes: {num_classes}"
        self.num_classes = num_classes
        self.input_key, self.label_key = input_key, label_key
        super().__init__(**kwargs)
        if isinstance(encoder, (dict, omegaconf.DictConfig)):
            encoder = hydra.utils.instantiate(encoder)
        assert isinstance(encoder, torch.nn.Module), f"incompatible encoder type: {type(encoder)}"
        self.encoder = encoder
        if isinstance(head, (dict, omegaconf.DictConfig)) and head:
            head = hydra.utils.instantiate(head)
        assert head is None or isinstance(head, torch.nn.Module), f"incompatible head type: {type(head)}"
        self.head = head
        if isinstance(loss_fn, (dict, omegaconf.DictConfig)):
            loss_fn = hydra.utils.instantiate(loss_fn)
        assert isinstance(loss_fn, torch.nn.Module), f"incompatible loss_fn type: {type(loss_fn)}"
        self.loss_fn = loss_fn
        assert optimization is None or isinstance(
            optimization, (dict, omegaconf.DictConfig)
        ), f"incompatible optimization config type: {type(optimization)}"
        self.optim_config = optimization  # this will be instantiated later, if we actually need it

    def configure_metrics(self) -> torchmetrics.MetricCollection:
        """Configures and returns the metric objects to update when given predictions + labels."""
        return torchmetrics.MetricCollection(
            dict(
                accuracy=torchmetrics.classification.accuracy.Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                ),
            )
        )

    def forward(self, batch: typing.Dict[typing.AnyStr, typing.Any]) -> torch.Tensor:
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
        assert logits.ndim == 2 and logits.shape == (batch_size, self.num_classes)
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
        assert self.label_key in batch, f"missing mandatory '{self.label_key}' tensor from batch"
        target = batch[self.label_key]
        loss = self.loss_fn(preds, target)
        return {
            "loss": loss,  # mandatory for training loop, optional for validation/testing
            "preds": preds,  # used in metrics, logging, and potentially even returned to user
            "target": target,  # so that metric update functions have access to the tensor itself
            "batch_size": ssl4rs.data.get_batch_size(batch),  # so that logging functions can use it
        }
