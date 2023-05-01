"""Implements a simple MNIST image classification module based on Lightning."""

import typing

import torch
import torch.nn.functional

import ssl4rs.data
import ssl4rs.models.classif.base
import ssl4rs.utils

logger = ssl4rs.utils.logging.get_logger(__name__)


class MNISTClassifier(ssl4rs.models.classif.base.GenericClassifier):
    """Example of LightningModule used for MNIST image classification.

    This class is derived from the framework's base model interface and it implements all the extra
    goodies required for automatic rendering/logging of predictions.

    For more information on the role and responsibilities of the LightningModule, see:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

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
        # this line allows us to access hparams with `self.hparams` + auto-stores them in checkpoints
        self.save_hyperparameters(logger=False)  # logger=False since we don't need duplicated logs
        super().__init__(
            encoder=encoder_config,
            head=head_config,
            loss_fn=loss_config,
            num_classes=10,
            optimization=optimization,
            input_key="data",
            label_key="target",
        )
        self._create_example_input_array(  # for easier tracing/profiling; fake tensors for 'forward'
            data=torch.randn(4, 1, 28, 28),
            batch_size=4,
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
