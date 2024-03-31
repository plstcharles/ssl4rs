import typing

import cv2 as cv
import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional
import transformers

import ssl4rs.data
import ssl4rs.utils
from ssl4rs.models.classif.base import GenericSegmenter

if typing.TYPE_CHECKING:
    from ssl4rs.models.classif.base import TorchModuleOrDictConfig

logger = ssl4rs.utils.logging.get_logger(__name__)


class HFSegmenter(GenericSegmenter):
    """Wraps a huggingface model to handle image segmentation (with preprocessing).

    For now, only supports semantic segmentation. The 'forward' of the model is assumed to expect
    the exact kwargs provided as output by the preprocessor object.

    For more information on the role and responsibilities of the LightningModule, see:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

    For more information on the base class, see:
        `ssl4rs.models.classif.base.GenericSegmenter`
        `ssl4rs.models.utils.BaseModel`
    """

    def __init__(
        self,
        model: "TorchModuleOrDictConfig",
        preprocessor: "TorchModuleOrDictConfig",
        preproc_data_key: str = "preproc_data",
        example_image_shape: typing.Tuple[int, int] = (320, 320),  # height, width
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
            model=model,
            example_image_shape=example_image_shape,
            save_hyperparams=False,
            **kwargs,
        )
        self.preproc_data_key = preproc_data_key
        if isinstance(preprocessor, (dict, omegaconf.DictConfig)):
            preprocessor = hydra.utils.instantiate(preprocessor)
        self.preprocessor = preprocessor

    @property
    def example_input_array(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """Updates the Lightning Module's "example input array" with customized contents.

        When defined, this attribute is used internally by Lightning to offer lots of small
        debugging/logging features, but remains optional. It is assumed to be an example input
        that the model can process directly using its `forward` call.
        """
        return dict(
            batch={
                self.input_key: np.random.randint(
                    low=0,
                    high=255,
                    size=(
                        self._example_batch_size,
                        self.num_input_channels,
                        *self._example_image_shape,
                    ),
                    dtype=np.uint8,
                ),
                # special thing for the maskformer: it needs to preprocess label data too
                self.label_key: np.zeros(
                    shape=(self._example_batch_size, *self._example_image_shape),
                    dtype=np.int64,
                ),
                ssl4rs.data.batch_size_key: self._example_batch_size,
            }
        )

    def forward(self, batch: ssl4rs.data.BatchDictType) -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        if self.preproc_data_key not in batch:
            logger.warning("missing preprocessed data in batch, extracting now (slower!)")
            preproc_output = self.preprocessor(batch[self.input_key], return_tensors="pt")
            # assume the only interesting thing in there is the pixel_values tensor
            if "pixel_values" in preproc_output:
                preproc_output["pixel_values"] = preproc_output["pixel_values"].to(self.device)
            batch[self.preproc_data_key] = preproc_output
        else:
            preproc_output = batch[self.preproc_data_key]
        outputs = self.model(**preproc_output)
        assert "loss" not in outputs or outputs["loss"] is None, "unexpected loss computation?"
        assert "logits" in outputs and outputs["logits"] is not None, "unexpected null logits?"
        logits = outputs["logits"]
        return logits

    def _render_and_log_samples(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
        sample_idxs: typing.List[int],
        sample_ids: typing.List[typing.Hashable],
        outputs: ssl4rs.data.BatchDictType,
        dataloader_idx: int = 0,
    ) -> typing.Any:
        """Renders and logs specific samples from the current batch using available loggers."""
        if self.num_output_classes == 2:
            class_colormap = [[0, 0, 0], [255, 255, 255]]
        else:
            class_colormap = [np.random.randint(0, 256, size=3) for _ in range(self.num_output_classes)]
        assert len(sample_idxs) == len(sample_ids)
        batch_size = ssl4rs.data.get_batch_size(batch)
        preds, targets = outputs["preds"], outputs["targets"]
        assert targets.ndim == 3 and targets.shape[0] == batch_size and targets.dtype == torch.long
        tensor_shape = targets.shape[1:]
        if preds.dtype == torch.long:
            assert preds.ndim == 3 and preds.shape == (batch_size, *tensor_shape)
        else:
            assert preds.ndim == 4 and preds.shape == (batch_size, self.num_output_classes, *tensor_shape)
            preds = torch.argmax(preds, dim=1)
        # we'll render the input tensors with a prediction mask and target mask side-by-side
        outputs = []
        for sample_idx, sample_id in zip(sample_idxs, sample_ids):
            input_tensor = batch[self.input_key][sample_idx].cpu()
            assert input_tensor.ndim == 3
            assert input_tensor.shape == (self.num_input_channels, *tensor_shape)
            input_image = ssl4rs.utils.drawing.get_displayable_image(input_tensor)
            curr_preds = preds[sample_idx].cpu().numpy()
            curr_target = targets[sample_idx].cpu().numpy()
            preds_image = np.zeros(shape=(*tensor_shape, 3), dtype=np.uint8)
            target_image = np.zeros(shape=(*tensor_shape, 3), dtype=np.uint8)
            for class_idx, class_color in enumerate(class_colormap):
                preds_image[curr_preds == class_idx, :] = class_color
                target_image[curr_target == class_idx, :] = class_color
            dontcare_mask = (targets[sample_idx] == self.ignore_index).cpu().numpy()
            target_image[dontcare_mask, :] = 128
            output_image = cv.hconcat([input_image, preds_image, target_image])
            self._log_rendered_image(output_image, key=f"{loop_type}/{sample_id}")
            outputs.append(output_image)
        return outputs


def create_collate_fn_with_preproc(
    image_key: str,
    preproc_data_key: str,
    pad_tensor_names_and_values: typing.Optional[typing.Dict[str, typing.Any]],
    pad_to_shape: typing.Optional[typing.Tuple[int, int]],
    keys_to_batch_manually: typing.Sequence[str],
    preprocessor: transformers.AutoImageProcessor,
) -> typing.Callable[[typing.List[ssl4rs.data.BatchDictType]], ssl4rs.data.BatchDictType]:
    """Creates a callable object that can be used as a custom collate for most HF-based models."""

    def _collate_with_preprocessor(
        batches: typing.List[ssl4rs.data.BatchDictType],
    ) -> ssl4rs.data.BatchDictType:
        for batch in batches:
            ssl4rs.data.transforms.pad.pad_arrays_in_batch(
                batch=batch,
                pad_tensor_names_and_values=pad_tensor_names_and_values,
                pad_to_shape=pad_to_shape,
            )
        output_without_main_tensors = ssl4rs.data.default_collate(
            batches=batches,
            keys_to_batch_manually=keys_to_batch_manually,
            keys_to_ignore=[image_key],
        )
        image_data = [batch[image_key] for batch in batches]
        preproc_output = preprocessor(  # noqa
            images=image_data,
            return_tensors="pt",
        )
        output = {
            **output_without_main_tensors,
            preproc_data_key: dict(**preproc_output),
            image_key: image_data,
        }
        return output

    return _collate_with_preprocessor
