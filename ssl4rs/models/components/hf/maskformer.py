"""Implements a huggingface MaskFormer wrapper to handle data unpacking + logging."""

import typing

import torch
import torch.nn.functional
import transformers

import ssl4rs.data
import ssl4rs.utils
from ssl4rs.models.components.hf.base import HFSegmenter

logger = ssl4rs.utils.logging.get_logger(__name__)


def create_maskformer_model_from_pretrained(
    id2label: typing.Dict[str, str],
    freeze_encoder: bool = False,
    **kwargs,
) -> transformers.MaskFormerForInstanceSegmentation:
    """Creates and returns a maskformer model to use for instance segmentation.

    This function wraps the original model creator in order to fix the DictConfig issue with the
    `id2label` argument.
    """
    id2label = {int(k): label for k, label in id2label.items()}
    hf_model = transformers.MaskFormerForInstanceSegmentation.from_pretrained(
        id2label=id2label,
        **kwargs,
    )
    if freeze_encoder:  # freeze all parameters in the encoder, after loading it
        for param in hf_model.model.pixel_level_module.encoder.parameters():
            param.requires_grad = False
    return hf_model


def create_custom_collate_with_preproc(
    image_key: str,
    mask_key: str,
    preproc_data_key: str,
    pad_tensor_names_and_values: typing.Optional[typing.Dict[str, typing.Any]],
    pad_to_shape: typing.Optional[typing.Tuple[int, int]],
    keys_to_batch_manually: typing.Sequence[str],
    preprocessor: transformers.MaskFormerImageProcessor,
) -> typing.Callable[[typing.List[ssl4rs.data.BatchDictType]], ssl4rs.data.BatchDictType]:
    """Creates a callable object that can be used as a custom collate for maskformer models."""

    def _custom_collate_with_preprocessor(
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
            keys_to_ignore=[image_key, mask_key],
        )
        preproc_output = preprocessor(
            images=[batch[image_key] for batch in batches],
            segmentation_maps=[batch[mask_key] for batch in batches],
            return_tensors="pt",
        )
        output = {
            **output_without_main_tensors,
            preproc_data_key: dict(**preproc_output),
            image_key: [batch[image_key] for batch in batches],
            mask_key: [batch[mask_key] for batch in batches],
        }
        return output

    return _custom_collate_with_preprocessor


class MaskFormerSegmenter(HFSegmenter):
    """Performs image segmentation (with preprocessing) via HF's MaskFormer model."""

    def __init__(
        self,
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
            loss_fn=None,  # the maskformer provides its own loss
            save_hyperparams=False,
            **kwargs,
        )
        assert self.model.config.num_labels == self.num_output_classes
        assert self.model.config.num_labels == len(self.model.config.id2label)
        assert self.model.config.backbone_config.in_channels == self.num_input_channels

    @property
    def example_input_array(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """Updates the Lightning Module's "example input array" with customized contents.

        When defined, this attribute is used internally by Lightning to offer lots of small
        debugging/logging features, but remains optional. It is assumed to be an example input
        that the model can process directly using its `forward` call.
        """
        return dict(
            batch={
                self.input_key: torch.randn(
                    self._example_batch_size,
                    self.num_input_channels,
                    *self._example_image_shape,
                ),
                # special thing for the maskformer: it needs to preprocess label data too
                self.label_key: torch.zeros(
                    (self._example_batch_size, *self._example_image_shape),
                    dtype=torch.long,
                ),
                ssl4rs.data.batch_size_key: self._example_batch_size,
            }
        )

    def forward(self, batch: ssl4rs.data.BatchDictType) -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        if self.preproc_data_key not in batch:
            logger.warning("missing preprocessed data in batch, extracting now (slower!)")
            preproc_data = self.preprocessor(
                images=batch[self.input_key],
                segmentation_maps=batch[self.label_key],
                return_tensors="pt",
            )
            expected_keys = ["pixel_values", "pixel_mask", "class_labels", "mask_labels"]
            assert all([k in preproc_data for k in expected_keys])
            preproc_data["pixel_values"] = preproc_data["pixel_values"].to(self.device)
            preproc_data["pixel_mask"] = preproc_data["pixel_mask"].to(self.device)
            preproc_data["class_labels"] = [t.to(self.device) for t in preproc_data["class_labels"]]
            preproc_data["mask_labels"] = [t.to(self.device) for t in preproc_data["mask_labels"]]
        else:
            preproc_data = batch[self.preproc_data_key]
        outputs = self.model(**preproc_data)
        assert "loss" in outputs
        return outputs

    def _generic_step(
        self,
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
    ) -> ssl4rs.data.BatchDictType:
        """Runs a generic version of the forward + evaluation step for the train/valid/test loops.

        We override the base class step implementation, as the MaskFormer provides its own loss.
        """
        outputs = self(batch)  # noqa; this will call the 'forward' implementation above
        assert "loss" in outputs
        orig_input_shapes = [t.shape[1:] for t in batch[self.input_key]]  # CxHxW => (h,w)
        target = batch[self.label_key]  # this is the 'original' mask, before preprocessing
        assert all([t.shape == s for t, s in zip(target, orig_input_shapes)])
        preds = self.preprocessor.post_process_semantic_segmentation(
            outputs,
            target_sizes=orig_input_shapes,
        )
        return {
            **{k: v for k, v in outputs.items()},
            "targets": torch.stack(target),  # so that metric update functions have access to the tensor itself
            "preds": torch.stack(preds).detach(),  # used in metrics, logging, and potentially even returned to user
            ssl4rs.data.batch_size_key: ssl4rs.data.get_batch_size(batch),  # so that logging can use it
        }
