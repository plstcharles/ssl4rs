"""Implements a huggingface MaskFormer wrapper to handle data unpacking + logging."""

import typing

import cv2 as cv
import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional
import transformers

import ssl4rs.data
import ssl4rs.models.classif
import ssl4rs.utils

logger = ssl4rs.utils.logging.get_logger(__name__)
TorchModuleOrDictConfig = typing.Union[torch.nn.Module, ssl4rs.utils.DictConfig]


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
    model = transformers.MaskFormerForInstanceSegmentation.from_pretrained(
        id2label=id2label,
        **kwargs,
    )
    if freeze_encoder:  # freeze all parameters in the encoder, after loading it
        for param in model.encoder.parameters():
            param.requires_grad = False
    return model


def create_custom_collate_with_preproc(
    image_key: str,
    mask_key: str,
    pad_tensor_names_and_values: typing.Optional[typing.Dict[str, typing.Any]],
    pad_to_shape: typing.Optional[typing.Tuple[int, int]],
    keys_to_batch_manually: typing.Sequence[typing.AnyStr],
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
        # @@@@@ TODO: update to make sure it works on non-preview-images? (i.e. NxCxHxW arrays)
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
        assert all([k not in output_without_main_tensors for k in preproc_output.keys()])
        output = {
            **output_without_main_tensors,
            **preproc_output,
            image_key: [batch[image_key] for batch in batches],
            mask_key: [batch[mask_key] for batch in batches],
        }
        return output

    return _custom_collate_with_preprocessor


class HFMaskFormerSegmenter(ssl4rs.models.classif.base.GenericSegmenter):
    """Wraps the hugginface MaskFormer model components to handle data unpacking + logging.

    For now, only supports semantic segmentation. The 'forward' of the model is assumed to have
    three arguments that correspond to the `pixel_values`, `class_labels`, and `mask_labels`
    tensors prepared by an instance of `transformers.MaskFormerImageProcessor`.

    For more information on the wrapped MaskFormer class, see:
        https://huggingface.co/docs/transformers/en/model_doc/maskformer
        https://huggingface.co/docs/transformers/en/model_doc/mask2former

    For more information on the role and responsibilities of the LightningModule, see:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

    For more information on the base class, see:
        `ssl4rs.models.classif.base.GenericSegmenter`
        `ssl4rs.models.utils.BaseModel`
    """

    def __init__(
        self,
        model: TorchModuleOrDictConfig,
        preprocessor: TorchModuleOrDictConfig,
        metrics: ssl4rs.utils.DictConfig,
        optimization: typing.Optional[ssl4rs.utils.DictConfig],
        num_output_classes: int,
        num_input_channels: int,
        orig_image_key: typing.AnyStr = "input",
        orig_mask_key: typing.AnyStr = "label",
        ignore_index: typing.Optional[int] = None,
        example_image_shape: typing.Tuple[int, int] = (256, 256),  # height, width
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
            loss_fn=None,
            metrics=metrics,
            optimization=optimization,
            num_output_classes=num_output_classes,
            num_input_channels=num_input_channels,
            input_key=orig_image_key,  # note: we actually use the "pixel_values" tensor prepared by the preprocessor
            label_key=orig_mask_key,  # note: we actually use the "class_labels" and "mask_labels" tensors
            ignore_index=ignore_index,
            example_image_shape=None,  # we override and create our own example input array (no need for this)
            save_hyperparams=False,
            **kwargs,
        )
        assert self.model.config.num_labels == self.num_output_classes
        assert self.model.config.num_labels == len(self.model.config.id2label)
        assert self.model.config.backbone_config.in_channels == self.num_input_channels
        if isinstance(preprocessor, (dict, omegaconf.DictConfig)):
            preprocessor = hydra.utils.instantiate(preprocessor)
        self.preprocessor = preprocessor

    def _create_example_input_array(self, **kwargs) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Creates a fake batch dict to be used as the example (pre-preprocessing!) input.

        The `self.example_input_array` attribute is actually used by Lightning to offer lots of
        small debugging/logging features, but remains optional. This version fills a dummy image
        and mask data batch so that the preprocessor can be applied and that its outputs can feed
        the model directly.
        """
        assert not kwargs, f"unexpected ex input array kwargs: {kwargs}"
        batch_data = {
            self.input_key: torch.randn(4, 3, 320, 320, dtype=torch.float),
            self.label_key: torch.zeros((4, 320, 320), dtype=torch.long),
        }
        self.example_input_array = dict(batch=batch_data)
        return self.example_input_array

    def forward(self, batch: ssl4rs.data.BatchDictType) -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        expected_keys = ["pixel_values", "class_labels", "mask_labels"]
        found_missing_keys = not all([k in batch for k in expected_keys])
        if found_missing_keys:
            logger.warning("missing preprocessed tensor keys in batch, extracting them now (slower!)")
            preproc_output = self.preprocessor(
                images=batch[self.input_key],
                segmentation_maps=batch[self.label_key],
                return_tensors="pt",
            )
            pixel_values = preproc_output["pixel_values"]
            class_labels = preproc_output["class_labels"]
            mask_labels = preproc_output["mask_labels"]
        else:
            pixel_values = batch["pixel_values"]
            class_labels = batch["class_labels"]
            mask_labels = batch["mask_labels"]
        batch_size = ssl4rs.data.get_batch_size(batch)
        assert pixel_values.ndim == 4 and pixel_values.shape[:2] == (batch_size, self.num_input_channels)
        assert len(class_labels) == batch_size and len(mask_labels) == batch_size
        outputs = self.model(pixel_values, class_labels=class_labels, mask_labels=mask_labels)
        assert "loss" in outputs
        return outputs

    def _generic_step(
        self,
        batch: ssl4rs.data.BatchDictType,
        batch_idx: int,
    ) -> typing.Dict[typing.AnyStr, typing.Any]:
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
        """Renders and logs specific samples from the current batch using available loggers."""
        if self.num_output_classes == 2:
            class_colormap = [[0, 0, 0], [255, 255, 255]]
        else:
            class_colormap = [np.random.randint(0, 256, size=3) for _ in range(self.num_output_classes)]
        assert len(sample_idxs) == len(sample_ids) and len(sample_idxs) > 0
        batch_size = ssl4rs.data.get_batch_size(batch)
        preds, targets = outputs["preds"], outputs["targets"]
        assert targets.ndim == 3 and targets.shape[0] == batch_size and targets.dtype == torch.long
        tensor_shape = targets.shape[1:]
        assert preds.ndim == 3 and preds.shape == (batch_size, *tensor_shape) and preds.dtype == torch.long
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
            dontcare_mask = (targets[sample_idx] == self.preprocessor.ignore_index).cpu().numpy()
            target_image[dontcare_mask, :] = 128
            output_image = cv.hconcat([input_image, preds_image, target_image])
            self._log_rendered_image(output_image, key=f"{loop_type}/{sample_id}")
            outputs.append(output_image)
        return outputs
