"""Implements a huggingface DPT wrapper to handle data unpacking + logging."""

import typing

import hydra
import omegaconf
import torch
import torch.nn.functional
import transformers

import ssl4rs.utils
from ssl4rs.models.components.hf.base import HFSegmenter

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType
    from ssl4rs.models.classif.base import TorchModuleOrDictConfig

logger = ssl4rs.utils.logging.get_logger(__name__)


def create_dpt_model_from_pretrained(
    freeze_backbone: bool = False,
    freeze_neck: bool = False,
    new_head: typing.Optional["TorchModuleOrDictConfig"] = None,
    **kwargs,
) -> transformers.DPTForSemanticSegmentation:
    """Creates and returns a DPT model to use for semantic segmentation."""
    hf_model = transformers.DPTForSemanticSegmentation.from_pretrained(**kwargs)
    if freeze_backbone:  # freeze all parameters in the backbone
        for param in hf_model.dpt.parameters():
            param.requires_grad = False
    if freeze_neck:  # freeze all parameters in the neck
        for param in hf_model.neck.parameters():
            param.requires_grad = False
    if new_head is not None:
        if isinstance(new_head, (dict, omegaconf.DictConfig)):
            new_head = hydra.utils.instantiate(new_head)
        assert isinstance(new_head, torch.nn.Module), f"incompatible head type: {type(new_head)}"
        hf_model.head.head = new_head
    return hf_model


class DPTSegmenter(HFSegmenter):
    """Performs image segmentation (with preprocessing) via HF's DPT model."""

    def forward(self, batch: "BatchDictType") -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        orig_input_image = batch[self.input_key]
        if isinstance(orig_input_image, list):
            # assume these were padded and can be stacked
            orig_input_image = torch.stack(orig_input_image)
        assert orig_input_image.ndim == 4  # BxCxHxW
        orig_image_shape = orig_input_image.shape[2:]  # HxW
        logits = super().forward(batch)
        # the input image was likely rescaled from its original resolution, scale predictions back...
        preds = torch.nn.functional.interpolate(
            logits,
            size=orig_image_shape,
            mode="bicubic",
            align_corners=False,
        )
        return preds
