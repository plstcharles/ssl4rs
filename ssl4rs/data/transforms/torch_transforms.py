import typing

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

constants = '/network/projects/ai4h-disa/ssl4rs/ssl4rs/data/metadata/disa.py'

from constants import tensor_normalization_stats

# import ssl4rs.utils.imgproc
# import ssl4rs.utils.patch_coord

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType


class TorchTransforms(torch.nn.Module):
    def __init__(self):
        super(TorchTransforms, self).__init__()
        # Define the transformations here; assuming tensors as input for now
        # Note: Some operations like RandomCrop are not directly applicable to tensors
        # without conversion to PIL Images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert tensors to PIL Images to apply certain transforms
            transforms.RandomCrop(224),  # Random crop to 224x224
            transforms.ToTensor(),  # Convert back to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, batch: BatchDictType) -> BatchDictType:
        return self.transform_batch(batch)

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform(tensor)

    def transform_batch(self, batch: BatchDictType) -> BatchDictType:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                # Convert tensor to PIL Image, apply transformations, and convert back to tensor if necessary
                batch[key] = self.transform_tensor(batch[key])
        return batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"