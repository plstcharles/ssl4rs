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

#TODO: CHW -> HWC?????

class Convert4BandTo3Band(torch.nn.Module):
    def forward(self, tensor):
        # Assuming tensor shape is [C, H, W] and C=4 for BGR+NIR channels
        R, G, B, NIR = tensor[2], tensor[1], tensor[0], tensor[3]  # Reorder to RGBN
        band1 = (R + G) / 2
        band2 = (R + NIR) / 2
        band3 = (G + B) / 2
        return torch.stack([band1, band2, band3], dim=0)  # Stack to form a new tensor

class TorchTransforms(torch.nn.Module):
    def __init__(self):
        super(TorchTransforms, self).__init__()
        # Define the transformations here; assuming tensors as input for now
        # Note: Some operations like RandomCrop are not directly applicable to tensors
        # without conversion to PIL Images
        self.IMAGE_DATA_KEY = 4
        self.mean = tensor_normalization_stats["image_data"][self.IMAGE_DATA_KEY]["mean"]
        self.std = image_data_mean_values = tensor_normalization_stats["image_data"][self.IMAGE_DATA_KEY]["std"]

        self.transform = transforms.Compose([
             Convert4BandTo3Band(), # Custom transformation to convert 4-band to 3-band
            transforms.ToPILImage(),  # Convert tensors to PIL Images to apply certain transforms
            transforms.RandomCrop(224),  # Random crop to 224x224
            transforms.ToTensor(),  # Convert back to tensor
            #TODO: convert 4band image to 3band derivative with the following formula
            # of BGR+NIR -> Band 1: avg(R,G), Band 2: avg(R, NIR), Band 3: avg(G,B)
            #? - in 3band derivative, original image values cannot be inverted.
            transforms.Normalize(mean=self.mean, std=self.std)
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