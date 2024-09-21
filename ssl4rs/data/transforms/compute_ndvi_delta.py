import typing
from typing import List

import torch
import torchvision.transforms as transforms


class GetAvgNdviDeltaFromStack(torch.nn.Module):
    """Custom transformation to compute the mean of ndvi delta across a stack of images

        Args:
            input_stack_key (str): The key to access the input stack of images in the batch dictionary.
            output_image_key (str): The key to store the grabbed image in the batch dictionary.
            image_index (int, optional): The index of the image to grab from the stack. Defaults to 0.
        """
    # todo: presumably num channels needs to be fixed
    def __init__(
            self,
            input_stack_key: str,
            output_image_key: str,
    ):
        """Initializes transform settings."""
        super().__init__()
        self.input_stack_key = input_stack_key
        self.output_image_key = output_image_key

    def forward(self, batch: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """ Takes a batch (Dict[str, Any]), grabs the image from the stack and re-inserts it into the batch dictionary."""
        assert self.input_stack_key in batch, f"missing key in batch dict: {self.input_stack_key}"
        image_data = batch[self.input_stack_key]

        # stack assumed to be list
        image_ndvi_n = [self.image_to_ndvi(image) for image in image_data]
        image_deltas_n = self.compute_image_deltas(image_ndvi_n)
        if image_deltas_n: 
            avg_image_delta = self.compute_nan_mean(image_deltas_n)
        delta_composite = [avg_image_delta, image_deltas_n[0], image_ndvi_n[0]] if image_deltas_n else [image_ndvi_n[0], image_ndvi_n[0], image_ndvi_n[0]]
        delta_composite_norm = self.channel_wise_normalization(delta_composite)
        mixture_of_delta = torch.stack(delta_composite_norm, dim=0)
        nan_mask = mixture_of_delta.isnan()
        mixture_of_delta[nan_mask] = 0.0

        batch[self.output_image_key] = mixture_of_delta
        return batch

    def channel_wise_normalization(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        normalized_channels = []

        # Iterate through each channel
        for channel in tensors:
            valid_mask = ~torch.isnan(channel)

            # Compute the mean and std for the current channel
            mean = channel[valid_mask].mean()
            std = channel[valid_mask].std()

            # Normalize the current channel (z-score normalization)
            normalized_channel = (channel - mean) / std

            # Append normalized channel to the list
            normalized_channels.append(normalized_channel)

        # Stack the normalized channels back together
        return normalized_channels

    def compute_nan_mean(self, image_deltas_n: List[torch.Tensor]) -> torch.Tensor:
        image_deltas = torch.stack(image_deltas_n, dim=0)
        image_deltas_mean = torch.nanmean(image_deltas, dim=0)
        return image_deltas_mean

    def image_to_ndvi(self, image: torch.Tensor) -> torch.Tensor:
        B = image[0]
        G = image[1]
        R = image[2]
        NIR = image[3]

        ndvi = (NIR - R) / (R + NIR)

        return ndvi

    def compute_image_deltas(self, image_n: List[torch.Tensor]) -> List[torch.Tensor]:

        image_delta_n = []
        for image_idx in range(0, len(image_n) - 1):
            image_delta_n.append(torch.abs(image_n[image_idx] - image_n[image_idx+1]))
        return image_delta_n


if __name__ == '__main__':
    transform = GetAvgNdviDeltaFromStack(input_stack_key='image_data', output_image_key='avg_ndvi_delta')
    mock_image_data = [torch.ones((4, 320, 320)) for _ in range(5)]
    mock_image_data = [tensor * idx*2*2 for idx, tensor in enumerate(mock_image_data)]

    batch = {}
    batch['image_data'] = mock_image_data

    transform_output = transform(batch)
    assert isinstance(transform_output['avg_ndvi_delta'], torch.Tensor)
