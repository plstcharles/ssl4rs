import typing

import torch


class GrabImageFromStack(torch.nn.Module):
    """Custom transformation to grab the n-th (at index n) image from a stack of N images.

    Args:
        input_stack_key (str): The key to access the input stack of images in the batch dictionary.
        output_image_key (str): The key to store the grabbed image in the batch dictionary.
        image_index (int, optional): The index of the image to grab from the stack. Defaults to 0.
    """

    def __init__(
        self,
        input_stack_key: str,
        output_image_key: str,
        image_index: int = 0,
    ):
        """Initializes transform settings."""
        super().__init__()
        self.input_stack_key = input_stack_key
        self.output_image_key = output_image_key
        self.image_index = image_index

    def forward(self, batch: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """ Takes a batch (Dict[str, Any]), grabs the image from the stack and re-inserts it into the batch dictionary."""
        assert self.input_stack_key in batch, f"missing key in batch dict: {self.input_stack_key}"
        image_data = batch[self.input_stack_key]
        assert image_data.ndim > 1  # should be something like (N, ...channels, height, width)
        image_data = image_data[self.image_index]
        batch[self.output_image_key] = image_data
        return batch
