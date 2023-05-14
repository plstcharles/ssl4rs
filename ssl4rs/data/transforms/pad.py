import typing

import torch
import torch.nn.functional

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType


def pad_if_needed(
    input_tensor: torch.Tensor,
    min_height: int,
    min_width: int,
    centered: bool = False,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Pads a tensor to a minimum height and width.

    This function pads the last two dimensions (height and width) of an input tensor
    to reach a minimum size, if they are smaller than the given minimum height and width.
    Padding can be applied symmetrically (centered) or asymmetrically (to the bottom and right).

    Args:
        input_tensor (torch.Tensor): The tensor to pad. It should have the shape [..., H, W].
        min_height (int): The minimum height the tensor should have after padding.
        min_width (int): The minimum width the tensor should have after padding.
        centered (bool, optional): Whether to distribute the padding evenly on both sides
            of each dimension. Default is False, which pads at the bottom and right.
        *args: Variable length argument list to be passed to the `torch.nn.functional.pad` function.
        **kwargs: Arbitrary keyword arguments to be passed to the `torch.nn.functional.pad` function.

    Returns:
        torch.Tensor: The padded tensor, with the same number of dimensions as the input,
            and with height and width at least `min_height` and `min_width`, respectively.
    """
    height, width = input_tensor.shape[-2:]
    pad_height = max(0, min_height - height)
    pad_width = max(0, min_width - width)
    if centered:
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
    else:
        pad_top = pad_left = 0
        pad_bottom = pad_height
        pad_right = pad_width
    if pad_height > 0 or pad_width > 0:
        input_tensor = torch.nn.functional.pad(
            input=input_tensor,
            pad=(pad_left, pad_right, pad_top, pad_bottom),
            *args,
            **kwargs,
        )
    return input_tensor


class PadIfNeeded(torch.nn.Module):
    """Module-based implementation of the `pad_if_needed` function.

    Args:
        target_key (str): The key for the batch element that should be padded.
        min_height (int): The minimum height the tensor should have after padding.
        min_width (int): The minimum width the tensor should have after padding.
        centered (bool, optional): Whether to distribute the padding evenly on both sides
            of each dimension. Default is False, which pads at the bottom and right.
        save_prepad_shape (bool): Whether to save the original shape of the tensor in the batch
            dictionary. Default is False. If True, the shape will be saved under a key that
            corresponds to `target_key` with a `/prepad_shape` suffix.
        *args: Variable length argument list to be passed to the `torch.nn.functional.pad` function.
        **kwargs: Arbitrary keyword arguments to be passed to the `torch.nn.functional.pad` function.
    """

    def __init__(
        self,
        target_key: str,
        min_height: int,
        min_width: int,
        centered: bool = False,
        save_prepad_shape: bool = True,
        *args,
        **kwargs,
    ):
        """Initializes and validates padding op settings."""
        super().__init__()
        assert min_height > 0, f"invalid min height: {min_height}"
        assert min_width > 0, f"invalid min height: {min_width}"
        self.target_key = target_key
        self.min_height = min_height
        self.min_width = min_width
        self.centered = centered
        self.save_prepad_shape = save_prepad_shape
        self.args = args
        self.kwargs = kwargs

    def forward(self, batch: "BatchDictType") -> "BatchDictType":
        """Forward pass of the PadIfNeeded module.

        Args:
            batch: the loaded batch dictionary that contains a tensor to be padded. This tensor
                will be replaced by its updated version in-place within this dictionary.

        Returns:
            The updated batch dictionary with the padded target tensor.
        """
        input_tensor = batch[self.target_key]
        assert isinstance(input_tensor, torch.Tensor) and input_tensor.ndim >= 2
        if self.save_prepad_shape:
            batch[f"{self.target_key}/prepad_shape"] = input_tensor.shape
        output_tensor = pad_if_needed(
            input_tensor=input_tensor,
            min_height=self.min_height,
            min_width=self.min_width,
            centered=self.centered,
            *self.args,
            **self.kwargs,
        )
        batch[self.target_key] = output_tensor
        return batch
