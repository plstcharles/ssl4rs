"""Implements a simple ConvNet PyTorch module for example/demo/debug models."""

import typing

import torch.nn

default_kernel_size = (3, 3)
default_stride = (1, 1)
default_padding = (1, 1)
default_dropout = 0.0


class SimpleConvNet(torch.nn.Module):
    """Simple convolutional neural network implementation.

    This class is meant to be used inside other PyTorch modules, or in a LightningModule.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: typing.Sequence[int],
        kernel_sizes: typing.Optional[typing.Sequence[typing.Tuple[int, int]]] = None,
        strides: typing.Optional[typing.Sequence[typing.Tuple[int, int]]] = None,
        paddings: typing.Optional[typing.Sequence[typing.Tuple[int, int]]] = None,
        dropouts: typing.Optional[typing.Sequence[float]] = None,
        with_batch_norm: bool = True,
        with_max_pool: bool = True,
        with_output_upsample: bool = False,
        head_channels: typing.Optional[int] = None,  # for a final 1x1 conv head, if needed
    ):
        """Initializes the CNN using the provided settings."""
        assert len(hidden_channels) > 0 and all([c > 0 for c in hidden_channels])
        assert kernel_sizes is None or len(kernel_sizes) == len(hidden_channels)
        assert strides is None or len(strides) == len(hidden_channels)
        assert paddings is None or len(paddings) == len(hidden_channels)
        assert dropouts is None or len(dropouts) == len(hidden_channels)
        super().__init__()
        self.model = torch.nn.Sequential(
            build_layer(
                in_channels=in_channels,
                out_channels=hidden_channels[0],
                kernel_size=kernel_sizes[0] if kernel_sizes else default_kernel_size,
                stride=strides[0] if strides else default_stride,
                padding=paddings[0] if paddings else default_padding,
                dropout=dropouts[0] if dropouts else default_dropout,
                with_batch_norm=with_batch_norm,
                with_max_pool=with_max_pool,
            ),
            *[
                build_layer(
                    in_channels=hidden_channels[idx],
                    out_channels=hidden_channels[idx + 1],
                    kernel_size=kernel_sizes[idx + 1] if kernel_sizes else default_kernel_size,
                    stride=strides[idx + 1] if strides else default_stride,
                    padding=paddings[idx + 1] if paddings else default_padding,
                    dropout=dropouts[idx + 1] if dropouts else default_dropout,
                    with_batch_norm=with_batch_norm,
                    with_max_pool=with_max_pool,
                )
                for idx in range(0, len(hidden_channels) - 1)
            ],
            torch.nn.Conv2d(
                in_channels=hidden_channels[-1],
                out_channels=head_channels,
                kernel_size=(1, 1),
            )
            if head_channels is not None
            else torch.nn.Identity(),
            torch.nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            if with_output_upsample
            else torch.nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ingest a 2D input tensor (B x C x H x W) and returns the result."""
        assert x.ndim == 4
        y = self.model(x)
        return y


def build_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: typing.Tuple[int, int] = (3, 3),
    stride: typing.Tuple[int, int] = (1, 1),
    padding: typing.Tuple[int, int] = (1, 1),
    dropout: float = 0.0,
    with_batch_norm: bool = True,
    with_max_pool: bool = True,
) -> torch.nn.Module:
    """Returns a Conv2d-BN-ReLU layer with optional pooling based on PyTorch modules."""
    modules = [
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        torch.nn.BatchNorm2d(out_channels) if with_batch_norm else torch.nn.Identity(),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity(),
    ]
    if with_max_pool:
        modules.append(torch.nn.MaxPool2d(kernel_size=(2, 2)))
    return torch.nn.Sequential(*modules)
