"""Implements a simple MLP PyTorch module for example/demo/debug models."""

import typing

import torch.nn


class SimpleMLP(torch.nn.Module):
    """Simple Multi Layer Perceptron (MLP) neural network implementation.

    This class is meant to be used inside other PyTorch modules, or in a LightningModule.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: typing.Sequence[int],
        out_channels: int,
        with_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        """Initializes the MLP using the provided settings."""
        assert isinstance(in_channels, int) and in_channels > 0
        assert isinstance(out_channels, int) and out_channels > 0
        if isinstance(hidden_channels, str):
            hidden_channels = hidden_channels.strip(" []()")
            hidden_channels = [int(ch) for ch in hidden_channels.split(",")]
        assert len(hidden_channels) > 0 and all([c > 0 for c in hidden_channels])
        super().__init__()
        self.model = torch.nn.Sequential(
            build_layer(
                in_channels=in_channels,
                out_channels=hidden_channels[0],
                with_batch_norm=with_batch_norm,
                dropout=dropout,
            ),
            *[
                build_layer(
                    in_channels=hidden_channels[idx],
                    out_channels=hidden_channels[idx + 1],
                    with_batch_norm=with_batch_norm,
                    dropout=dropout,
                )
                for idx in range(0, len(hidden_channels) - 1)
            ],
            torch.nn.Linear(
                in_features=hidden_channels[-1],
                out_features=out_channels,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ingest a 1D input tensor (B x C) and returns the result."""
        assert x.ndim >= 2
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        y = self.model(x)
        return y


def build_layer(
    in_channels: int,
    out_channels: int,
    with_batch_norm: bool = False,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """Returns a Conv2d-BN-ReLU layer with optional pooling based on PyTorch modules."""
    assert 0 <= dropout <= 1.0
    modules = [
        torch.nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
        ),
        torch.nn.BatchNorm1d(out_channels) if with_batch_norm else torch.nn.Identity(),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity(),
    ]
    return torch.nn.Sequential(*modules)
