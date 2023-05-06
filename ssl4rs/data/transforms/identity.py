import typing

import torch


class Identity(torch.nn.Module):
    """Simple/clean implementation of an identity transformation. Yep, it does nothing.

    This may be useful for unit testing, in conditional transforms, or in composition operations.
    Note that like most torchvision transforms, this class inherits from `torch.nn.Module` in order
    to be compatible with torchscript (and to be compatible with accelerated transform pipelines).
    """

    def forward(self, batch: typing.Any) -> typing.Any:
        """Does nothing, and returns the provided batch object as-is.

        Args:
            batch: the batch object to be 'transformed'.

        Returns:
            The same batch object.
        """
        return batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
