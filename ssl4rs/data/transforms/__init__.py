import typing

import hydra.utils
import omegaconf
import torchvision.transforms

import ssl4rs.data.transforms.batch_sizer
import ssl4rs.data.transforms.patchify
import ssl4rs.data.transforms.tuple_mapper
from ssl4rs.data.transforms.batch_sizer import BatchSizer, get_batch_size
from ssl4rs.data.transforms.patchify import Patchify
from ssl4rs.data.transforms.tuple_mapper import TupleMapper

BatchDictType = typing.Dict[typing.AnyStr, typing.Any]
_BatchTransformType = typing.Callable[[BatchDictType], BatchDictType]
BatchTransformType = typing.Union[_BatchTransformType, typing.Sequence[_BatchTransformType], None]


class Identity:
    """Simple/clean implementation of an identity transformation. Yep, it does nothing.

    This may be useful for unit testing, in conditional transforms, or in composition operations.
    """

    def __call__(self, batch: typing.Any) -> typing.Any:
        """Does nothing, and returns the provided batch object as-is.

        Args:
            batch: the batch object to be 'transformed'.

        Returns:
            The same batch object.
        """
        return batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def validate_or_convert_transform(transform: BatchTransformType) -> BatchTransformType:
    """Validates or converts the given transform object to a proper (torchvision-style) object."""
    if transform is None:
        return Identity()  # if nothing is provided, assume that's a shortcut for the identity func
    if isinstance(transform, (dict, omegaconf.DictConfig)) and "_target_" in transform:
        transform = hydra.utils.instantiate(transform)
    if isinstance(transform, typing.Sequence):
        out_t = []
        for t in transform:
            if isinstance(t, (dict, omegaconf.DictConfig)) and "_target_" in t:
                t = hydra.utils.instantiate(t)
            assert callable(t), "if a sequence of transforms is given, each transform must be a callable function"
            out_t.append(t)
        if len(out_t) == 0:
            return Identity()  # there are no transforms to apply, return an identity function
        return torchvision.transforms.Compose(out_t)
    assert callable(transform), "the batch transform function must be a callable object"
    return transform
