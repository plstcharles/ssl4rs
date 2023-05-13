import typing

import hydra.utils
import omegaconf
import torchvision.transforms

import ssl4rs.data.transforms.batch_sizer
import ssl4rs.data.transforms.geo
import ssl4rs.data.transforms.identity
import ssl4rs.data.transforms.patchify
import ssl4rs.data.transforms.tuple_mapper
import ssl4rs.data.transforms.wrappers
from ssl4rs.data.transforms.batch_sizer import (
    BatchSizer,
    batch_size_key,
    get_batch_size,
)
from ssl4rs.data.transforms.identity import Identity
from ssl4rs.data.transforms.patchify import Patchify
from ssl4rs.data.transforms.tuple_mapper import TupleMapper
from ssl4rs.data.transforms.wrappers import BatchDictToArgsWrapper

BatchDictType = typing.Dict[typing.AnyStr, typing.Any]
_BatchTransformType = typing.Callable[[BatchDictType], BatchDictType]
BatchTransformType = typing.Union[_BatchTransformType, typing.Sequence[_BatchTransformType], None]


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
