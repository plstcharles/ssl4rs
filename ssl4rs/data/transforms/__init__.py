import typing

import hydra.utils
import omegaconf
import torchvision.transforms

import ssl4rs.data.transforms.batch
import ssl4rs.data.transforms.geo
import ssl4rs.data.transforms.identity
import ssl4rs.data.transforms.pad
import ssl4rs.data.transforms.patchify
import ssl4rs.data.transforms.tuple_mapper
import ssl4rs.data.transforms.wrappers
from ssl4rs.data.transforms.batch import (
    BatchIdentifier,
    BatchSizer,
    batch_id_key,
    batch_index_key,
    batch_size_key,
    default_collate,
    get_batch_id,
    get_batch_index,
    get_batch_size,
)
from ssl4rs.data.transforms.identity import Identity
from ssl4rs.data.transforms.pad import PadIfNeeded, pad_if_needed
from ssl4rs.data.transforms.patchify import Patchify
from ssl4rs.data.transforms.tuple_mapper import TupleMapper
from ssl4rs.data.transforms.wrappers import BatchDictToArgsWrapper

BatchDictType = typing.Dict[typing.AnyStr, typing.Any]
"""Default type used to represent a data batch loaded by a data parser/loader and fed to a model."""

_BatchTransformType = typing.Callable[[BatchDictType], BatchDictType]

BatchTransformType = typing.Union[_BatchTransformType, typing.Sequence[_BatchTransformType], None]
"""Default type used to represent a callable object or function that transforms a data batch."""


def validate_or_convert_transform(
    transform: BatchTransformType,
    add_default_transforms: bool = True,
    batch_id_prefix: typing.Optional[str] = None,
    batch_index_key_: typing.Optional[str] = None,
    dataset_name: typing.Optional[str] = None,
) -> BatchTransformType:
    """Validates or converts the given transform object to a proper (torchvision-style) object.

    Args:
        transform: a callable object, DictConfig of a callable object (or a list of those), or a
            list of such objects that constitute the transformation pipeline to be validated.
        add_default_transforms: toggles whether to add "default" transformation ops to the returned
            transform pipeline. The other arguments are related to these default transforms only,
            and are ignored if `add_default_transforms` is False.
        batch_id_prefix: a prefix used when building batch identifiers. Will be ignored if a batch
            identifier is already present in the `batch`.
        batch_index_key_: an attribute name (key) under which we should be able to find the "index"
            of the provided batch dictionary. Will be ignored if a batch identifier is already
            present in the `batch`.
        dataset_name: an extra name to add when building batch identifiers. Will be ignored if a
            batch identifier is already present in the `batch`.

    Returns:
        The "composed" (assembled, and ready-to-be-used) batch transformation pipeline.
    """
    if transform is None:
        transform = [Identity()]  # if nothing is provided, assume that's a shortcut for the identity func
    if isinstance(transform, (dict, omegaconf.DictConfig)) and "_target_" in transform:
        t = hydra.utils.instantiate(transform)
        assert callable(t), f"instantiated transform object not callable: {type(t)}"
        transform = [t]
    if callable(transform):
        transform = [transform]
    assert isinstance(transform, typing.Sequence), (
        "transform must be provided as a callable object, as a DictConfig for a callable object, or as "
        f"a sequence of such DictConfig/callable objects; instead, we got: {type(transform)}"
    )
    out_t = []
    for t in transform:
        if isinstance(t, (dict, omegaconf.DictConfig)) and "_target_" in t:
            t = hydra.utils.instantiate(t)
        assert callable(t), f"transform object not callable: {type(t)}"
        out_t.append(t)
    if len(out_t) == 0:
        out_t = [Identity()]  # there are no transforms to apply, return an identity function
    if add_default_transforms:
        out_t = [
            BatchSizer(batch_size_hint=1),
            BatchIdentifier(
                batch_id_prefix=batch_id_prefix,
                batch_index_key_=batch_index_key_,
                dataset_name=dataset_name,
            ),
            *out_t,
        ]
    return torchvision.transforms.Compose(out_t)
