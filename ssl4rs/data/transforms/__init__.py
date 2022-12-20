import typing

import ssl4rs.data.transforms.batch_sizer
import ssl4rs.data.transforms.patchify
import ssl4rs.data.transforms.tuple_mapper
from ssl4rs.data.transforms.batch_sizer import BatchSizer, get_batch_size
from ssl4rs.data.transforms.patchify import Patchify
from ssl4rs.data.transforms.tuple_mapper import TupleMapper

BatchDictType = typing.Dict[typing.AnyStr, typing.Any]
BatchTransformType = typing.Callable[[BatchDictType], BatchDictType]
