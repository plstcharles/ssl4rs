"""Implements a basic data parser interface for all index-based dataset formats."""
import abc
import typing

import numpy as np
import torch.utils.data.dataset

import ssl4rs.data.transforms


class DataParser(torch.utils.data.dataset.Dataset):
    """Base interface used to provide common definitions for all index-based dataset formats.

    Since this interface is based on PyTorch's Dataset interface, it requires that `__len__` and
    `__getitem__` are implemented. On top of that, we add a few generic functions to be defined in
    other classes below (see the functions with `raise NotImplementedError`).
    """

    # TODO @@@@@@: add metadata getter, transform map, filter, select, cache, tensor names, ...

    batch_id_key: str = "batch_id"
    """Attribute name used to insert batch identifiers inside the loaded batch dictionaries."""

    def __init__(
        self,
        batch_transforms: "ssl4rs.data.BatchTransformType" = None,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
    ):
        """Base class constructor that validates batch transforms and batch id settings."""
        self.batch_transforms = ssl4rs.data.transforms.validate_or_convert_transform(batch_transforms)
        if batch_id_prefix is None:
            batch_id_prefix = ""
        self.batch_id_prefix = str(batch_id_prefix)

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the total size (in terms of data batch count) of the dataset.

        This needs to be implemented in each derived class. It might be called fairly often to
        validate indexing ranges, so try to not have to re-iterate over your entire dataset in
        order to figure out its size each time this function is called.
        """
        raise NotImplementedError

    def __getitem__(self, index: typing.Hashable) -> typing.Dict[str, typing.Any]:
        """Returns a single data batch loaded from the dataset at the given index."""
        index = self._validate_or_convert_index(index)
        batch = self._get_raw_batch(index)
        # here, the 'batch' can be any type, but following the transforms, we'll expect a dict
        if self.batch_transforms:
            batch = self.batch_transforms(batch)
        assert isinstance(batch, dict), f"unexpected post-transform batch type: {type(batch)}"
        # finally, we'll add the batch id in the dict (if it's not already there)
        if self.batch_id_key not in batch:
            batch[self.batch_id_key] = self._get_batch_id_for_index(index)
        return batch

    def _validate_or_convert_index(self, index: typing.Hashable) -> typing.Hashable:
        """Validates or converts (if needed) the data batch index used to fetch a data batch."""
        # by default, this impl does NOT support slicing, and we assume all indices are ints
        if np.issubdtype(type(index), np.integer):
            # convert numpy ints here if needed, as some datasets might not be familiar w/ those
            index = int(index)  # noqa
        assert isinstance(index, int), f"unsupported index type for base parser: {type(index)}"
        assert 0 <= index < len(self), f"invalid data batch index being queried: {index}"
        return index

    @abc.abstractmethod
    def _get_raw_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch loaded from the dataset at a specified index.

        In its raw version, the loaded data can be in any format (tensors, arrays, tuples, dicts,
        references/views inside the parent dataset object, ...). We assume that if it is not a
        string-keyed dictionary object (as is expected as the output of the `__getitem__`
        function), it will be converted into one by the batch transforms.
        """
        raise NotImplementedError

    def _get_batch_id_for_index(
        self,
        index: typing.Hashable,
    ) -> str:
        """Returns the unique batch identifier (a string) for a given batch index.

        The default format is the combination of the batch identifier prefix (provided in the class
        constructor), the dataset name (class-defined), an underscore, and the batch index itself.
        If the index is an integer, its value will be zero-padded to 8 digits.
        """
        prefix = f"{self.batch_id_prefix}_" if self.batch_id_prefix else ""
        if np.issubdtype(type(index), np.integer) or isinstance(index, int):
            index = f"batch{index:08d}"
        return f"{prefix}{self.dataset_name}_{index}"

    @property
    @abc.abstractmethod
    def tensor_names(self) -> typing.List[str]:
        """Names of the data tensors that will be provided in the loaded batches.

        Note that additional tensors and other attributes may be loaded as well, but these are the
        'primary' fields that should be expected by downstream processing stages.

        Typical 'tensor' names are include: 'image', 'label', 'mask', etc.
        """
        raise NotImplementedError

    @property
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information parsed from the dataset (if any)."""
        return dict()  # there's nothing by default in the base class impl

    @property
    @abc.abstractmethod
    def dataset_name(self) -> typing.AnyStr:
        """Returns the dataset name used to identify this particular dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def summary(self, *args, **kwargs) -> None:
        """Prints a summary of the dataset using the default logger.

        This function should be easy-to-call (parameter-free, if possible) and fast-to-return
        (takes seconds or tens-of-seconds at most) in order to remain friendly to high-level users.
        What it does specifically is totally up to the derived class.

        All outputs should be sent to the default logger.
        """
        raise NotImplementedError
