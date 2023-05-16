"""Implements a basic data parser interface for all index-based dataset formats."""
import abc
import typing

import lightning.pytorch.core.mixins as pl_mixins
import numpy as np
import torch.utils.data.dataset

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType, BatchTransformType


class DataParser(torch.utils.data.dataset.Dataset, pl_mixins.HyperparametersMixin):
    """Base interface used to provide common definitions for all index-based dataset formats.

    Since this interface is based on PyTorch's Dataset interface, it requires that `__len__` and
    `__getitem__` are implemented. On top of that, we add a few generic functions to be defined in
    other classes below (see the functions with `raise NotImplementedError`).

    Note that we make this class inherit from Lightning's `HyperparametersMixin` interface in order
    to save/restore constructor parameters. This allows us to build and return new data parser
    instances constructed with the original hyperparameters whenever we e.g. create a "filtered"
    version of this object.

    Args:
        batch_transforms: configuration dictionary or list of transformation operations that
            will be applied to the "raw" batch data read by this class. These should be
            callable objects that expect to receive a batch dictionary, and that also return
            a batch dictionary.
        add_default_transforms: specifies whether the 'default transforms' (batch sizer, batch
            identifier) should be added to the provided list of transforms. The following
            settings are used by these default transforms.
        batch_id_prefix: string used as a prefix in the batch identifiers generated for the
            data samples read by this parser.
        batch_index_key: an attribute name (key) under which we should be able to find the "index"
            of the batch dictionaries. Will be ignored if a batch identifier is already present in
            the loaded batches.
    """

    # TODO @@@@@@: add metadata getter, transform map, filter, select, cache, tensor names, ...

    def __init__(
        self,
        batch_transforms: "BatchTransformType" = None,
        add_default_transforms: bool = True,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
        batch_index_key: typing.Optional[str] = None,
    ):
        """Base class constructor that validates batch transforms and batch id settings."""
        import ssl4rs.data.transforms

        super().__init__()
        self.batch_id_key = ssl4rs.data.transforms.batch.batch_id_key
        self.batch_size_key = ssl4rs.data.transforms.batch.batch_size_key
        self.batch_id_prefix = batch_id_prefix
        self.batch_index_key = batch_index_key
        self.batch_transforms = ssl4rs.data.transforms.validate_or_convert_transform(
            batch_transforms,
            add_default_transforms=add_default_transforms,
            batch_id_prefix=self.batch_id_prefix,
            batch_index_key_=self.batch_index_key,
            dataset_name=self.dataset_name,
        )

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the total size (in terms of data batch count) of the dataset.

        This needs to be implemented in each derived class. It might be called fairly often to
        validate indexing ranges, so try to not have to re-iterate over your entire dataset in
        order to figure out its size each time this function is called.
        """
        raise NotImplementedError

    def __getitem__(self, index: typing.Hashable) -> "BatchDictType":
        """Returns a single data batch loaded from the dataset at the given index."""
        index = self._validate_or_convert_index(index)
        batch = self._get_raw_batch(index)
        # here, the 'batch' can be any type, but following the transforms, we'll expect a dict
        if self.batch_transforms:
            batch = self.batch_transforms(batch)
        assert isinstance(batch, typing.Dict), f"unexpected post-transform batch type: {type(batch)}"
        if self.batch_index_key is not None and self.batch_index_key not in batch:
            batch[self.batch_index_key] = index
        if self.batch_size_key not in batch:
            batch[self.batch_size_key] = self._get_batch_size_from_index(batch, index)
        if self.batch_id_key not in batch:
            batch[self.batch_id_key] = self._get_batch_id_for_index(batch, index)
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
        batch: "BatchDictType",
        index: typing.Hashable,
    ) -> typing.Hashable:
        """Returns the unique batch identifier for the given batch + (validated, converted) index.

        This will be called in case the "default transforms" were deactivated, and when the raw
        batch dictionaries that are loaded do not contain a batch id attribute. Derived classes
        should implement this if a special format of identifiers is required.
        """
        import ssl4rs.data.transforms

        return ssl4rs.data.transforms.get_batch_id(
            batch=batch,
            batch_id_prefix=self.batch_id_prefix,
            batch_index_key_=self.batch_index_key,
            dataset_name=self.dataset_name,
            index=index,
        )

    def _get_batch_size_from_index(self, batch: "BatchDictType", index: typing.Hashable) -> int:
        """Returns the expected batch size for the given batch + (validated, converted) index.

        This will be called in case the "default transforms" were deactivated, and when the raw
        batch dictionaries that are loaded do not contain a batch size attribute. Derived classes
        should implement this if the index may be a slice or any non-integer value.
        """
        return 1

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
