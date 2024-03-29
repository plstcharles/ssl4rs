import copy
import math
import typing

import numpy as np
import torch.utils.data

from ssl4rs.data.parsers.utils.base import DataParser

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchTransformType


class ParserWrapper(DataParser):
    """Base interface used to wrap generic data parsers (e.g. from other DL frameworks).

    This is meant to provide a compatibility layer between data parsers that might be imported from
    other framework and that do not have a simple way to interface with the rest of this framework,
    e.g. to apply batch-level transformation operations and generate batch identifiers.

    Args:
        dataset: the dataset-compatible object to be wrapped by this class.
        dataset_name: the name of the dataset to use in batch identifiers and logging. If "AUTO",
            will use the name of the wrapped "dataset" object's class.
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

    def __init__(
        self,
        dataset: typing.Any,
        dataset_name: typing.AnyStr = "AUTO",  # if 'AUTO', will use wrapped object class name
        batch_transforms: "BatchTransformType" = None,
        add_default_transforms: bool = True,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
        batch_index_key: typing.Optional[str] = None,
    ):
        """Validates that the provided PyTorch-Dataset-compatible object can be wrapped."""
        self.save_hyperparameters(ignore="dataset", logger=False)
        self._dataset_size: typing.Optional[int] = None
        if dataset_name == "AUTO":
            import ssl4rs.utils.filesystem

            dataset_name = ssl4rs.utils.filesystem.slugify(type(dataset).__name__)
        self._dataset_name = str(dataset_name)
        self._tensor_names: typing.List[str] = []  # will be filled when needed/available
        super().__init__(
            batch_transforms=batch_transforms,
            add_default_transforms=add_default_transforms,
            batch_id_prefix=batch_id_prefix,
            batch_index_key=batch_index_key,
        )
        assert hasattr(dataset, "__len__"), "missing mandatory dataset length attribute!"
        assert hasattr(dataset, "__getitem__"), "missing mandatory dataset item getter!"
        self.dataset = dataset  # should be read-only, as we'll cache the dataset size

    def __len__(self) -> int:
        """Returns the total size (in terms of data batch count) of the dataset.

        It might be called fairly often to validate indexing ranges, so in order to avoid issues
        with dataset implementations that re-iterate over their entire data in order to figure out
        the batch count each time this function is called, we cache the returned length.
        """
        if self._dataset_size is None:
            self._dataset_size = len(self.dataset)  # cached in case it takes a while to get it
        return self._dataset_size

    def __getitem__(self, index: typing.Hashable) -> typing.Dict[str, typing.Any]:
        """Returns a single data batch loaded from the dataset at the given index."""
        batch = super().__getitem__(index)  # load the batch data using the base class impl
        if not self._tensor_names:  # if we have not seen tensor names yet, fill them in
            # note: this assumes that ALL data batches have the same attribs across all indices
            self._tensor_names = [name for name in batch.keys() if name != self.batch_id_key]
        return batch

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
        # we fetch the corresponding batch data for the given index from the wrapped dataset object;
        # if this line fails, it means we do not have a PyTorch-Dataset-compatible object
        batch = self.dataset[index]
        # if the raw data is a dictionary, and if the default batch index key is not in there...
        if isinstance(batch, typing.Dict) and self.batch_index_key not in batch:
            # ... add the index used to fetch the batch with the default key
            batch[self.batch_index_key] = index
        return batch

    @property
    def tensor_names(self) -> typing.List[str]:
        """Names of the data tensors that will be provided in the loaded batches.

        In this particular class, the tensor names will be filled in when we first load a data
        batch, so if this function returns nothing, make sure you call `__getitem__` at least once
        before calling it.

        Note that additional tensors and other attributes may be loaded as well, but these are the
        'primary' fields that should be expected by downstream processing stages.

        Typical 'tensor' names are include: 'image', 'label', 'mask', etc.
        """
        return self._tensor_names

    @property
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information parsed from the dataset (if any)."""
        return dict(name=self.dataset_name)

    @property
    def dataset_name(self) -> typing.AnyStr:
        """Returns the dataset name used to identify this particular dataset."""
        return self._dataset_name

    def summary(self, *args, **kwargs) -> None:
        """Prints a summary of the dataset using the default logger.

        This function should be easy-to-call (parameter-free, if possible) and fast-to-return (takes
        seconds or tens-of-seconds at most) in order to remain friendly to high-level users. What it
        does specifically is totally up to the derived class.

        All outputs should be sent to the default logger.
        """
        import ssl4rs.utils.logging

        logger = ssl4rs.utils.logging.get_logger(__name__)
        logger.info(self)
        logger.info(f"dataset_name={self.dataset_name}, length={len(self)}")


class IterableDataset(torch.utils.data.IterableDataset):
    """Wraps a dataset so that items provided in varying-size arrays are returned one at a time.

    If shuffling is not activated, the items will be returned in the order defined by the arrays of
    the wrapped dataset as if those arrays were concatenated. Otherwise, an item buffer will be
    filled up to a specified size, shuffled, and slowly emptied repeatedly until all items have
    been returned.

    Note: only ONE data loader should use this object at a time because of its internal buffer and
    because of the state variables required to iterate over the wrapped dataset.

    Args:
        dataset_to_wrap: dataset to wrap for array item retrieval.
        target_array_keys: list of keys for target arrays to retrieve items from.
        constant_copy_keys: list of keys that contain other values/tensors to copy for each item.
        shuffle: specifies whether to shuffle the items inside the internal buffer.
        buffer_size: size of the item buffer to use. The buffer will be filled with up to that many
            items loaded from wrapped dataset arrays, shuffled (if needed), and them emptied one
            item at a time. Once totally empty, the process is repeated until all items have been
            returned.
        shuffle_seed: random seed used to initialize the RNG used for shuffling.
    """

    def __init__(
        self,
        dataset_to_wrap: torch.utils.data.Dataset,
        target_array_keys: typing.List[str],
        constant_copy_keys: typing.Optional[typing.List[str]] = None,
        shuffle: bool = False,
        buffer_size: int = 0,
        shuffle_seed: typing.Optional[int] = None,
    ) -> None:
        """Initializes the dataset wrapper while validating settings."""
        assert hasattr(dataset_to_wrap, "__getitem__") and hasattr(
            dataset_to_wrap, "__len__"
        ), f"invalid dataset type: {type(dataset_to_wrap)}"
        self.dataset = dataset_to_wrap
        assert len(target_array_keys) > 0, "need to specify at least one target array key!"
        self.target_array_keys = target_array_keys
        if constant_copy_keys is None:
            constant_copy_keys = []
        self.constant_copy_keys = constant_copy_keys
        assert len(set(self.target_array_keys) & set(self.constant_copy_keys)) == 0
        self.shuffle = shuffle
        assert buffer_size >= 0, f"invalid buffer size: {buffer_size}"
        self.buffer_size = buffer_size
        self.shuffle_seed = shuffle_seed
        self._reset_iterator()

    def _reset_iterator(self) -> None:
        """Resets the iterator internal state + buffer (if one is used)."""
        dataset_size = len(self.dataset)  # noqa
        self._buffer = []
        # TODO @@@@@@ also update this to work with distributed setup (i.e. with world size > 1)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self._remaining_idxs = list(range(dataset_size))
        else:  # in a worker process, share the workload by splitting indices across all workers
            idx_count_per_worker = int(math.ceil(dataset_size / float(worker_info.num_workers)))
            idx_start = worker_info.id * idx_count_per_worker
            idx_end = min(idx_start + idx_count_per_worker, dataset_size)
            self._remaining_idxs = list(range(idx_start, idx_end))
        if self.shuffle:
            self._shuffle_rng = np.random.default_rng(self.shuffle_seed)
            self._shuffle_rng.shuffle(self._remaining_idxs)

    def __iter__(self) -> "IterableDataset":
        """Get an iterator for the items in the target arrays of the wrapper dataset."""
        self._reset_iterator()
        return self

    def __next__(self) -> typing.Dict[str, typing.Any]:
        """Get the next item from the next array of the wrapped dataset."""
        if not self._buffer:  # time for a refill
            while self._remaining_idxs:  # there are still items to fetch from the dataset
                curr_idx = self._remaining_idxs.pop(0)
                curr_items = list(self._yield_items_for_idx(curr_idx))
                if not curr_items:
                    continue
                self._buffer.extend(curr_items)
                if len(self._buffer) >= self.buffer_size:
                    break
            if self.shuffle:
                self._shuffle_rng.shuffle(self._buffer)  # refill done, shuffle the buffer
        if not self._buffer and not self._remaining_idxs:
            raise StopIteration  # we're at the total end, no more refills
        # buffer is ready, return the next item
        return self._buffer.pop(0)

    def _yield_items_for_idx(
        self,
        index: int,
    ) -> typing.Generator[typing.Dict[str, typing.Any], None, None]:
        """Yields all items for the specified dataset index while deep copying constants."""
        assert index <= len(self.dataset), f"invalid index: {index}"  # noqa
        batch = self.dataset[index]
        assert isinstance(batch, dict), f"unexpected batch type: {type(batch)}"
        array_data, constant_data = {}, {}
        for key in self.constant_copy_keys:
            assert key in batch, f"missing constant value '{key}' from batch"
            constant_data[key] = batch[key]
        array_length = None
        for key in self.target_array_keys:
            assert key in batch, f"missing target array '{key}' from batch"
            if array_length is None:
                array_length = len(batch[key])
            else:
                assert len(batch[key]) == array_length, "mismatched array length(s)"
            array_data[key] = batch[key]
        for item_idx in range(array_length):
            output = {
                **{key: copy.deepcopy(val) for key, val in constant_data.items()},
                **{key: array[item_idx] for key, array in array_data.items()},
            }
            yield output
