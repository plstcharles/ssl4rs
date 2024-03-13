import typing

import numpy as np
import torch
import torch.utils.data

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType

batch_size_key: str = "batch_size"
"""Default batch dictionary key (string) used to store/fetch the batch size."""

batch_id_key: str = "batch_id"
"""Default batch dictionary key (string) used to store/fetch the batch identifier."""

batch_index_key: str = "index"
"""Default batch dictionary key (string) used to store/fetch the batch index."""


class BatchSizer:
    """Adds a `batch_size` attribute to a loaded batch dictionary using a specified arg length.

    This operation is used to 'wrap' dataset parsers that already return batch dictionaries but that
    do not have a `batch_size` field yet. If one is present, we'll check that its value is correct,
    but do nothing more. If the `batch_size` field is truly missing, we'll use the specified hints
    to discover it and insert its value in the batch dict.

    With this wrapper on top a dataset parser or dataloader, you will be able to use some cool
    utility functions such as ``ssl4rs.data.get_batch_size``.

    Args:
        batch_size_hint: hint to be used in order to get the batch size. If it's an integer, we'll
            assign it directly. If it's a string, we'll assume it's the key to a tensor/array
            that's already in the batch, and we'll return its length as the batch size.
        throw_if_smaller: toggles whether to throw an exception if the batch size is found in the
            batch dictionary already and its size is actually smaller than expected (will always
            throw is its size is larger!).
    """

    def __init__(
        self,
        batch_size_hint: typing.Union[int, typing.AnyStr],
        throw_if_smaller: bool = True,
    ):
        """Validates and initializes the batch size settings."""
        assert isinstance(batch_size_hint, (int, str)), f"invalid batch size hint type: {type(batch_size_hint)}"
        if isinstance(batch_size_hint, int):
            assert batch_size_hint > 0, "batch size should be a strictly-positive value!"
        self.batch_size_hint = batch_size_hint
        self.throw_if_smaller = throw_if_smaller

    def __call__(
        self,
        batch: "BatchDictType",
    ) -> "BatchDictType":
        """Converts the given batch data tuple into a batch data dictionary using the key map.

        Args:
            batch: the loaded batch dictionary whose batch size might be missing. We will either
                insert it directly or use the hint to figure it out from the current content. This
                dictionary will be UPDATED IN PLACE since we do not deep copy this reference.

        Returns:
            The same dictionary as the one passed it, with the potentially updated batch size.
        """
        assert isinstance(batch, typing.Dict), f"unexpected input batch type: {type(batch)}"
        if not batch:  # easy case: the batch dict is empty
            batch[batch_size_key] = 0
            return batch
        if isinstance(self.batch_size_hint, int):
            batch_size = self.batch_size_hint
        else:
            assert (
                self.batch_size_hint in batch
            ), f"could not locate batch size hint target in given dict: {self.batch_size_hint}"
            batch_size = len(batch[self.batch_size_hint])
        if batch_size_key in batch:
            found_batch_size = get_batch_size(batch)
            if self.throw_if_smaller:
                assert found_batch_size == batch_size, f"batch sizes mismatch! ({batch['batch_size']} vs {batch_size})"
            else:
                assert (
                    found_batch_size <= batch_size
                ), f"bad batch size upper bound! ({found_batch_size} > {batch_size})"
        else:
            batch[batch_size_key] = batch_size
        return batch

    def __repr__(self) -> str:
        out_str = self.__class__.__name__ + "("
        out_str += f"batch_size_hint={self.batch_size_hint}"
        out_str += f", throw_if_smaller={self.throw_if_smaller}"
        out_str += ")"
        return out_str


def get_batch_size(batch: "BatchDictType") -> int:
    """Checks the provided batch dictionary and attempts to return the batch size.

    If the given dictionary does not contain a `batch_size` attribute that we can interpret, we will
    throw an exception. Otherwise, if that attribute is an integer or a tensor/array, the resulting
    batch size will be returned.

    If the batch size is stored as an array, we will assume that it is as the result of collating
    the loaded batches of multiple parsers/dataloaders; we will therefore sum all values in that
    array (where each value should be the batch size of a single collated chunk) in order to return
    the total batch size.
    """
    if batch is None or not batch:
        return 0
    assert batch_size_key in batch, "could not find the mandatory 'batch_size' key in the given batch dictionary!"
    batch_size = batch[batch_size_key]
    # we'll try to interpret this potential object in any way we can...
    if isinstance(batch_size, int):
        pass  # nothing to do, it's good as-is!
    elif np.issubdtype(type(batch_size), np.integer):
        batch_size = int(batch_size)  # in case we're using numpy ints (might break slicing)
    elif isinstance(batch_size, np.ndarray):
        assert np.issubdtype(batch_size.dtype, np.integer), f"invalid batch size array type: {batch_size.dtype}"
        batch_size = int(batch_size.astype(np.int64).sum())
    elif isinstance(batch_size, torch.Tensor):
        batch_size = batch_size.long().sum().item()
    else:
        raise NotImplementedError(f"cannot handle batch size type: {type(batch_size)}")
    assert batch_size >= 0, f"found an invalid batch size! ({batch_size})"
    return batch_size


class BatchIdentifier:
    """Adds a `batch_id` attribute to a loaded batch dictionary.

    This operation is used to 'wrap' dataset parsers that already return batch dictionaries but that
    do not have a `batch_id` field yet. If one is present, we'll check that its value is correct,
    but do nothing more. If the `batch_id` field is truly missing, we'll use the specified hints
    to create it and insert its value in the batch dict.

    With this wrapper on top a dataset parser or dataloader, you will be able to use some cool
    utility functions such as ``ssl4rs.data.get_batch_id``.

    Args:
        batch_id_prefix: a prefix used when building batch identifiers. Will be ignored if a batch
            identifier is already present in the `batch`.
        batch_index_key_: an attribute name (key) under which we should be able to find the "index"
            of the provided batch dictionary. Will be ignored if a batch identifier is already
            present in the `batch`.
        dataset_name: an extra name to add when building batch identifiers. Will be ignored if a
            batch identifier is already present in the `batch`.
    """

    def __init__(
        self,
        batch_id_prefix: typing.Optional[str] = None,
        batch_index_key_: typing.Optional[str] = None,
        dataset_name: typing.Optional[str] = None,
    ):
        """Validates and initializes the batch id settings."""
        self.batch_id_prefix = batch_id_prefix
        self.batch_index_key = batch_index_key_
        self.dataset_name = dataset_name

    def __call__(
        self,
        batch: "BatchDictType",
        index: typing.Optional[typing.Hashable] = None,
    ) -> "BatchDictType":
        """Converts the given batch data tuple into a batch data dictionary using the key map.

        Args:
            batch: the loaded batch dictionary whose batch size might be missing. We will either
                insert it directly or use the hint to figure it out from the current content. This
                dictionary will be UPDATED IN PLACE since we do not deep copy this reference.
            index: the hashable index that corresponds to the integer or unique ID used to fetch
                the given `batch` from a dataset parser. Constitutes the basis for the creation of
                a batch identifier. If not specified, the `batch_index_key_` must be provided so
                that we can find the actual index from a field within the batch dictionary.

        Returns:
            The same dictionary as the one passed it, with the potentially updated batch size.
        """
        assert isinstance(batch, typing.Dict), f"unexpected input batch type: {type(batch)}"
        if batch_id_key in batch:
            assert isinstance(
                batch[batch_id_key], typing.Hashable
            ), f"invalid batch id found in batch dict: {type(batch[batch_id_key])}"
            return batch
        batch_id = get_batch_id(
            batch=batch,
            batch_id_prefix=self.batch_id_prefix,
            batch_index_key_=self.batch_index_key,
            dataset_name=self.dataset_name,
            index=index,
        )
        batch[batch_id_key] = batch_id
        return batch

    def __repr__(self) -> str:
        out_str = self.__class__.__name__ + "("
        out_str += f"batch_id_prefix={self.batch_id_prefix}"
        out_str += f", batch_index_key_={self.batch_index_key}"
        out_str += f", dataset_name={self.dataset_name}"
        out_str += ")"
        return out_str


def get_batch_id(
    batch: typing.Optional["BatchDictType"] = None,
    batch_id_prefix: typing.Optional[str] = None,
    batch_index_key_: typing.Optional[str] = None,
    dataset_name: typing.Optional[str] = None,
    index: typing.Optional[typing.Hashable] = None,
) -> typing.Hashable:
    """Checks the provided batch dictionary and attempts to return its batch identifier.

    If the given dictionary does not contain a 'batch_id' attribute that we can return, we will
    create such an id with the provided prefix/dataset/index info. If no info is available, we will
    throw an exception.

    Args:
        batch: the batch dictionary from which we'll return the batch identifier (if it's already
            there), or from which we'll gather data in order to build a batch identifier.
        batch_id_prefix: a prefix used when building batch identifiers. Will be ignored if a batch
            identifier is already present in the `batch`.
        batch_index_key_: an attribute name (key) under which we should be able to find the "index"
            of the provided batch dictionary. Will be ignored if a batch identifier is already
            present in the `batch`. If necessary yet `None`, we will at least try the default
            `batch_index_key` value before throwing an exception.
        dataset_name: an extra name to add when building batch identifiers. Will be ignored if a
            batch identifier is already present in the `batch`.
        index: the hashable index that corresponds to the integer or unique ID used to fetch the
            targeted batch dictionary from a dataset parser. Constitutes the basis for the creation
            of a batch identifier. If not specified, the `batch_index_key` must be provided so that
            we can find the actual index from a field within the batch dictionary.  Will be ignored
            if a batch identifier is already present in the `batch`.

    Returns:
        The (hopefully) unique batch identifier used to reference this batch elsewhere.
    """
    if batch is None or not batch or batch_id_key not in batch:
        if index is None:
            if batch_index_key_ is None:
                batch_index_key_ = batch_index_key  # fallback to module-wide default
            assert isinstance(batch, typing.Dict) and batch_index_key_ in batch, (
                "batch dict did not contain a batch identifier, and we need at least an index to "
                f"build such an identifier!\n (provide it via the `{batch_index_key_}` dict key in"
                f"the data parser, or implement your own transform to add it)"
            )
            index = batch[batch_index_key_]
        if isinstance(index, np.ndarray):
            assert index.ndim == 1 and index.size > 0, "index should be non-empty 1d vector"
            index = tuple(index)
            if len(index) == 1:
                index = index[0]
        assert isinstance(index, typing.Hashable), f"bad index for batch identifier: {type(index)}"
        prefix = f"{batch_id_prefix}_" if batch_id_prefix else ""
        dataset = f"{dataset_name}_" if dataset_name else ""
        if isinstance(index, int) or np.issubdtype(type(index), np.integer):
            index = f"batch{index:08d}"
        return f"{prefix}{dataset}{index}"
    else:
        batch_id = batch[batch_id_key]
        assert isinstance(batch_id, typing.Hashable), f"found batch id has bad type: {type(batch_id)}"
    return batch_id


def get_batch_index(
    batch: "BatchDictType",
    batch_index_key_: typing.Optional[str] = None,
) -> typing.Hashable:
    """Checks the provided batch dictionary and attempts to return the batch index.

    If the given dictionary does not contain a `batch_index` attribute that we can interpret, we
    will throw an exception.


    Args:
        batch: the batch dictionary from which we'll return the batch index.
        batch_index_key_: an attribute name (key) under which we should be able to find the "index"
            of the provided batch dictionary. If `None`, will default to the module-defined value.
    """
    assert isinstance(batch, typing.Dict), f"invalid batch type: {type(batch)}"
    if batch_index_key_ is None:
        batch_index_key_ = batch_index_key
    assert batch_index_key_ in batch, f"batch dict does not contain key: {batch_index_key_}"
    batch_index = batch[batch_index_key_]
    if isinstance(batch_index, np.ndarray):
        assert batch_index.ndim == 1 and batch_index.size > 0, "index should be non-empty 1d vector"
        batch_index = tuple(batch_index)
        if len(batch_index) == 1:
            batch_index = batch_index[0]
    assert isinstance(batch_index, typing.Hashable), f"invalid batch index type: {type(batch_index)}"
    return batch_index

import pdb
def default_collate(
    batches: typing.List["BatchDictType"],
    keys_to_batch_manually: typing.Sequence[typing.AnyStr] = (),
    keys_to_ignore: typing.Sequence[typing.AnyStr] = (),
) -> "BatchDictType":
    """Performs the default collate function while manually handling some given special cases."""
    assert isinstance(batches, (list, tuple)) and all(
        [isinstance(b, dict) for b in batches]
    ), f"unexpected type for batch array provided to collate: {type(batches)}"
    assert all(
        [len(np.setxor1d(list(batches[idx].keys()), list(batches[0].keys()))) == 0 for idx in range(1, len(batches))]
    ), "not all batches have the same sets of keys! (implement your own custom collate fn!)"
    avail_batch_keys = list(batches[0].keys())
    output = dict()
    # first step: look for the keys that we need to batch manually, and handle those
    default_keys_to_batch_manually = [
        "batch_id",  # should correspond to hashable objects that might hurt torch's default_collate
    ]
    keys_to_batch_manually = {*keys_to_batch_manually, *default_keys_to_batch_manually}
    for key in keys_to_batch_manually:
        if key in avail_batch_keys:
            output[key] = [b[key] for b in batches]
    keys_to_skip_or_already_done = {*keys_to_ignore, *keys_to_batch_manually}
    pdb.set_trace()
    output.update(
        torch.utils.data.default_collate(
            [{k: v for k, v in b.items() if k not in keys_to_skip_or_already_done} for b in batches]
        )
    )
    pdb.set_trace()
    output['image_data'] = torch.stack(output['image_data'], axis=0)

    if isinstance(output['field_mask'], list):
        output['field_mask'] = torch.stack(output['field_mask'], axis=0)
    if output['field_mask'].dim() == 3:
        output['field_mask'] = output['field_mask'].unsqueeze(0)

    pdb.set_trace()
    if batch_size_key not in output:
        output[batch_size_key] = len(batches)
    pdb.set_trace()
    return output
