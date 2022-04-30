import typing

import numpy as np
import torch

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType


class BatchSizer:
    """Adds a `batch_size` attribute to a loaded batch dictionary using a specified arg length.

    This operation is used to 'wrap' dataset parsers that already return batch dictionaries but that
    do not have a `batch_size` field yet. If one is present, we'll check that its value is correct,
    but do nothing more. If the `batch_size` field is truly missing, we'll use the specified value
    (or the specified key to look in up) and insert it.

    With this wrapper on top a dataset parser or dataloader, you will be able to use some cool
    utility functions such as ``ssl4rs.data.get_batch_size``.

    Attributes:
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
        """Validates and initializes the batch size hint parameter.

        Args:
            batch_size_hint: hint to be used in order to get the batch size. If it's an integer,
                we'll assign it directly. If it's a string, we'll assume it's the key to a
                tensor/array that's already in the batch, and we'll return its length as the batch
                size.
        """
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
            batch["batch_size"] = 0
            return batch
        if isinstance(self.batch_size_hint, int):
            batch_size = self.batch_size_hint
        else:
            assert (
                self.batch_size_hint in batch
            ), f"could not locate batch size hint target in given dict: {self.batch_size_hint}"
            batch_size = len(batch[self.batch_size_hint])
        if "batch_size" in batch:
            found_batch_size = get_batch_size(batch)
            if self.throw_if_smaller:
                assert found_batch_size == batch_size, f"batch sizes mismatch! ({batch['batch_size']} vs {batch_size})"
            else:
                assert (
                    found_batch_size <= batch_size
                ), f"bad batch size upper bound! ({found_batch_size} > {batch_size})"
        else:
            batch["batch_size"] = batch_size
        return batch


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
    assert "batch_size" in batch, "could not find the mandatory 'batch_size' key in the given batch dictionary!"
    batch_size = batch["batch_size"]
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
