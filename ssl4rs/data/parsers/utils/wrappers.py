import typing

import numpy as np
import torch.utils.data
import torchvision

import ssl4rs


class ParserWrapper(torch.utils.data.dataset.Dataset):
    """Base interface used to wrap generic data parsers.

    This is meant to provide a compatibility layer between data parsers that might be imported from
    other framework and that do not have a simple way to interface with the rest of the framework,
    e.g. to apply batch-level transformation operations and generate batch identifiers.
    """

    def __init__(
        self,
        dataset: typing.Any,
        batch_transforms: typing.Sequence["ssl4rs.data.BatchTransformType"] = (),
        batch_id_prefix: typing.AnyStr = "",
        dataset_id_prefix: typing.AnyStr = "AUTO",  # if 'AUTO', will use wrapped object class name
    ):
        """Parses data from a PyTorch-Dataset-interface compatible object.

        For the batch identifiers, the default format provided by the `__getitem__` function will
        be the dataset identifier prefix, followed by the batch identifier prefix, followed by the
        batch identifier itself (`batchXXXXXXXX`, with the index-specific number padded to 8
        digits).
        """
        assert hasattr(dataset, "__len__"), "missing mandatory dataset length attribute!"
        dataset_size = len(dataset)
        assert dataset_size >= 0, f"invalid dataset sample count: {dataset_size}"
        assert hasattr(dataset, "__getitem__"), "missing mandatory dataset item getter!"
        self.dataset = dataset
        if batch_transforms is not None and len(batch_transforms) > 0:
            batch_transforms = torchvision.transforms.Compose(batch_transforms)
        self.batch_transforms = batch_transforms
        self.batch_id_prefix = str(batch_id_prefix)
        if dataset_id_prefix == "AUTO":
            dataset_id_prefix = ssl4rs.utils.filesystem.slugify(type(self.dataset).__name__)
        self.dataset_id_prefix = str(dataset_id_prefix)

    def __len__(self) -> int:
        """Returns the total size (data sample count) of the dataset."""
        return len(self.dataset)

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Returns a single data sample loaded from the dataset.

        If the wrapper was provided with a set of transforms, those will be applied here.
        Afterwards, the batch identifier will be generated and added to the dictionary for the
        loaded sample.
        """
        if np.issubdtype(type(item), np.integer):
            item = int(item)
        batch = self.dataset[item]
        if self.batch_transforms:
            batch = self.batch_transforms(batch)
        assert isinstance(batch, dict), "unexpected data sample type (should be dict)"
        if "batch_id" not in batch:
            prefix = f"{self.batch_id_prefix}_" if self.batch_id_prefix else ""
            batch["batch_id"] = f"{self.dataset_id_prefix}_{prefix}batch{item:08d}"
        return batch
