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
    ):
        """Parses a hub dataset file or wraps an already-opened object.

        Note that due to the design of this class (and in contrast to the exporter class), all
        datasets should only ever be opened in read-only mode here.
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

    def __len__(self) -> int:
        """Returns the total size (data sample count) of the dataset."""
        return len(self.dataset)

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Returns a single data sample loaded from the dataset.

        The data sample is provided as a dictionary where the `tensor_names` property defined
        above should each be the keys to tensors. Additional tensors may also be returned.
        """
        if np.issubdtype(type(item), np.integer):
            item = int(item)
        batch = self.dataset[item]
        if self.batch_transforms:
            batch = self.batch_transforms(batch)
        if "batch_id" not in batch:
            prefix = f"{self.batch_id_prefix}_" if self.batch_id_prefix else ""
            type_slug = ssl4rs.utils.filesystem.slugify(type(self.dataset).__name__)
            batch["batch_id"] = f"{prefix}{type_slug}_batch{item:08d}"
        return batch
