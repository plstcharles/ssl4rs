import typing

import hub
import torch.utils.data.dataset


class HubDatasetParser(torch.utils.data.dataset.Dataset):
    """Base interface used to provide common definitions for all dataset parsers.

    For very simple datasets (e.g. those used for image classification), this wrapper should be
    sufficient to load the data directly. Other datasets that need to preprocess/unpack sample
    data in a specific fashion before it can be used should rely on their own derived class.
    """

    @property
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor info objects (hub-defined) parsed from the dataset.

        The returned objects can help downstream processing stages figure out what kind of data
        they will be receiving from this parser.
        """
        return {k: v.info for k, v in self.dataset.tensors.items()}

    @property
    def tensor_names(self) -> typing.List[str]:
        """Names of the tensors that will be provided in the loaded data samples.

        Note that additional tensors and other attributes may be loaded as well, but these are the
        'primary' fields that should be expected by downstream processing stages.
        """
        return list(self.tensor_info.keys())

    @property
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information parsed from the hub dataset object."""
        return dict(self.dataset.info)

    @property
    def dataset_name(self) -> typing.AnyStr:
        """Returns the dataset name used to identify this particular dataset."""
        return self.dataset_info["name"]

    def __init__(
        self,
        dataset_path_or_object: typing.Union[typing.AnyStr, hub.Dataset],
        **extra_hub_kwargs,
    ):
        """Parses a hub dataset file or wraps an already-opened object.

        Note that due to the design of this class (and in contrast to the exporter class), all
        datasets should only ever be opened in read-only mode here.
        """
        if isinstance(dataset_path_or_object, hub.Dataset):
            assert not extra_hub_kwargs, "dataset is already opened, can't use kwargs"
            self.dataset = dataset_path_or_object
        else:
            self.dataset = hub.load(dataset_path_or_object, read_only=True, **extra_hub_kwargs)

    def __len__(self) -> int:
        """Returns the total size (data sample count) of the dataset."""
        return len(self.dataset)

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Returns a single data sample loaded from the dataset.

        The data sample is provided as a dictionary where the `tensor_names` property defined
        above should each be the keys to tensors. Additional tensors may also be returned.
        """
        return self.dataset[item]

    def summary(self) -> None:
        """Forwards the call to print a summary of the dataset."""
        return self.dataset.summary()

    def visualize(self, *args, **kwargs):
        """Forwards the call to show the dataset content (notebook-only)"""
        return self.dataset.visualize(*args, **kwargs)
