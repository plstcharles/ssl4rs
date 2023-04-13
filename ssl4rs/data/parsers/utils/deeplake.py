"""Implements parsing utilities for the ActiveLoop DeepLake format."""

import typing

import deeplake
import deeplake.util.pretty_print

import ssl4rs.utils.logging
from ssl4rs.data.parsers.utils.base import DataParser

logger = ssl4rs.utils.logging.get_logger(__name__)
DeepLakeParserDerived = typing.TypeVar("DeepLakeParserDerived")


class DeepLakeParser(DataParser):
    """Base interface used to provide common definitions for all deeplake parsers.

    For very simple datasets (e.g. those used for image classification), this wrapper should be
    sufficient to load the data directly. Other datasets that need to preprocess/unpack sample data
    in a specific fashion before it can be used should rely on their own derived class. See the
    `_get_raw_batch()` function specifically for more info.
    """

    def __init__(
        self,
        dataset_path_or_object: typing.Union[typing.AnyStr, deeplake.Dataset],
        batch_transforms: "ssl4rs.data.BatchTransformType" = None,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
        **extra_deeplake_kwargs,
    ):
        """Parses a deeplake archive or wraps an already-opened object.

        Note that due to the design of this class (and in contrast to the exporter class), all
        datasets should only ever be opened in read-only mode here.
        """
        super().__init__(batch_transforms=batch_transforms, batch_id_prefix=batch_id_prefix)
        if isinstance(dataset_path_or_object, deeplake.Dataset):
            assert not extra_deeplake_kwargs, "dataset is already opened, can't use extra kwargs"
            self.dataset = dataset_path_or_object
        else:
            self.dataset = deeplake.load(dataset_path_or_object, read_only=True, **extra_deeplake_kwargs)
        assert (
            hasattr(self.dataset, "info") and "name" in self.dataset.info
        ), "dataset info should at least contain a 'name' field!"

    def __len__(self) -> int:
        """Returns the total size (in terms of data batch count) of the dataset."""
        return len(self.dataset)

    def _get_raw_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch loaded from the deeplake dataset at a specified index.

        In contrast with the `__getitem__` function, this internal call will not apply transforms.
        """
        # the following line will fetch the corresponding batch data for the given index across
        # all deeplake tensors stored in the dataset (assuming they're all indexed identically)
        data = self.dataset[index]  # noqa
        # we will now convert all these tensors to numpy arrays, which might not be adequate for
        # funky datasets composed of non-stackable arrays (e.g. when using images with different
        # shapes, when loading json data, when decoding raw bytes, etc.) ... in those cases, you
        # should derive and implement your own version of this function to do your own unpacking!
        batch = {tensor_name: data[tensor_name].numpy() for tensor_name in self.tensor_names}
        return batch

    @property
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor info objects (deeplake-defined) from the dataset.

        The returned objects can help downstream processing stages figure out what kind of data
        they will be receiving from this parser.
        """
        return {k: v.info for k, v in self.dataset.tensors.items()}

    @property
    def tensor_names(self) -> typing.List[str]:
        """Names of the data tensors that will be provided in the loaded batches.

        Note that additional tensors and other attributes may be loaded as well, but these are the
        'primary' fields that should be expected by downstream processing stages.
        """
        return list(self.dataset.tensors.keys())

    @property
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information parsed from the deeplake object."""
        return dict(self.dataset.info)

    @property
    def dataset_name(self) -> typing.AnyStr:
        """Returns the dataset name used to identify this particular dataset."""
        return self.dataset.info["name"]

    def summary(self) -> None:
        """Prints a summary of the deeplake dataset using the default logger."""
        # note: this code is derived from the original deeplake dataset's "summary" implementation
        pretty_print = deeplake.util.pretty_print.summary_dataset(self.dataset)
        logger.info(self)
        logger.info(self.dataset)
        logger.info(pretty_print)

    def visualize(self, *args, **kwargs):
        """Forwards the call to show the dataset content (notebook-only)"""
        return self.dataset.visualize(*args, **kwargs)


def get_deeplake_parser_subset(
    parser: DeepLakeParserDerived,
    indices: typing.Sequence[int],
) -> DeepLakeParserDerived:
    """Returns a parser for a subset of a deeplake."""

    assert isinstance(parser, DeepLakeParser), "need to derive from DeepLakeParser!"
    assert hasattr(parser, "dataset") and isinstance(parser.dataset, deeplake.Dataset)
    assert all([idx < len(parser) for idx in indices]), "some indices are out-of-range!"
    TODO  # @@@@@@@@@@ TODO IMPL ME! (might be much faster than regular) # noqa: F821
    # OR REPLACE EXISTING SPLIT LOGIC w/ deeplake.core.dataset.random_split @@@@@@ !
    # ... and then save splits as 'views' (save_view) that can be reloaded easily

    # @deeplake.compute
    def filter_indices(sample_in) -> bool:
        return todo  # noqa: F821

    subset_parser = parser.dataset.filter(filter_indices)
    return type(parser)(subset_parser)
