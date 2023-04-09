import typing

import ssl4rs.utils.logging
from ssl4rs.data.parsers.utils.base import DataParser

logger = ssl4rs.utils.logging.get_logger(__name__)


class ParserWrapper(DataParser):
    """Base interface used to wrap generic data parsers (e.g. from other DL frameworks).

    This is meant to provide a compatibility layer between data parsers that might be imported from
    other framework and that do not have a simple way to interface with the rest of this framework,
    e.g. to apply batch-level transformation operations and generate batch identifiers.
    """

    def __init__(
        self,
        dataset: typing.Any,
        dataset_name: typing.AnyStr = "AUTO",  # if 'AUTO', will use wrapped object class name
        batch_transforms: "ssl4rs.data.BatchTransformType" = None,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
    ):
        """Validates that the provided PyTorch-Dataset-compatible object can be wrapped."""
        super().__init__(batch_transforms=batch_transforms, batch_id_prefix=batch_id_prefix)
        assert hasattr(dataset, "__len__"), "missing mandatory dataset length attribute!"
        assert hasattr(dataset, "__getitem__"), "missing mandatory dataset item getter!"
        self.dataset = dataset  # should be read-only, as we'll cache the dataset size
        self._dataset_size: typing.Optional[int] = None
        if dataset_name == "AUTO":
            dataset_name = ssl4rs.utils.filesystem.slugify(type(self.dataset).__name__)
        self._dataset_name = str(dataset_name)
        self._tensor_names: typing.List[str] = []  # will be filled when needed/available

    def __len__(self) -> int:
        """Returns the total size (in terms of data batch count) of the dataset.

        It might be called fairly often to validate indexing ranges, so in order to avoid issues
        with dataset implementations that re-iterate over their entire data in order to figure out
        the batch count each time this function is called, we cache that value.
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
        return self.dataset[index]

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

        This function should be easy-to-call (parameter-free, if possible) and fast-to-return
        (takes seconds or tens-of-seconds at most) in order to remain friendly to high-level users.
        What it does specifically is totally up to the derived class.

        All outputs should be sent to the default logger.
        """
        logger.info(self)
        logger.info(f"dataset_name={self.dataset_name}, length={len(self)}")
