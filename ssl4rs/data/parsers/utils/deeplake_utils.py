"""Implements parsing utilities for the ActiveLoop DeepLake format."""
import functools
import pathlib
import typing

import deeplake
import torch.utils.data

from ssl4rs.data.parsers.utils.base import DataParser

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchTransformType

DeepLakeParserDerived = typing.TypeVar("DeepLakeParserDerived")


class DeepLakeParser(DataParser):
    """Base interface used to provide common definitions for all deeplake parsers.

    For very simple datasets (e.g. those used for image classification), this wrapper should be
    sufficient to load the data directly. Other datasets that need to preprocess/unpack sample data
    in a specific fashion before it can be used should rely on their own derived class. See the
    `_get_raw_batch()` function specifically for more info.

    Args:
        dataset_path_or_object: path to the deeplake dataset to be read, or deeplake dataset
            object to be wrapped by this reader. Will be set to READ-ONLY if it is not already.
        save_hyperparams: toggles whether hyperparameters should be saved in this class. This
            should be `False` when this class is derived, and the `save_hyperparameters` function
            should be called in the derived constructor.
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
        extra_deeplake_kwargs: extra parameters sent to the deeplake dataset constructor.
            Should not be used if an already-opened dataset is provided.
    """

    def __init__(
        self,
        dataset_path_or_object: typing.Union[typing.AnyStr, pathlib.Path, "deeplake.Dataset"],
        save_hyperparams: bool = True,  # turn this off in derived classes
        batch_transforms: "BatchTransformType" = None,
        add_default_transforms: bool = True,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
        batch_index_key: typing.Optional[str] = None,
        **extra_deeplake_kwargs,
    ):
        """Parses a deeplake archive or wraps an already-opened object.

        Due to the design of this class (and in contrast to the exporter class), all datasets
        should only ever be opened in read-only mode here.

        Note: we should NOT call `self.save_hyperparameters` in this class constructor if it is not
        intended to be used as the FINAL derivation before being instantiated into an object; in other
        words, if you intend on using this class as an interface, turn `save_hyperparams` OFF! See
        these links for more information:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
            https://github.com/Lightning-AI/lightning/issues/16206
        """
        if save_hyperparams:
            self.save_hyperparameters(
                ignore=["dataset_path_or_object", "extra_deeplake_kwargs"],
                logger=False,
            )
        if isinstance(dataset_path_or_object, deeplake.Dataset):
            assert not extra_deeplake_kwargs, "dataset is already opened, can't use extra kwargs"
            dataset = dataset_path_or_object
        else:
            dataset = deeplake.load(dataset_path_or_object, read_only=True, **extra_deeplake_kwargs)
        if not dataset.read_only:
            dataset.read_only = True
        assert (
            hasattr(dataset, "info") and "name" in dataset.info
        ), "dataset info should at least contain a 'name' field!"
        self.dataset = dataset
        super().__init__(
            batch_transforms=batch_transforms,
            add_default_transforms=add_default_transforms,
            batch_id_prefix=batch_id_prefix,
            batch_index_key=batch_index_key,
        )

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
        # as a bonus, we provide the index used to fetch the batch with the default key
        batch[self.batch_index_key] = index
        return batch

    @property
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor info objects (deeplake-defined) from the dataset.

        The returned objects can help downstream processing stages figure out what kind of data they
        will be receiving from this parser.
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
        """Prints a summary of the deeplake dataset using the default logger.

        Note: this might take a while (minutes) with huge datasets!
        """
        # note: this code is derived from the original deeplake dataset's "summary" implementation
        from deeplake.util.pretty_print import summary_dataset

        import ssl4rs.utils.logging

        pretty_print = deeplake.util.pretty_print.summary_dataset(self.dataset)
        logger = ssl4rs.utils.logging.get_logger(__name__)
        logger.info(
            "PRINTING FULL DATASET SUMMARY:"
            f"\ndataset object: {self.dataset}"
            f"\ndataset metadata: {self.dataset.info}"
            f"\ndataset content:\n{pretty_print}"
        )

    def visualize(self, *args, **kwargs):
        """Forwards the call to show the dataset content (notebook-only)"""
        if self.dataset.min_len != self.dataset.max_len:
            raise NotImplementedError("cannot visualize variable length datasets")
        return self.dataset.visualize(*args, **kwargs)

    def query(self, query_str: str) -> "DeepLakeParser":
        """Returns a sliced deeplake dataset with given query results.

        See `deeplake.core.dataset.Dataset.query` for more information.
        """
        if self.dataset.min_len != self.dataset.max_len:
            raise NotImplementedError("cannot query variable length datasets")
        query_result = self.dataset.query(query_str)
        return self.__class__(dataset_path_or_object=query_result, **self.hparams)

    def sample_by(
        self,
        weights: typing.Union[str, list, tuple],
        replace: typing.Optional[bool] = True,
        size: typing.Optional[int] = None,
    ) -> "DeepLakeParser":
        """Returns a sliced deeplake dataset with given weighted sampler applied.

        See `deeplake.core.dataset.Dataset.sample_by` for more information.
        """
        if self.dataset.min_len != self.dataset.max_len:
            raise NotImplementedError("cannot sample variable length datasets")
        sampler_wrapped_dataset = self.dataset.sample_by(
            weights=weights,
            replace=replace,
            size=size,
        )
        return self.__class__(dataset_path_or_object=sampler_wrapped_dataset, **self.hparams)

    def filter(
        self,
        function: typing.Union[typing.Callable, str],
        **filter_kwargs,
    ):
        """Filters the dataset in accordance of filter function `f(x: sample) -> bool`.

        See `deeplake.core.dataset.Dataset.filter` for more information.
        """
        if self.dataset.min_len != self.dataset.max_len:
            raise NotImplementedError("cannot filter variable length datasets")
        filtered_dataset = self.dataset.filter(function=function, **filter_kwargs)
        return self.__class__(dataset_path_or_object=filtered_dataset, **self.hparams)

    def get_dataloader(
        self,
        # arguments for dataloader wrapper: deeplake.enterprise.dataloader.DeepLakeDataLoader.batch
        batch_size: int = 1,
        drop_last: bool = False,
        # arguments for dataloader wrapper: deeplake.enterprise.dataloader.DeepLakeDataLoader.shuffle
        shuffle: bool = False,
        shuffle_buffer_size: int = 2048,
        # arguments for dataloader wrapper: deeplake.enterprise.dataloader.DeepLakeDataLoader.pytorch
        num_workers: int = 0,
        collate_fn: typing.Optional[typing.Callable] = None,
        tensors: typing.Optional[typing.List[str]] = None,
        num_threads: typing.Optional[int] = None,
        prefetch_factor: int = 2,
        distributed: bool = False,
        return_index: bool = True,
        decode_method: typing.Optional[typing.Dict[str, str]] = None,
        persistent_workers: bool = False,
        # arguments for dataloader construction: deeplake.enterprise.dataloader.dataloader
        ignore_errors: bool = False,
        verbose: bool = False,
        # arguments to toggle the dataloader type
        use_optimized_dataloader: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Returns a deeplake data loader for this data parser object.

        Derived classes may implement/use more complex collate, wrapper, or transform functions. By
        default, we simply forward the default settings to deeplake's dataloader creator.
        """
        if collate_fn is None:
            from ssl4rs.data import default_collate

            collate_fn = default_collate
        if use_optimized_dataloader:
            dataloader = self.dataset.dataloader(ignore_errors=ignore_errors, verbose=verbose).batch(
                batch_size=batch_size,
                drop_last=drop_last,
            )
            if self.batch_transforms:
                dataloader = dataloader.transform(self.batch_transforms)
            if shuffle:
                dataloader = dataloader.shuffle(shuffle=shuffle, buffer_size=shuffle_buffer_size)
            dataloader = dataloader.pytorch(
                num_workers=num_workers,
                collate_fn=collate_fn,
                tensors=tensors,
                num_threads=num_threads,
                prefetch_factor=prefetch_factor,
                distributed=distributed,
                return_index=return_index,
                decode_method=decode_method,
                persistent_workers=persistent_workers,
            )
        else:
            assert not distributed, "missing distributed implementation with fallback dataloader"
            dataloader = deeplake.integrations.pytorch.pytorch.dataset_to_pytorch(
                self.dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=collate_fn,
                pin_memory=False,
                shuffle=shuffle,
                buffer_size=shuffle_buffer_size,
                use_local_cache=False,
                transform=self.batch_transforms,
                tensors=tensors,
                return_index=return_index,
                pad_tensors=False,
                decode_method=decode_method,
                persistent_workers=persistent_workers,
                # cache_size=?
            )
        return dataloader


def get_dataloader(
    parser: DeepLakeParser,
    **deeplake_pytorch_dataloader_kwargs,
) -> torch.utils.data.DataLoader:
    """Returns a deeplake data loader for the given data parser object.

    This will call the `get_dataloader` function from the parser class itself, which may be derived
    from the base class for some datasets.
    """
    assert isinstance(
        parser, DeepLakeParser
    ), f"invalid data parser type to use the deeplake dataloader getter: {type(parser)}"
    return parser.get_dataloader(**deeplake_pytorch_dataloader_kwargs)
