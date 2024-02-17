"""Implements generic data repackaging and utilities for common dataset formats."""

import abc
import pathlib
import typing

import deepdiff
import deeplake
import deeplake.util.exceptions
import numpy as np
import tqdm

import ssl4rs.utils.config
import ssl4rs.utils.logging
from ssl4rs.data import BatchDictType

logger = ssl4rs.utils.logging.get_logger(__name__)


class DeepLakeRepackager:
    """Base interface used to provide common definitions for all deeplake exporters/repackagers.

    The abstract properties defined below should be overridden in the derived classes based on the
    dataset info, either statically or at runtime.
    """

    @property
    @abc.abstractmethod
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor declarations used during repackaging.

        Note: the arguments used for each key in the returned dictionary should be usable directly
        in the `deeplake.Dataset.create_tensor` function; see
        https://api-docs.activeloop.ai/#deeplake.Dataset.create_tensor for more information.
        """
        raise NotImplementedError

    @property
    def tensor_names(self) -> typing.List[str]:
        """Names of the tensors that will be exported in the dataset object."""
        return list(self.tensor_info.keys())

    @property
    @abc.abstractmethod
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information that will be exported in the dataset object.

        Note: the values provided in the dictionary should be compatible with the values that
        can be exported in the `deeplake.Dataset.info` object; see
        https://api-docs.activeloop.ai/#deeplake.Dataset.info for more information.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dataset_name(self) -> typing.AnyStr:
        """Returns the dataset name used to identify this particular dataset.

        Note: if this name is not present in the dataset info (under the `name` key), it will be
        added automatically when exporting the dataset.
        """
        raise NotImplementedError

    def __init__(self):
        """Does nothing; the derived class is responsible for parsing the data to be repackaged."""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the total size (sample count) of the dataset that will be exported."""
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, item: int) -> BatchDictType:
        """Returns a single data sample that should be exported in the dataset.

        The data sample is provided as a dictionary where the `tensor_names` property defined above
        should each be the key to a tensor. Other tensors may also be present.
        """
        raise NotImplementedError

    def _export_sample_data(self, sample_index, sample_out):
        """Fetches a data sample from the getitem implementation and appends it to the output."""
        # this default implementation assumes all tensor datasets have the same length; override this
        # function if this is not the case, as exportation will fail otherwise!
        sample_data = self[sample_index]  # this is where the __getitem__ is called...
        if sample_data is not None:
            assert isinstance(sample_data, dict) and all([tn in sample_data for tn in self.tensor_names])
            sample_out.append(sample_data)  # this should add all the tensors at once...
        return sample_out

    @staticmethod
    @deeplake.compute
    def _data_sample_exporter(sample_index, sample_out, exporter):
        """Fetches a data sample from a derived getitem implementation for exportation."""
        exporter._export_sample_data(sample_index, sample_out)  # noqa

    def _finalize_dataset_export(
        self,
        dataset: deeplake.Dataset,
        num_workers: int,
    ) -> None:
        """Finalizes the exportation of the deeplake dataset, adding extra info as needed."""
        # by default, we do nothing here, as the default implementation has nothing to add
        pass

    def export(
        self,
        output_path: typing.Union[typing.AnyStr, pathlib.Path],
        overwrite: bool = False,
        verbose: bool = True,
        num_workers: int = 4,
        scheduler: str = "threaded",
        progressbar: bool = True,
        **extra_deeplake_kwargs,
    ) -> deeplake.Dataset:
        """Exports the dataset to the specified location.

        By default, if a file exists at the specified path, we will not be overwriting it; we will
        instead verify that it is the same dataset with the same content, and throw an exception
        otherwise.

        Note that with deeplake, the output path can be a local path or a remote (server) path under
        the form `PROTOCOL://SERVERNAME/DATASETNAME`. Deeplake will take care of exporting the data
        during the dataset creation. In-memory datasets can also be created using this exporter;
        simply specify a dummy path with a `mem://` prefix, e.g. `mem://dummy/path`.

        Once exported, the dataset object that was created (or that was found) will be returned.
        """
        output_path = str(output_path)
        # first, figure out if the dataset exists and whether we need to validate it...
        if not overwrite:
            existing_dataset = None
            try:
                existing_dataset = deeplake.load(
                    output_path,
                    read_only=True,
                    verbose=verbose,
                    **extra_deeplake_kwargs,
                )
            except deeplake.util.exceptions.DatasetHandlerError:
                pass  # dataset does not exist, we won't be overwriting anything, perfect
            if existing_dataset is not None:
                # if we get here, we need to verify dataset overlap...
                check_info_overlap(existing_dataset, self.dataset_info, self.tensor_info)
                # if the above check passed, we can return right away, as the dataset is all good!
                return existing_dataset
            dataset = deeplake.empty(output_path, overwrite=False, **extra_deeplake_kwargs)
        else:
            dataset = deeplake.empty(output_path, overwrite=True, **extra_deeplake_kwargs)

        # time to do the actual export now!
        with dataset:  # this will make sure the export is cached/buffered properly
            dataset_info = self.dataset_info
            if "name" in dataset_info:
                assert dataset_info["name"] == self.dataset_name
            else:
                dataset_info["name"] = self.dataset_name
            dataset.info.update(dataset_info)
            assert not np.setdiff1d(self.tensor_names, list(self.tensor_info.keys()))
            for tensor_name, tensor_kwargs in self.tensor_info.items():
                assert isinstance(tensor_name, str) and isinstance(tensor_kwargs, dict)
                dataset.create_tensor(name=tensor_name, **tensor_kwargs)
            sample_idxs = list(range(len(self)))
            assert num_workers >= 0, f"invalid number of workers: {num_workers}"
            if num_workers > 0:
                self._data_sample_exporter(self).eval(
                    data_in=sample_idxs,
                    ds_out=dataset,
                    num_workers=num_workers,
                    scheduler=scheduler,
                    progressbar=progressbar,
                )
            else:
                if progressbar:
                    sample_idxs = tqdm.tqdm(sample_idxs, desc=f"exporting {self.dataset_name}")
                for sample_idx in sample_idxs:
                    self._export_sample_data(sample_idx, dataset)
        # finalize after writing the first batch of data (if needed)
        with dataset:
            self._finalize_dataset_export(dataset=dataset, num_workers=num_workers)
        # all done!
        size_approx_mb = dataset.size_approx() // (1024 * 1024)
        logger.debug(f"export complete, approx size = {size_approx_mb} MB")
        return dataset


def check_info_overlap(
    dataset: deeplake.Dataset,
    expected_dataset_info: typing.Dict[typing.AnyStr, typing.Any],
    expected_tensor_info: typing.Dict[typing.AnyStr, typing.Dict[typing.AnyStr, typing.Any]],
    strict: bool = False,
) -> None:
    """Checks whether the provided dataset's info matches with the given info dictionary.

    If using a strict check, any attributes that are mismatched will result in a failed comparison
    (and a thrown exception). Otherwise, the exporter info may differ without failing the check.
    """
    found_dataset_info = dict(dataset.info)
    runtime_info = ssl4rs.utils.config.get_runtime_tags()
    if not strict:
        found_dataset_info = {k: v for k, v in found_dataset_info.items() if k not in runtime_info}
        expected_dataset_info = {k: v for k, v in expected_dataset_info.items() if k not in runtime_info}
    diff_res = deepdiff.DeepDiff(found_dataset_info, expected_dataset_info)
    if diff_res:
        raise RuntimeError(
            "dataset metadata comparison failed!\n"
            f"\texpected dataset info: {str(expected_dataset_info)}\n"
            f"\tfound dataset info: {str(found_dataset_info)}\n"
            f"\tdataset info diff: {str(diff_res)}"
        )
    for tname, tinfo in expected_tensor_info.items():
        if tname not in dataset.tensors:
            raise RuntimeError(f"dataset is missing tensor with name: {tname}")
