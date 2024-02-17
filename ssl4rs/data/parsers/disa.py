"""Implements a data parser for the DISA dataset."""

import typing

import deeplake
import numpy as np

import ssl4rs.data.metadata.disa
import ssl4rs.data.parsers.utils

if typing.TYPE_CHECKING:
    import pathlib

    from ssl4rs.data import BatchTransformType


class DeepLakeParser(ssl4rs.data.parsers.utils.DeepLakeParser):
    """We override the base deeplake parser to add a flag to convert the uint16 data.

    See the base class documentation for more information on role + arguments. The only novel
    argument in this derived class is the `convert_uint16_data` flag, which will be used to
    automatically convert image data (originally uint16) into float32 format in order to avoid
    pytorch cast issues.

    The tensors that are expected by this parser are the ones defined by the corresponding
    deeplake dataset repackager, `ssl4rs.data.repackagers.disa.DeepLakeRepackager`. These are
    also described in the `ssl4rs.data.metadata.disa` module.
    """

    metadata = ssl4rs.data.metadata.disa

    def __init__(
        self,
        dataset_path_or_object: typing.Union[typing.AnyStr, "pathlib.Path", deeplake.Dataset],
        convert_uint16_data: bool = True,
        save_hyperparams: bool = True,  # turn this off in derived classes
        batch_transforms: "BatchTransformType" = None,
        add_default_transforms: bool = True,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
        batch_index_key: typing.Optional[str] = None,
        **extra_deeplake_kwargs,
    ):
        """Parses a deeplake archive or wraps an already-opened object.

        See the base class constructor for more info.
        """
        if save_hyperparams:
            self.save_hyperparameters(
                ignore=["dataset_path_or_object", "extra_deeplake_kwargs"],
                logger=False,
            )
        self.convert_uint16_data = convert_uint16_data
        using_dataset_obj = isinstance(dataset_path_or_object, deeplake.Dataset)
        if not using_dataset_obj and "check_integrity" not in extra_deeplake_kwargs:
            # add check to get rid of annoying warning caused by having many tensors
            extra_deeplake_kwargs["check_integrity"] = True
        super().__init__(
            dataset_path_or_object=dataset_path_or_object,
            save_hyperparams=False,  # we saved hparams above, no need to do it in the base class
            batch_transforms=batch_transforms,
            add_default_transforms=add_default_transforms,
            batch_id_prefix=batch_id_prefix,
            batch_index_key=batch_index_key,
            **extra_deeplake_kwargs,
        )

    def _get_raw_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch loaded from the deeplake dataset at a specified index."""
        data = self.dataset[index]  # noqa
        batch = {tensor_name: data[tensor_name].numpy() for tensor_name in self.tensor_names}
        batch[self.batch_index_key] = index
        if self.convert_uint16_data:
            batch["image_data"] = batch["image_data"].astype(np.float32)
        return batch
