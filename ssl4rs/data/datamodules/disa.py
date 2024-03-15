"""Implements a data module for the AI4H-DISA train/valid/test loaders.

TODO: add some URLs for more info here...
"""

import pathlib
import typing

import hydra
import numpy as np
import torch.nn.functional
import torch.utils.data

import ssl4rs.data.datamodules.utils
import ssl4rs.data.metadata.disa
import ssl4rs.data.parsers.disa
import ssl4rs.data.repackagers.disa
import ssl4rs.data.transforms.boundary
import ssl4rs.utils.config
import ssl4rs.utils.logging

logger = ssl4rs.utils.logging.get_logger(__name__)


class DataModule(ssl4rs.data.datamodules.utils.DataModule):
    """Implementation derived from the standard LightningDataModule for the AI4H-DISA dataset.

    TODO: add description + URLs here...
    """

    metadata = ssl4rs.data.metadata.disa

    # noinspection PyUnusedLocal
    def __init__(
        self,
        data_dir: typing.Union[typing.AnyStr, pathlib.Path],
        dataparser_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
        dataloader_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
        train_val_test_split: typing.Optional[typing.Tuple[float, float, float]] = None,
        deeplake_kwargs: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
    ):
        """Initializes the AI4H-DISA data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.

        Note2: for the train-valid-test split, we actually use Sherrie Wang's original split by
        default (if the `train_val_test_split` argument is left as `None`).

        Args:
            data_dir: directory where the AI4H-DISA dataset is located.
            dataparser_configs: configuration dictionary of data parser settings, separated by
                data subset type, which may contain for example definitions for data transforms.
            dataloader_configs: configuration dictionary of data loader settings, separated by data
                subset type, which may contain for example batch sizes and worker counts.
            train_val_test_split: split proportions to use when separating the data into subsets.
                If `None`, will use Sherrie Wang's original split.
            deeplake_kwargs: extra arguments forwarded to the deeplake dataset parser.
        """
        self.save_hyperparameters(logger=False)
        super().__init__(dataparser_configs=dataparser_configs, dataloader_configs=dataloader_configs)
        data_dir = pathlib.Path(data_dir)
        assert data_dir.is_dir(), f"invalid AI4H-DISA dataset directory: {data_dir}"
        if data_dir.name != ".deeplake":
            deeplake_subdir = data_dir / ".deeplake"
        logger.info(f"AI4H-DISA dataset root: {data_dir}")
        if train_val_test_split is not None:
            assert len(train_val_test_split) == 3 and sum(train_val_test_split) == 1.0
        self.data_train: typing.Optional[ssl4rs.data.parsers.disa.DeepLakeParser] = None
        self.data_valid: typing.Optional[ssl4rs.data.parsers.disa.DeepLakeParser] = None
        self.data_test: typing.Optional[ssl4rs.data.parsers.disa.DeepLakeParser] = None

    @property
    def _base_dataparser_configs(self) -> ssl4rs.utils.DictConfig:
        """Returns the 'base' (class-specific-default) configs for the data parsers."""
        return {
            "_default_": {  # all data parsers will be based on the internal disa parser class
                "_target_": "ssl4rs.data.parsers.disa.DeepLakeParser",
            },
            # bonus: we will also give all parsers have a nice prefix
            "train": {"batch_id_prefix": "train"},
            "valid": {"batch_id_prefix": "valid"},
            "test": {"batch_id_prefix": "test"},
        }

    @property
    def num_classes(self) -> int:
        """Returns the number of labels/classes in the dataset."""
        return 2  # this is a binary segmentation dataset (i.e. field/not-field)

    def _is_preparation_complete(self) -> bool:
        """Returns whether the dataset is prepared for setup/loading or not."""
        root_data_dir = pathlib.Path(self.hparams.data_dir)
        if root_data_dir.name == ".deeplake" and root_data_dir.is_dir():
            return True
        potential_deeplake_subdir = root_data_dir / ".deeplake"
        if potential_deeplake_subdir.is_dir():
            return True
        return False  # can't find the .deeplake subdir

    def prepare_data(self) -> None:
        """Verifies that the AI4H-DISA dataset is already in its deeplake format.

        This method will NOT download the dataset; the repackaging also needs to have happened
        already. For more information, refer to the `ssl4rs.data.repackagers.disa` module.
        """
        if self._is_preparation_complete():
            return
        raise NotImplementedError(
            "the AI4H-DISA dataset cannot be repackaged into a deeplake format directly from this "
            "data module!\n\rrefer to the ssl4rs.data.repackagers.disa module for more info."
        )

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Sets up AI4H-DISA data parsing across train/valid/test sets.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`, so be
        careful not to execute the random split twice! The `stage` can be used to differentiate
        whether it's called before `trainer.fit()` or `trainer.test()`.
        """
        assert self._is_preparation_complete(), "dataset is not ready, call `prepare_data()` first!"
        # load datasets only if they're not loaded already (no matter the requested stage)
        if self.data_train is None:
            deeplake_kwargs = self.hparams.deeplake_kwargs or {}
            root_data_dir = pathlib.Path(self.hparams.data_dir)
            if root_data_dir.name != ".deeplake":
                root_data_dir = root_data_dir / ".deeplake"
            assert root_data_dir.is_dir()
            dataset = ssl4rs.data.parsers.disa.DeepLakeParser(root_data_dir, **deeplake_kwargs)
            if self.hparams.train_val_test_split is not None:
                train_idxs, valid_idxs, test_idxs = torch.utils.data.random_split(
                    dataset=list(range(len(dataset))),  # noqa
                    lengths=self.hparams.train_val_test_split,
                    generator=torch.Generator().manual_seed(self.split_seed),
                )
            else:  # if split not specified, default back to Sherrie Wang's split labels
                subset_labels = dataset.dataset.location_subset.numpy().flatten()
                assert subset_labels.size == len(dataset) and subset_labels.dtype == np.uint32
                train_idxs, valid_idxs, test_idxs = (
                    [
                        sidx
                        for sidx in range(len(dataset))
                        if subset_labels[sidx] == self.metadata.location_subset_labels.index(subset)
                    ]
                    for subset in ["train", "val", "test"]
                )
            train_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "train")
            self.data_train = hydra.utils.instantiate(train_parser_config, dataset.dataset[list(train_idxs)])
            valid_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "valid")
            self.data_valid = hydra.utils.instantiate(valid_parser_config, dataset.dataset[list(valid_idxs)])
            test_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "test")
            self.data_test = hydra.utils.instantiate(test_parser_config, dataset.dataset[list(test_idxs)])
            logger.info(
                "parser setup complete;"
                f"\n\ttrain dataset size = {len(self.data_train)}"
                f"\n\tvalid dataset size = {len(self.data_valid)}"
                f"\n\ttest dataset size = {len(self.data_test)}"
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AI4H-DISA training set data loader."""
        assert self.data_train is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_train, subset_type="train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AI4H-DISA validation set data loader."""
        assert self.data_valid is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_valid, subset_type="valid")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AI4H-DISA testing set data loader."""
        assert self.data_test is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_test, subset_type="test")

def custom_collate(
    batches: typing.List[ssl4rs.data.BatchDictType],
    pad_to_shape: typing.Optional[typing.Tuple[int, int]] = None,
) -> ssl4rs.data.BatchDictType:
    """Defines a custom collate function to deal with non-pytorch-compatible dataset elements."""
    # first, pad to the specified shape (if needed)
    for batch in batches:
        ssl4rs.data.transforms.pad.pad_arrays_in_batch(
            batch=batch,
            pad_tensor_names_and_values=DataModule.metadata.tensor_pad_values,
            pad_to_shape=pad_to_shape,
        )

    for batch in batches:
        batch['image_data'] = batch['image_data'].squeeze()
        batch['field_mask'] = batch['field_mask'].squeeze()

    # second, do the actual collate while bypassing torch for the funkier arrays
    output = ssl4rs.data.default_collate(
        batches=batches,
        keys_to_batch_manually=DataModule.metadata.tensor_names_to_collate_manually,
    )
    return output


def convert_deeplake_tensors_to_pytorch_tensors(
    batch: ssl4rs.data.BatchDictType,
    normalize_input_tensors: bool = False,
    mask_input_tensors: bool = False,
) -> ssl4rs.data.BatchDictType:
    """Transform used in parser class to convert tensors to pytorch-model-ready format."""
    assert isinstance(batch, dict)
    convert_names = DataModule.metadata.tensor_names_to_convert
    ch_transp_names = DataModule.metadata.tensor_names_to_transpose_ch
    norm_names = DataModule.metadata.tensor_names_to_normalize if normalize_input_tensors else []
    norm_stats = DataModule.metadata.tensor_normalization_stats
    for tname, tval in batch.items():
        if tname in norm_names:
            if tname == "image_data":
                # special handling: stats depend on number of bands in the data
                ch = tval.shape[1]
                assert ch in DataModule.metadata.supported_band_counts
                mean, std = norm_stats[tname][ch]["mean"], norm_stats[tname][ch]["std"]
                # we also need to unsqueeze the mean/std arrays to fit the NxCxHxW image data format
                mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
                std = std[np.newaxis, :, np.newaxis, np.newaxis]
            else:
                mean, std = norm_stats[tname]["mean"], norm_stats[tname]["std"]
            tval = (tval.astype(np.float32) - mean) / std
            if mask_input_tensors:
                # bonus: zero-out values in normalized tensors according to ROIs
                if tname == "location_preview_image" and "location_preview_roi" in batch:
                    invalid_px_mask = np.logical_not(batch["location_preview_roi"])
                    tval[invalid_px_mask, :] = 0
                elif tname == "image_data" and "image_roi" in batch:
                    invalid_px_mask = np.logical_not(batch["image_roi"])[:, np.newaxis, :, :]
                    invalid_px_mask = np.broadcast_to(invalid_px_mask, batch["image_data"].shape)
                    tval[invalid_px_mask] = 0
        elif tname == "image_data":
            # special handling: convert uint16 data to float32 even when not normalizing
            tval = tval.astype(np.float32)
        if tname == "field_mask":
            # special handling: convert boolean mask data to class labels even when not normalizing
            tval = tval.astype(np.int64)
            if "location_preview_roi" in batch:
                # todo @@@@@: update masking of invalid input pixels in target mask based on each image?
                invalid_px_mask = np.logical_not(batch["location_preview_roi"])
                tval[invalid_px_mask, :] = DataModule.metadata.dontcare_label
        if tname in ch_transp_names:
            assert isinstance(tval, np.ndarray) and tval.ndim == 3
            tval = tval.transpose(2, 0, 1)  # HxWxC to CxHxW
        if tname in convert_names:
            assert isinstance(tval, np.ndarray)
            tval = torch.as_tensor(tval)
        batch[tname] = tval
    return batch


def generate_field_boundary_mask(
    batch: ssl4rs.data.BatchDictType,
    output_field_boundary_mask_name: str = "field_boundary_mask",
) -> ssl4rs.data.BatchDictType:
    """Transform used in parser class to generate field boundary (contour) masks."""
    assert isinstance(batch, dict)
    assert "field_mask" in batch
    class_map = batch["field_mask"]
    unannotated_mask = class_map == 0  # this is the real 'dontcare' which we reapply below
    boundary_mask = ssl4rs.data.transforms.boundary.generate_boundary_mask_from_class_label_map(
        class_label_map=class_map,
        target_class_label=1,  # target the "positive" (field) class inside the binary mask
        ignore_index=DataModule.metadata.dontcare_label,
    )
    boundary_mask[unannotated_mask] = DataModule.metadata.dontcare_label
    if isinstance(class_map, torch.Tensor):
        # need to convert the new mask to the same format
        boundary_mask = torch.as_tensor(boundary_mask).to(device=class_map.device)
    batch[output_field_boundary_mask_name] = boundary_mask
    return batch


def _local_main(config) -> None:
    import hydra.utils

    datamodule = hydra.utils.instantiate(config.data.datamodule)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    minibatch = next(iter(train_dataloader))
    assert isinstance(minibatch, dict)
    logger.info("all done")


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config(
        overrides=["data=disa.yaml"],
    )
    _local_main(config_)
