"""Implements a data module for the UC Merced train/valid/test loaders.

See the following URL(s) for more info on this dataset:
http://weegee.vision.ucmerced.edu/datasets/landuse.html
http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
"""

import pathlib
import typing

import hydra
import torch.utils.data

import ssl4rs.data.datamodules.utils
import ssl4rs.data.metadata.ucmerced
import ssl4rs.data.parsers.ucmerced
import ssl4rs.data.repackagers.ucmerced
import ssl4rs.utils.config

logger = ssl4rs.utils.logging.get_logger(__name__)


class DataModule(ssl4rs.data.datamodules.utils.DataModule):
    """Implementation derived from the standard LightningDataModule for the UC Merced dataset.

    This dataset contains 256x256 images with a GSD of ~0.3m that can be used for classification.
    There are 21 classes and 21x100=2100 images in this dataset.

    If the dataset does not already exist at the specified path, it will be downloaded there. The
    zip size is roughly 330MB.

    For more info, see the dataset pages here:
    http://weegee.vision.ucmerced.edu/datasets/landuse.html
    https://www.tensorflow.org/datasets/catalog/uc_merced
    """

    metadata = ssl4rs.data.metadata.ucmerced

    # noinspection PyUnusedLocal
    def __init__(
        self,
        data_dir: typing.Union[typing.AnyStr, pathlib.Path],
        dataparser_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
        dataloader_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
        train_val_test_split: typing.Tuple[float, float, float] = (0.8, 0.1, 0.1),
        deeplake_kwargs: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
    ):
        """Initializes the UC Merced Land Use data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.

        Args:
            data_dir: directory where the UC Merced Land Use dataset is located.
            dataparser_configs: configuration dictionary of data parser settings, separated by
                data subset type, which may contain for example definitions for data transforms.
            dataloader_configs: configuration dictionary of data loader settings, separated by data
                subset type, which may contain for example batch sizes and worker counts.
            train_val_test_split: split proportions to use when separating the data.
            deeplake_kwargs: extra arguments forwarded to the deeplake dataset parser.
        """
        self.save_hyperparameters(logger=False)
        super().__init__(dataparser_configs=dataparser_configs, dataloader_configs=dataloader_configs)
        assert data_dir is not None
        data_dir = pathlib.Path(data_dir)  # it might not exist, in which case we'll do a download
        logger.debug(f"UC Merced Land Use dataset root: {data_dir}")
        assert len(train_val_test_split) == 3 and sum(train_val_test_split) == 1.0
        self.data_train: typing.Optional[ssl4rs.data.parsers.ucmerced.DeepLakeParser] = None
        self.data_valid: typing.Optional[ssl4rs.data.parsers.ucmerced.DeepLakeParser] = None
        self.data_test: typing.Optional[ssl4rs.data.parsers.ucmerced.DeepLakeParser] = None

    @property
    def _base_dataparser_configs(self) -> ssl4rs.utils.DictConfig:
        """Returns the 'base' (class-specific-default) configs for the data parsers."""
        return {
            "_default_": {  # all data parsers will be based on the internal ucmerced parser class
                "_target_": "ssl4rs.data.parsers.ucmerced.DeepLakeParser",
            },
            # bonus: we will also give all parsers have a nice prefix
            "train": {"batch_id_prefix": "train"},
            "valid": {"batch_id_prefix": "valid"},
            "test": {"batch_id_prefix": "test"},
        }

    @property
    def num_classes(self) -> int:
        """Returns the number of labels/classes in the dataset."""
        return len(self.metadata.class_names)

    @classmethod
    def _download_and_unpack(cls, dataset_root_path: pathlib.Path) -> None:
        """Downloads and unpacks the UC Merced Land Use dataset to the specified path."""
        dataset_root_path.mkdir(parents=True, exist_ok=True)
        zip_path = ssl4rs.utils.filesystem.download_file(
            url=cls.metadata.zip_download_url,
            root=dataset_root_path,
            filename="UCMerced_LandUse.zip",
            md5=cls.metadata.zip_file_md5_hash,
        )
        ssl4rs.utils.filesystem.extract_zip(zip_path, dataset_root_path)
        extected_dir_path = dataset_root_path / "UCMerced_LandUse"
        assert extected_dir_path.is_dir()
        for class_name in cls.metadata.class_names:
            class_name_slug = ssl4rs.utils.filesystem.slugify(class_name)
            expected_dir_path = extected_dir_path / "Images" / class_name_slug
            assert expected_dir_path.is_dir(), f"missing class dir: {expected_dir_path}"
            target_dir_path = dataset_root_path / class_name_slug
            assert not target_dir_path.exists()
            expected_dir_path.rename(target_dir_path)
        ssl4rs.utils.filesystem.recursively_remove_all(extected_dir_path)

    def _is_preparation_complete(self) -> bool:
        """Returns whether the dataset is prepared for setup/loading or not."""
        root_data_dir = pathlib.Path(self.hparams.data_dir)
        if root_data_dir.name == ".deeplake" and root_data_dir.is_dir():
            return True
        potential_deeplake_subdir = root_data_dir / ".deeplake"
        if potential_deeplake_subdir.is_dir():
            return True
        return False  # can't find the .deeplake subdir, we'll likely have to create it

    def prepare_data(self) -> None:
        """Will download and repackage the raw UC Merced dataset, if necessary."""
        if self._is_preparation_complete():
            return
        root_data_dir = pathlib.Path(self.hparams.data_dir)
        if not root_data_dir.exists():
            logger.warning(f"UC Merced Land Use dataset missing from: {root_data_dir}")
            logger.warning("will download and extract the zip contents in that location...")
            self._download_and_unpack(root_data_dir)
        logger.info("Starting deep lake repackager for UC Merced dataset...")
        repackager = ssl4rs.data.repackagers.ucmerced.DeepLakeRepackager(root_data_dir)
        deeplake_output_path = root_data_dir / ".deeplake"
        repackager.export(deeplake_output_path)
        assert pathlib.Path(deeplake_output_path).exists()

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Sets up UC Merced Land Use data parsing across train/valid/test sets.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`, so be
        careful not to execute the random split twice! The `stage` can be used to differentiate
        whether it's called before `trainer.fit()` or `trainer.test()`.

        Note that the data splitting happening here will be stratified, i.e. we will try to provide
        the requested balance of data for the three subsets across each dataset class.
        """
        assert self._is_preparation_complete(), "dataset is not ready, call `prepare_data()` first!"
        # load datasets only if they're not loaded already (no matter the requested stage)
        if self.data_train is None:
            deeplake_kwargs = self.hparams.deeplake_kwargs or {}
            root_data_dir = pathlib.Path(self.hparams.data_dir)
            if root_data_dir.name != ".deeplake":
                root_data_dir = root_data_dir / ".deeplake"
            assert root_data_dir.is_dir()
            logger.debug(f"UC Merced deeplake dataset root: {root_data_dir}")
            dataset = ssl4rs.data.parsers.ucmerced.DeepLakeParser(root_data_dir, **deeplake_kwargs)
            train_idxs, valid_idxs, test_idxs = self._get_subset_idxs_with_stratified_sampling(
                # with only 10k samples, all labels should fit into memory without any worries
                labels=dataset.dataset.label.numpy().flatten(),
                class_idx_to_count_map={idx: c for idx, c in enumerate(self.metadata.class_distrib.values())},
                split_ratios=self.hparams.train_val_test_split,
            )
            train_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "train")
            self.data_train = hydra.utils.instantiate(train_parser_config, dataset.dataset[train_idxs])
            valid_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "valid")
            self.data_valid = hydra.utils.instantiate(valid_parser_config, dataset.dataset[valid_idxs])
            test_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "test")
            self.data_test = hydra.utils.instantiate(test_parser_config, dataset.dataset[test_idxs])

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AID training set data loader."""
        assert self.data_train is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_train, subset_type="train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AID validation set data loader."""
        assert self.data_valid is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_valid, subset_type="valid")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AID testing set data loader."""
        assert self.data_test is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_test, subset_type="test")


def _local_main(data_root_dir: pathlib.Path) -> None:
    datamodule = DataModule(data_dir=data_root_dir / "UCMerced_LandUse")
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    minibatch = next(iter(train_dataloader))
    assert isinstance(minibatch, dict)
    valid_dataloader = datamodule.val_dataloader()
    minibatch = next(iter(valid_dataloader))
    assert isinstance(minibatch, dict)
    test_dataloader = datamodule.test_dataloader()
    minibatch = next(iter(test_dataloader))
    assert isinstance(minibatch, dict)


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config(config_name="profiler.yaml")
    _local_main(pathlib.Path(config_.utils.data_root_dir))
