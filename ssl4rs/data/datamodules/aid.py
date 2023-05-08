"""Implements a data module for the AID train/valid/test loaders.

See the following URL for more info on this dataset: https://captain-whu.github.io/AID/
"""

import pathlib
import typing

import hydra
import omegaconf
import torch.utils.data

import ssl4rs.data.datamodules.utils
import ssl4rs.data.metadata.aid
import ssl4rs.data.parsers.aid
import ssl4rs.data.repackagers.aid
import ssl4rs.utils.config
import ssl4rs.utils.logging

logger = ssl4rs.utils.logging.get_logger(__name__)


class DataModule(ssl4rs.data.datamodules.utils.DataModule):
    """Implementation derived from the standard LightningDataModule for the AID dataset.

    This dataset contains large-scale aerial images that can be used for classification. There are
    10,000 images (600x600, RGB) in this dataset, and these are given one of 30 class labels. See
    https://captain-whu.github.io/AID/ for more information and download links. This dataset CANNOT
    be downloaded on the spot by this data module, meaning we assume it is on disk at runtime.

    Note that this dataset does NOT have a fixed Ground Sampling Distance (GSD); images contained
    herein are mixed across different sources with GSDs between 0.5m and 8m.
    """

    metadata = ssl4rs.data.metadata.aid

    # noinspection PyUnusedLocal
    def __init__(
        self,
        data_dir: typing.Union[typing.AnyStr, pathlib.Path],
        dataparser_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
        dataloader_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
        train_val_test_split: typing.Tuple[float, float, float] = (0.8, 0.1, 0.1),
        deeplake_kwargs: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
    ):
        """Initializes the AID data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.

        Args:
            data_dir: directory where the AID dataset is located.
            dataparser_configs: configuration dictionary of data parser settings, separated by
                data subset type, which may contain for example definitions for data transforms.
            dataloader_configs: configuration dictionary of data loader settings, separated by data
                subset type, which may contain for example batch sizes and worker counts.
            train_val_test_split: split proportions to use when separating the data.
            deeplake_kwargs: extra arguments forwarded to the deeplake dataset parser.
        """
        self.save_hyperparameters(logger=False)
        dataparser_configs = self._init_dataparser_configs(dataparser_configs)
        super().__init__(dataparser_configs=dataparser_configs, dataloader_configs=dataloader_configs)
        data_dir = pathlib.Path(data_dir)
        assert data_dir.is_dir(), f"invalid AID dataset directory: {data_dir}"
        if data_dir.name != ".deeplake":
            deeplake_subdir = data_dir / ".deeplake"
            all_class_subdirs = [data_dir / class_name for class_name in self.metadata.class_names]
            assert deeplake_subdir.is_dir() or all(
                [d.is_dir() for d in all_class_subdirs]
            ), "dataset directory should contain .deeplake folder or a folder for each data class"
        logger.debug(f"AID dataset root: {data_dir}")
        assert len(train_val_test_split) == 3 and sum(train_val_test_split) == 1.0
        self.data_train: typing.Optional[ssl4rs.data.parsers.aid.DeepLakeParser] = None
        self.data_valid: typing.Optional[ssl4rs.data.parsers.aid.DeepLakeParser] = None
        self.data_test: typing.Optional[ssl4rs.data.parsers.aid.DeepLakeParser] = None

    @staticmethod
    def _init_dataparser_configs(configs: ssl4rs.utils.DictConfig) -> omegaconf.DictConfig:
        """Updates the dataparser configs before they are passed to the base class w/ defaults."""
        # we'll add in the required defaults for the data parser configs based on our expected use
        if configs is None:
            configs = omegaconf.OmegaConf.create()
        elif isinstance(configs, dict):
            configs = omegaconf.OmegaConf.create(configs)
        base_dataparser_configs = {
            "_default_": {  # all data parsers will be based on the internal aid parser class
                "_target_": "ssl4rs.data.parsers.aid.DeepLakeParser",
            },
            # bonus: we will also give all parsers have a nice prefix
            "train": {"batch_id_prefix": "train"},
            "valid": {"batch_id_prefix": "valid"},
            "test": {"batch_id_prefix": "test"},
        }
        configs = omegaconf.OmegaConf.merge(
            omegaconf.OmegaConf.create(base_dataparser_configs),
            configs,
        )
        return configs

    @property
    def num_classes(self) -> int:
        """Returns the number of labels/classes in the dataset."""
        return len(self.metadata.class_names)

    def _is_preparation_complete(self) -> bool:
        """Returns whether the dataset is prepared for setup/loading or not."""
        root_data_dir = self.hparams.data_dir
        if root_data_dir.name == ".deeplake" and root_data_dir.is_dir():
            return True
        potential_deeplake_subdir = root_data_dir / ".deeplake"
        if potential_deeplake_subdir.is_dir():
            return True
        return False  # can't find the .deeplake subdir, we'll likely have to create it

    def prepare_data(self) -> None:
        """Will repackage the raw AID dataset into a deeplake format, if necessary.

        This method will NOT download the dataset; all the raw images need to be on disk already.
        """
        if self._is_preparation_complete():
            return
        # otherwise, let's do the repackaging now (`prepare_data` is called only once, ever)
        logger.info("Starting deep lake repackager for AID dataset...")
        root_data_dir = self.hparams.data_dir
        repackager = ssl4rs.data.repackagers.aid.DeepLakeRepackager(root_data_dir)
        deeplake_output_path = root_data_dir / ".deeplake"
        repackager.export(deeplake_output_path)
        assert pathlib.Path(deeplake_output_path).exists()

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Sets up AID data parsing across train/valid/test sets.

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
            root_data_dir = self.hparams.data_dir
            if root_data_dir.name != ".deeplake":
                root_data_dir = root_data_dir / ".deeplake"
            assert root_data_dir.is_dir()
            logger.debug(f"AID deeplake dataset root: {root_data_dir}")
            dataset = ssl4rs.data.parsers.aid.DeepLakeParser(root_data_dir, **deeplake_kwargs)
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
    datamodule = DataModule(data_dir=data_root_dir / "aid")
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
    logger.info("all done")


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config(config_name="data_profiler.yaml")
    _local_main(pathlib.Path(config_.utils.data_root_dir))
