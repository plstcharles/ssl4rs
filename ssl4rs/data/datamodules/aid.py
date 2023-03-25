"""Implements a data module for the AID train/valid/test loaders.

See the following URL for more info on this dataset: https://captain-whu.github.io/AID/
"""

import pathlib
import typing

import torch
import torch.utils.data

import ssl4rs.data.datamodules.utils
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

    class_distrib = ssl4rs.data.repackagers.aid.DeepLakeRepackager.class_distrib
    class_names = ssl4rs.data.repackagers.aid.DeepLakeRepackager.class_names
    image_shape = ssl4rs.data.repackagers.aid.DeepLakeRepackager.image_shape
    split_seed: int = 42  # should not really vary, ever...

    # noinspection PyUnusedLocal
    def __init__(
        self,
        data_dir: typing.Union[typing.AnyStr, pathlib.Path],
        dataloader_fn_map: typing.Optional[ssl4rs.data.datamodules.utils.DataLoaderFnMap] = None,
        train_val_test_split: typing.Tuple[float, float, float] = (0.8, 0.1, 0.1),
        deeplake_kwargs: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
    ):
        """Initializes the AID data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.

        Args:
            data_dir: directory where the AID dataset is located.
            dataloader_fn_map: dictionary of data loader creation settings. See the base class
                implementation for more information. When empty/null, the default data loader
                settings are assumed.
            train_val_test_split: split proportions to use when separating the data.
            deeplake_kwargs: extra arguments forwarded to the deeplake dataset parser.
        """
        super().__init__(dataloader_fn_map=dataloader_fn_map)
        data_dir = pathlib.Path(data_dir)
        assert data_dir.is_dir(), f"invalid AID dataset directory: {data_dir}"
        if data_dir.name != ".deeplake":
            deeplake_subdir = data_dir / ".deeplake"
            all_class_subdirs = [data_dir / class_name for class_name in self.class_names]
            assert deeplake_subdir.is_dir() or all(
                [d.is_dir() for d in all_class_subdirs]
            ), "dataset directory should contain .deeplake folder or a folder for each data class"
        logger.debug(f"AID dataset root: {data_dir}")
        assert len(train_val_test_split) == 3 and sum(train_val_test_split) == 1.0
        self.save_hyperparameters(logger=False)
        self.data_train: typing.Optional[ssl4rs.data.parsers.aid.DeepLakeParser] = None
        self.data_valid: typing.Optional[ssl4rs.data.parsers.aid.DeepLakeParser] = None
        self.data_test: typing.Optional[ssl4rs.data.parsers.aid.DeepLakeParser] = None

    @property
    def num_classes(self) -> int:
        """Returns the number of labels/classes in the dataset."""
        return len(self.class_names)

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
        """
        assert self._is_preparation_complete(), "dataset is not ready, call `prepare_data()` first!"
        # load datasets only if they're not loaded already (no matter the requested stage)
        if not self.data_train and not self.data_valid and not self.data_test:
            deeplake_kwargs = self.hparams.deeplake_kwargs or {}
            root_data_dir = self.hparams.data_dir
            if root_data_dir.name != ".deeplake":
                root_data_dir = root_data_dir / ".deeplake"
            assert root_data_dir.is_dir()
            logger.debug(f"AID deeplake dataset root: {root_data_dir}")
            dataset = ssl4rs.data.parsers.aid.DeepLakeParser(root_data_dir, **deeplake_kwargs)
            train_sample_count = int(round(self.hparams.train_val_test_split[0] * len(dataset)))
            valid_sample_count = int(round(self.hparams.train_val_test_split[1] * len(dataset)))
            test_sample_count = len(dataset) - (train_sample_count + valid_sample_count)
            # todo: update random split to use the `get_deeplake_parser_subset` function?
            self.data_train, self.data_valid, self.data_test = torch.utils.data.random_split(
                dataset=dataset,
                lengths=(train_sample_count, valid_sample_count, test_sample_count),
                generator=torch.Generator().manual_seed(self.split_seed),
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AID training set data loader."""
        assert self.data_train is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_train, loader_type="train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AID validation set data loader."""
        assert self.data_valid is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_valid, loader_type="valid")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AID testing set data loader."""
        assert self.data_test is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_test, loader_type="test")


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
