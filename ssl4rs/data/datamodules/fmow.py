"""Implements a data module for the Functional Map of the World (fMoW) dataset.

See the following URLs for more info on this dataset:
https://arxiv.org/abs/1711.07846
https://github.com/fMoW/dataset
https://spacenet.ai/iarpa-functional-map-of-the-world-fmow/
"""

import pathlib
import typing

import torch
import torch.utils.data

import ssl4rs.data.datamodules.utils
import ssl4rs.data.metadata.fmow
import ssl4rs.data.parsers.fmow
import ssl4rs.data.repackagers.fmow
import ssl4rs.utils.config
import ssl4rs.utils.logging

logger = ssl4rs.utils.logging.get_logger(__name__)


class DataModule(ssl4rs.data.datamodules.utils.DataModule):
    """Implementation derived from the standard LightningDataModule for the fMoW dataset.

    This dataset contains large-scale satellite imagery that can be used for object detection and
    classification. Version 1.2.1 of the dataset contains 523,846 images of 119421 unique objects,
    and each object instance has between 1 and 41 images. The images may be multispectral or RGB,
    and may have different Ground Sampling Distances (GSDs). The shape of these images ranges from
    a few hundred pixels to over 16,000 pixels in height/width.

    This dataset CANNOT be downloaded on the spot by this data module, meaning we assume it is on
    disk at runtime. Since the repackaging of the raw data into a deeplake format also takes a
    significant amount of time (probably 6-12 hours), this data module will not perform any
    automatic repackaging.
    """

    metadata = ssl4rs.data.metadata.fmow
    split_seed: int = 42  # should not really vary, ever...

    # noinspection PyUnusedLocal
    def __init__(
        self,
        data_dir: typing.Union[typing.AnyStr, pathlib.Path],
        dataloader_fn_map: typing.Optional[ssl4rs.data.datamodules.utils.DataLoaderFnMap] = None,
        deeplake_kwargs: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
    ):
        """Initializes the fMoW data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.

        Args:
            data_dir: directory where the fMoW dataset is located.
            dataloader_fn_map: dictionary of data loader creation settings. See the base class
                implementation for more information. When empty/null, the default data loader
                settings are assumed.
            deeplake_kwargs: extra arguments forwarded to the deeplake dataset parser.
        """
        self.save_hyperparameters(logger=False)
        super().__init__(dataloader_fn_map=dataloader_fn_map)
        data_dir = pathlib.Path(data_dir)
        assert data_dir.is_dir(), f"invalid fMoW dataset directory: {data_dir}"
        if data_dir.name != ".deeplake":
            data_dir = data_dir / ".deeplake"
        assert data_dir.is_dir(), f"dataset directory should contain .deeplake folder at: {data_dir}"
        logger.debug(f"fMoW dataset root: {data_dir}")
        self.data_parsers: typing.Dict[str, ssl4rs.data.parsers.fmow.DeepLakeParser] = {}

    @property
    def num_classes(self) -> int:
        """Returns the number of labels/classes in the dataset."""
        return len(self.metadata.class_names)

    def prepare_data(self) -> None:
        """Does nothing; all data preparation must be done in advance with fMoW since it is huge.

        See the `ssl4rs/data/repackagers/fmow.py` file for more information.
        """
        assert pathlib.Path(self.hparams.data_dir).is_dir()

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Sets up fMoW data parsing across the global ('all') set."""
        if "all" not in self.data_parsers:
            deeplake_kwargs = self.hparams.deeplake_kwargs or {}
            root_data_dir = pathlib.Path(self.hparams.data_dir)
            if root_data_dir.name != ".deeplake":
                root_data_dir = root_data_dir / ".deeplake"
            assert root_data_dir.is_dir()
            logger.debug(f"fMoW deeplake dataset root: {root_data_dir}")
            dataset = ssl4rs.data.parsers.fmow.DeepLakeParser(root_data_dir, **deeplake_kwargs)
            logger.debug(f"ready to parse {len(dataset)} fMoW instances")
            self.data_parsers["all"] = dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the fMoW training set data loader."""
        if "train" not in self.data_parsers:
            assert "all" in self.data_parsers, "global parser unavailable, call `setup()` first!"
            train_data_parser = self.data_parsers["all"].get_train_subset()
            # @@@@@ todo: optimize (rechunk) for better speed?
            self.data_parsers["train"] = train_data_parser
        return self._create_dataloader(self.data_parsers["train"], loader_type="train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the fMoW validation set data loader."""
        if "val" not in self.data_parsers:
            assert "all" in self.data_parsers, "global parser unavailable, call `setup()` first!"
            val_data_parser = self.data_parsers["all"].get_val_subset()
            self.data_parsers["val"] = val_data_parser
        return self._create_dataloader(self.data_parsers["val"], loader_type="val")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the fMoW testing set data loader."""
        if "test" not in self.data_parsers:
            assert "all" in self.data_parsers, "global parser unavailable, call `setup()` first!"
            test_data_parser = self.data_parsers["all"].get_test_subset()
            self.data_parsers["test"] = test_data_parser
        return self._create_dataloader(self.data_parsers["test"], loader_type="test")

    def seq_dataloader(self) -> torch.utils.data.DataLoader:
        if "seq" not in self.data_parsers:
            assert "all" in self.data_parsers, "global parser unavailable, call `setup()` first!"
            seq_data_parser = self.data_parsers["all"].get_seq_subset()
            self.data_parsers["seq"] = seq_data_parser
        return self._create_dataloader(self.data_parsers["seq"], loader_type="seq")

    def global_dataloader(self) -> torch.utils.data.DataLoader:
        assert "all" in self.data_parsers, "global parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_parsers["all"], loader_type="all")


def _local_main(data_root_dir: pathlib.Path) -> None:
    datamodule = DataModule(data_dir=data_root_dir / "fmow-rgb")
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
    global_dataloader = datamodule.global_dataloader()
    minibatch = next(iter(global_dataloader))
    assert isinstance(minibatch, dict)
    logger.info("all done")


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config(config_name="data_profiler.yaml")
    _local_main(pathlib.Path(config_.utils.data_root_dir) / "fmow")
