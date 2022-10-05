"""Implements a data module for the AID train/valid/test loaders.

See the following URL for more info on this dataset:
    https://captain-whu.github.io/AID/
"""

import pathlib
import typing

import torch
import torch.utils.data

import ssl4rs.data.datamodules.utils
import ssl4rs.data.parsers.aid
import ssl4rs.data.repackagers.aid


class DataModule(ssl4rs.data.datamodules.utils.DataModule):
    """Implementation derived from the standard LightningDataModule for the AID dataset.

    This dataset contains large-scale aerial images that can be used for classification. There are
    10,000 images (600x600, RGB) in this dataset, and these are given one of 30 class labels.
    See https://captain-whu.github.io/AID/ for more information and download links. This dataset
    CANNOT be downloaded on the spot by this data module, meaning we assume it is on disk at runtime.
    """

    class_distrib = ssl4rs.data.repackagers.aid.AIDRepackager.class_distrib
    class_names = ssl4rs.data.repackagers.aid.AIDRepackager.class_names
    image_shape = ssl4rs.data.repackagers.aid.AIDRepackager.image_shape
    split_seed: int = 42  # should not really vary, ever...

    # noinspection PyUnusedLocal
    def __init__(
        self,
        data_dir: typing.Union[typing.AnyStr, pathlib.Path],
        dataloader_fn_map: ssl4rs.data.datamodules.utils.DataLoaderFnMap,
        train_val_test_split: typing.Tuple[float, float, float] = (0.8, 0.1, 0.1),
        deeplake_kwargs: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
    ):
        """Initializes the AID data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.
        """
        super().__init__(dataloader_fn_map=dataloader_fn_map)
        assert data_dir is not None and pathlib.Path(data_dir).is_dir(), f"invalid dir: {data_dir}"
        assert len(train_val_test_split) == 3 and sum(train_val_test_split) == 1.0
        self.save_hyperparameters(logger=False)
        self.data_train: typing.Optional[ssl4rs.data.parsers.aid.DeepLakeParser] = None
        self.data_valid: typing.Optional[ssl4rs.data.parsers.aid.DeepLakeParser] = None
        self.data_test: typing.Optional[ssl4rs.data.parsers.aid.DeepLakeParser] = None

    @property
    def num_classes(self) -> int:
        """Returns the number of labels/classes in the dataset."""
        return len(self.class_names)

    def prepare_data(self) -> None:
        """Does nothing, as the dataset cannot be downloaded automatically here."""
        pass

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Loads the AID data under the train/valid/test parsers.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before `trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already (no matter the requested stage)
        if not self.data_train and not self.data_valid and not self.data_test:
            deeplake_kwargs = self.hparams.deeplake_kwargs or {}
            dataset = ssl4rs.data.parsers.aid.DeepLakeParser(
                self.hparams.data_dir,
                **deeplake_kwargs,
            )
            train_sample_count = int(round(self.hparams.train_val_test_split[0] * len(dataset)))
            valid_sample_count = int(round(self.hparams.train_val_test_split[1] * len(dataset)))
            test_sample_count = len(dataset) - (train_sample_count + valid_sample_count)
            self.data_train, self.data_valid, self.data_test = torch.utils.data.random_split(
                dataset=dataset,
                lengths=(train_sample_count, valid_sample_count, test_sample_count),
                generator=torch.Generator().manual_seed(self.split_seed),
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AID training set data loader."""
        assert self.data_train is not None, "parser unavailable, call 'setup()' first!"
        return self._create_dataloader(self.data_train, loader_type="train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AID validation set data loader."""
        assert self.data_valid is not None, "parser unavailable, call 'setup()' first!"
        return self._create_dataloader(self.data_valid, loader_type="valid")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the AID testing set data loader."""
        assert self.data_test is not None, "parser unavailable, call 'setup()' first!"
        return self._create_dataloader(self.data_test, loader_type="test")


def _local_main(data_root_dir: pathlib.Path) -> None:
    import ssl4rs.utils.config
    config = ssl4rs.utils.config.init_hydra_and_compose_config()
    datamodule = DataModule(
        data_dir=data_root_dir / "aid/aid.deeplake",
        dataloader_fn_map=config.data.dataloader_fn_map,
    )
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
    _local_main(ssl4rs.utils.config.get_data_root_dir())
