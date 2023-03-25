"""Implements a data module for the MNIST train/valid/test loaders.

See the following URL for more info on this dataset:
    https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
"""

import pathlib
import typing

import torch
import torch.utils.data
import torchvision

import ssl4rs.data.datamodules.utils
import ssl4rs.data.transforms


class DataModule(ssl4rs.data.datamodules.utils.DataModule):
    """Example of LightningDataModule for the MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This PyTorch Lightning interface allows you to share a full dataset without explaining how to
    download, split, transform, and process the data. More info here:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: typing.Union[typing.AnyStr, pathlib.Path],
        dataloader_fn_map: typing.Optional[ssl4rs.data.datamodules.utils.DataLoaderFnMap] = None,
        train_val_test_split: typing.Tuple[int, int, int] = (55_000, 5_000, 10_000),
    ):
        """Initializes the MNIST data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.

        Args:
            data_dir: directory where the MNIST dataset is located (or where it will be downloaded).
            dataloader_fn_map: dictionary of data loader creation settings. See the base class
                implementation for more information. When empty/null, the default data loader
                settings are assumed.
            train_val_test_split: sample split counts to use when separating the data.
        """
        super().__init__(dataloader_fn_map=dataloader_fn_map)
        assert data_dir is not None
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        assert len(train_val_test_split) == 3 and sum(train_val_test_split) == 70_000
        self.save_hyperparameters(logger=False)
        self.data_transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
        self.batch_transforms = [
            ssl4rs.data.transforms.TupleMapper({0: "data", 1: "target"}),
            ssl4rs.data.transforms.BatchSizer("data"),
        ]
        self.data_train: typing.Optional[torch.utils.data.Dataset] = None
        self.data_valid: typing.Optional[torch.utils.data.Dataset] = None
        self.data_test: typing.Optional[torch.utils.data.Dataset] = None

    @property
    def num_classes(self) -> int:
        """Returns the number of labels/classes in the dataset."""
        return 10

    def prepare_data(self) -> None:
        """Downloads the MNIST data to the dataset directory, if needed.

        This method is called only from a single device, so the data will only be downloaded once.
        """
        torchvision.datasets.MNIST(self.hparams.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Loads the MNIST data under the train/valid/test parsers.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`, so be
        careful not to execute the random split twice! The `stage` can be used to differentiate
        whether it's called before `trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_valid and not self.data_test:
            trainset = ssl4rs.data.parsers.ParserWrapper(
                dataset=torchvision.datasets.MNIST(
                    root=self.hparams.data_dir,
                    train=True,
                    transform=torchvision.transforms.Compose(self.data_transforms),
                ),
                batch_transforms=self.batch_transforms,
                batch_id_prefix="train",
            )
            testset = ssl4rs.data.parsers.ParserWrapper(
                dataset=torchvision.datasets.MNIST(
                    root=self.hparams.data_dir,
                    train=False,
                    transform=torchvision.transforms.Compose(self.data_transforms),
                ),
                batch_transforms=self.batch_transforms,
                batch_id_prefix="test",
            )
            dataset = trainset + testset
            self.data_train, self.data_valid, self.data_test = torch.utils.data.random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the MNIST training set data loader."""
        assert self.data_train is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_train, loader_type="train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the MNIST validation set data loader."""
        assert self.data_valid is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_valid, loader_type="valid")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the MNIST testing set data loader."""
        assert self.data_test is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_test, loader_type="test")
