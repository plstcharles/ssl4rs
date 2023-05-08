"""Implements a data module for the MNIST train/valid/test loaders.

See the following URL for more info on this dataset:
https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
"""

import pathlib
import typing

import hydra.utils
import omegaconf
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

    This Lightning interface allows you to share a full dataset without explaining how to download,
    split, transform, and process the data. More info here:
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: typing.Union[typing.AnyStr, pathlib.Path],
        dataparser_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
        dataloader_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
        train_val_split: typing.Tuple[int, int] = (55_000, 5_000),
    ):
        """Initializes the MNIST data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.

        Args:
            data_dir: directory where the MNIST dataset is located (or where it will be downloaded).
            dataparser_configs: configuration dictionary of data parser settings, separated by
                data subset type, which may contain for example definitions for data transforms.
            dataloader_configs: configuration dictionary of data loader settings, separated by data
                subset type, which may contain for example batch sizes and worker counts.
            train_val_split: sample split counts to use when separating the train/valid data.
        """
        self.save_hyperparameters(logger=False)
        dataparser_configs = self._init_dataparser_configs(dataparser_configs)
        super().__init__(dataparser_configs=dataparser_configs, dataloader_configs=dataloader_configs)
        assert data_dir is not None, "invalid data dir (must be specified, will download if needed)"
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        assert len(train_val_split) == 2 and sum(train_val_split) == 60_000
        self._internal_data_transforms = [  # we'll apply these to all tensors from the orig parser
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
        self.data_train: typing.Optional[torch.utils.data.Dataset] = None
        self.data_valid: typing.Optional[torch.utils.data.Dataset] = None
        self.data_test: typing.Optional[torch.utils.data.Dataset] = None

    @staticmethod
    def _init_dataparser_configs(configs: ssl4rs.utils.DictConfig) -> omegaconf.DictConfig:
        """Updates the dataparser configs before they are passed to the base class w/ defaults."""
        # we'll add in the required defaults for the data parser configs based on our expected use
        if configs is None:
            configs = omegaconf.OmegaConf.create()
        elif isinstance(configs, dict):
            configs = omegaconf.OmegaConf.create(configs)
        base_dataparser_configs = {
            "_default_": {  # all data parsers will wrap the torchvision mnist dataset parser
                "_target_": "ssl4rs.data.parsers.ParserWrapper",
                "batch_transforms": [  # we'll set up all the parsers with these two transforms
                    {
                        # this will map the loaded tuples with actual attribute names
                        "_target_": "ssl4rs.data.transforms.TupleMapper",
                        "key_map": {0: "data", 1: "target"},
                    },
                    {
                        # this will add a batch size to the batch based on the data tensor shape
                        "_target_": "ssl4rs.data.transforms.BatchSizer",
                        "batch_size_hint": "data",
                    },
                ],
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
        if self.data_train is None:
            orig_train_parser = torchvision.datasets.MNIST(
                root=self.hparams.data_dir,
                train=True,
                transform=torchvision.transforms.Compose(self._internal_data_transforms),
            )
            # NOTE: we regenerate a split here using the original mnist train set for train+valid
            train_parser, valid_parser = torch.utils.data.random_split(
                dataset=orig_train_parser,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(self.split_seed),
            )
            train_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "train")
            self.data_train = hydra.utils.instantiate(train_parser_config, train_parser)
            valid_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "valid")
            self.data_valid = hydra.utils.instantiate(valid_parser_config, valid_parser)
            orig_test_parser = torchvision.datasets.MNIST(
                root=self.hparams.data_dir,
                train=False,
                transform=torchvision.transforms.Compose(self._internal_data_transforms),
            )
            test_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "test")
            self.data_test = hydra.utils.instantiate(test_parser_config, orig_test_parser)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the MNIST training set data loader."""
        assert self.data_train is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_train, subset_type="train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the MNIST validation set data loader."""
        assert self.data_valid is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_valid, subset_type="valid")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the MNIST testing set data loader."""
        assert self.data_test is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_test, subset_type="test")
