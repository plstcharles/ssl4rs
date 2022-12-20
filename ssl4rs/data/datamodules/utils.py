"""Contains utility functions and a base interface for pytorch-lightning datamodules."""
import os
import typing

import hydra.utils
import numpy as np
import omegaconf
import pytorch_lightning.utilities.types
import torch
import torch.utils.data

import ssl4rs.utils.logging

if typing.TYPE_CHECKING:
    import ssl4rs

DataLoaderFnMap = typing.Mapping[typing.AnyStr, typing.Optional[omegaconf.DictConfig]]

logger = ssl4rs.utils.logging.get_logger(__name__)


class DataModule(pytorch_lightning.LightningDataModule):
    """Wraps the standard LightningDataModule interface to combine it with Hydra.

    Each derived data module will likely correspond to the combination of one data source and one
    target task. This interface provides common definitions regarding dataloader creation, and helps
    document what functions should have an override in the derived classes and why.

    For more information, see:
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    default_dataloader_types = tuple(["train", "valid", "test", "predict"])
    """Types of dataloaders that the base class supports; derived impls can support more/fewer."""

    def __init__(
        self,
        dataloader_fn_map: DataLoaderFnMap,
    ):
        """Initializes the base class interface using the map of loader-type-to-function pairs.

        If any of the functions for the specified/supported loader types is null or missing, it
        will be set as a basic default.
        """
        super().__init__()
        logger.debug("Instantiating LightningDataModule base class...")
        assert all(
            [isinstance(k, str) for k in dataloader_fn_map.keys()]
        ), "dataloader function map should have string keys that correspond to loader/loop types"
        assert all(
            [isinstance(v, (dict, omegaconf.DictConfig)) for v in dataloader_fn_map.values()]
        ), "dataloader function map values should be subconfigs (no partial functions or objs yet!)"
        self.dataloader_fn_map = {
            loader_type: dataloader_fn_map.get(loader_type, None)
            for loader_type in set(list(dataloader_fn_map.keys()) + list(self.default_dataloader_types))
        }

    def prepare_data(self) -> None:
        """Use this to download and prepare data.

        Downloading and saving data with multiple processes (distributed settings) will result in
        corrupted data. PyTorch-Lightning ensures this method is called only within a single process,
        so you can safely add your downloading logic within.

        For more information, see:
            https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#prepare-data
        """
        pass

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Called at the beginning of `fit` (training + validation), `validate`, `test`, or
        `predict`.

        This is where the metadata, size, and other high-level info of the already-downloaded/prepared
        dataset should be parsed. The outcome of this parsing should be a "state" inside the data
        module itself, likely in a data parser (e.g. derived from `torch.utils.data.Dataset`).

        For more information, see:
            https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#setup

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass

    def teardown(self, stage: typing.Optional[str] = None) -> None:
        """Called at the end of `fit` (training + validation), `validate`, `test`, or `predict`.

        When called, the "state" of the downloaded/prepared dataset parsed in `setup` should be
        cleared (if needed).

        For more information, see:
            https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#teardown

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass

    def train_dataloader(self) -> pytorch_lightning.utilities.types.TRAIN_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for training based on the parsed dataset.

        For more information, see:
            https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#train-dataloader

        Returns:
            A data loader (or a collection of them) that provides training samples.
        """
        raise NotImplementedError

    def test_dataloader(self) -> pytorch_lightning.utilities.types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for testing based on the parsed dataset.

        Note:
            In the case where this returns multiple test dataloaders, the LightningModule `test_step`
            method will have an argument `dataloader_idx` which matches the order here.

        For more information, see:
            https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#test-dataloader

        Returns:
            A data loader (or a collection of them) that provides testing samples.
        """
        raise NotImplementedError

    def val_dataloader(self) -> pytorch_lightning.utilities.types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for validation based on the parsed dataset.

        Note:
            During training, the returned dataloader(s) will not be reloaded between epochs unless
            you set the `reload_dataloaders_every_n_epochs` argument (in the trainer configuration)
            to a positive integer.

            In the case where this returns multiple dataloaders, the LightningModule `validation_step`
            method will have an argument `dataloader_idx` which matches the order here.

        For more information, see:
            https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#val-dataloader

        Returns:
            A data loader (or a collection of them) that provides validation samples.
        """
        raise NotImplementedError

    def valid_dataloader(self) -> pytorch_lightning.utilities.types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for validation based on the parsed dataset.

        This function simply redirects to the `val_dataloader` function. Why? Just because using
        'val' instead of 'valid' is not everyone's cup of tea.
        """
        return self.val_dataloader()

    def predict_dataloader(self) -> pytorch_lightning.utilities.types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for prediction runs.

        Note:
            In the case where this returns multiple dataloaders, the LightningModule `predict_step`
            method will have an argument `dataloader_idx` which matches the order here.

        Return:
            A data loader (or a collection of them) that provides prediction samples.
        """
        # note: most datasets do not offer a 'predict' dataloader; we'll make one w/ valid+test
        return [self.val_dataloader(), self.test_dataloader()]

    @property
    def dataloader_types(self) -> typing.List[typing.AnyStr]:
        """Types of dataloaders that this particular implementation supports."""
        return [t for t in self.dataloader_fn_map.keys() if not t.startswith("_")]

    def get_dataloader(
        self,
        loader_type: typing.AnyStr,
    ) -> typing.Union[
        pytorch_lightning.utilities.types.TRAIN_DATALOADERS,
        pytorch_lightning.utilities.types.EVAL_DATALOADERS,
    ]:
        """Returns a data loader object (or a collection of) for a given loader type.

        This function will verify that the specified loader type exists and is supported, and
        redirect the getter to the correct function that prepares the dataloader(s).
        """
        assert loader_type in self.dataloader_types, f"invalid loader type: {loader_type}"
        expected_getter_name = f"{loader_type}_dataloader"
        assert hasattr(self, expected_getter_name), f"invalid getter attrib: {expected_getter_name}"
        getter = getattr(self, expected_getter_name)
        assert callable(getter), f"invalid getter type: {type(getter)}"
        dataloader = getter()
        assert isinstance(
            dataloader,
            typing.Union[
                pytorch_lightning.utilities.types.TRAIN_DATALOADERS,
                pytorch_lightning.utilities.types.EVAL_DATALOADERS,
            ],
        ), f"invalid dataloader type: {type(dataloader)}"
        return dataloader

    def _create_dataloader(
        self,
        parser: torch.utils.data.Dataset,
        loader_type: typing.AnyStr,
    ) -> torch.utils.data.DataLoader:
        """Returns a data loader object for a given parser and loader type.

        This wrapper allows us to redirect the data loader creation function to a fully-initialized
        partial function config that is likely provided via Hydra.

        The provided parser is forwarded directly to the dataloader creation function.
        """
        logger.debug(f"Instantiating a new '{loader_type}' dataloader...")
        default_settings = self.dataloader_fn_map.get("_default_", omegaconf.OmegaConf.create())
        target_settings = self.dataloader_fn_map.get(loader_type, omegaconf.OmegaConf.create())
        if target_settings is None:  # in case it was specified as an empty group, same as default
            target_settings = omegaconf.OmegaConf.create()
        combined_settings = omegaconf.OmegaConf.merge(default_settings, target_settings)
        if os.getenv("PL_SEED_WORKERS"):
            if combined_settings.get("worker_init_fn", None) is not None:
                logger.warning(
                    "Using a custom worker init function with `seed_workers=True`! "
                    "(cannot use the lightning seed function here, make sure you use/call it yourself!)"
                )
            else:
                combined_settings["worker_init_fn"] = omegaconf.OmegaConf.create(
                    {
                        "_partial_": True,
                        "_target_": "pytorch_lightning.utilities.seed.pl_worker_init_function",
                    }
                )
        assert "_target_" in combined_settings, f"bad dataloader config for type: {loader_type}"
        assert not combined_settings.get(
            "_partial_", False
        ), "this function should not return a partial function, it's time to create the loader!"
        dataloader = hydra.utils.instantiate(combined_settings, parser)
        assert isinstance(dataloader, torch.utils.data.DataLoader), "invalid dataloader type!"
        return dataloader


def default_collate(
    batches: typing.List["ssl4rs.data.BatchDictType"],
    keys_to_batch_manually: typing.Sequence[typing.AnyStr] = (),
) -> "ssl4rs.data.transforms.BatchDictType":
    """Performs the default collate function while manually handling some given special cases."""
    assert isinstance(batches, (list, tuple)) and all(
        [isinstance(b, dict) for b in batches]
    ), f"unexpected type for batch array provided to collate: {type(batches)}"
    assert all(
        [len(np.setxor1d(list(batches[idx].keys()), list(batches[0].keys()))) == 0 for idx in range(1, len(batches))]
    ), "not all batches have the same sets of keys! (implement your own custom collate fn!)"
    avail_batch_keys = list(batches[0].keys())
    output = dict()
    # first step: look for the keys that we need to batch manually, and handle those
    default_keys_to_batch_manually = [
        "batch_id",  # should correspond to hashable objects that might hurt torch's default_collate
    ]
    keys_to_batch_manually = set(*keys_to_batch_manually, *default_keys_to_batch_manually)
    for key in keys_to_batch_manually:
        if key in avail_batch_keys:
            output[key] = [b[key] for b in batches]
    output.update(
        torch.utils.data.default_collate(
            [{k: v for k, v in b.items() if k not in keys_to_batch_manually} for b in batches]
        )
    )
    return output
