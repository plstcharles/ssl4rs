"""Contains utility functions and a base interface for lightning datamodules."""
import copy
import os
import typing

import hydra.utils
import lightning.pytorch as pl
import lightning.pytorch.utilities.types as pl_types
import numpy as np
import omegaconf
import torch
import torch.utils.data

import ssl4rs.utils.logging

if typing.TYPE_CHECKING:
    import ssl4rs

logger = ssl4rs.utils.logging.get_logger(__name__)

use_deeplake_loaders_key = "deeplake.integrations.pytorch.pytorch.dataset_to_pytorch"
"""Key to toggle on the use of the deeplake data loaders."""


class DataModule(pl.LightningDataModule):
    """Wraps the standard LightningDataModule interface to combine it with Hydra.

    Each derived data module will likely correspond to the combination of one dataset and one
    target task. This interface provides common definitions regarding data parser and loader
    creation, and helps document what functions should have an override in the derived classes
    and why. The reason to use it is to simplify the creation of data parsers and loaders with
    shared (and externally configurable) settings (e.g. data transformations).

    For more information on generic data modules, see:
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    _default_subset_types = tuple(["train", "valid", "test"])
    """Types of data subsets that the module supports; derived impls can support more/fewer."""

    split_seed: int = 42
    """Static seed used to initialize internal RNGs for dataset splits; should never change!"""

    def __init__(
        self,
        dataparser_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
        dataloader_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
    ):
        """Initializes the base class interface using the expected configs of parsers/loaders.

        Args:
            dataparser_configs: configuration dictionary of data parser settings, separated by
                data subset type, which may contain for example definitions for data transforms.
                These settings are not used directly in this base class interface, and are instead
                expected to be used by the derived DataModule classes.
            dataloader_configs: configuration dictionary of data loader settings, separated by data
                subset type, which may contain for example batch sizes and worker counts. These
                settings will be used directly in this base class interface via the
                `_create_dataloader` function.

        The provided configuration dictionaries may be empty if default data parsing/loading
        settings are fine. When settings are provided, each data subset (e.g. `train`, or `valid`)
        should have its own section in the configuration. In other words, the given configuration
        dictionary should contain subdictionaries, with one for each data subset. If a subset
        is left unspecified, default settings will be used. To override default settings across
        all data subsets, define new settings in a special `_default_` section. For example, the
        data loader config can be provided as:

            _default_:
                _target_: torch.utils.data.DataLoader
                batch_size: 64
            train:
                batch_size: 32
                shuffle: True

        In that case, all data loaders will be instantiated from the `torch.utils.data.DataLoader`
        target class, and they will all have a batch size of 64 by default. The training data
        loader will have a batch size of 32, and it will shuffle its data. For the creation of
        DeepLake PyTorch DataLoaders, see the implementation of `_create_dataloader` for more info.

        If any of the settings for the specified/supported subsets is null or missing, it
        will be set as the default from the _default_ section if a match exists, otherwise it will
        use the default value from the default data loader/parser constructor. For data loaders,
        a default target class is always assumed (`torch.utils.data.DataLoader`). For more info on
        the default values for that class, see:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        """
        super().__init__()
        assert len(self.subset_types) > 0, f"invalid data subset types: {self.subset_types}"
        self.dataparser_configs = self._init_configs(dataparser_configs, self._base_dataparser_configs)
        self.dataloader_configs = self._init_configs(dataloader_configs, self._base_dataloader_configs)

    @property
    def _base_dataparser_configs(self) -> ssl4rs.utils.DictConfig:
        """Returns the 'base' (class-specific-default) configs for the data parsers."""
        # we do not have anything to provide here (override if needed!)
        return omegaconf.OmegaConf.create()

    @property
    def _base_dataloader_configs(self) -> ssl4rs.utils.DictConfig:
        """Returns the 'base' (class-specific-default) configs for the data loaders."""
        # we do not have anything to provide here (override if needed!)
        return omegaconf.OmegaConf.create()

    def _init_configs(
        self,
        user_configs: typing.Optional[ssl4rs.utils.DictConfig],
        base_configs: typing.Optional[ssl4rs.utils.DictConfig] = None,
    ) -> omegaconf.DictConfig:
        """Initializes (merges) the given user + base configs with extra defaults and/or subsets.

        Derived classes may override the `_base_dataparser_configs` or `_base_dataloader_configs`
        properties or this function to add extra defaults or subsets specific to each datamodule.
        """
        if user_configs is None:
            user_configs = omegaconf.OmegaConf.create()
        elif isinstance(user_configs, dict):
            user_configs = omegaconf.OmegaConf.create(user_configs)
        if base_configs is None:
            base_configs = omegaconf.OmegaConf.create()
        elif isinstance(user_configs, dict):
            base_configs = omegaconf.OmegaConf.create(base_configs)
        subsets = set(list(user_configs.keys()) + list(base_configs.keys()) + list(self.subset_types))
        output_configs = {}
        for subset in subsets:
            # instead of a global merge, we update subset-level settings key by key to replace groups
            output_config = copy.deepcopy(base_configs.get(subset, omegaconf.OmegaConf.create()))
            if not isinstance(output_config, omegaconf.DictConfig):
                output_config = omegaconf.OmegaConf.create(output_config)
            user_config = user_configs.get(subset, omegaconf.OmegaConf.create())
            for setting_name in user_config.keys():
                omegaconf.OmegaConf.update(
                    cfg=output_config,
                    key=setting_name,
                    value=user_config.get(setting_name),
                    merge=False,
                )
            output_configs[subset] = output_config
        output_configs = omegaconf.OmegaConf.create(output_configs)
        return output_configs

    @property
    def subset_types(self) -> typing.Sequence[str]:
        """Returns the data subsets supported by the datamodule.

        By default, if no hint is available, we will use the list of commonly used subsets:
        "train", "valid", and "test". If metadata is available, we will look for a `subset_types`
        attribute, make sure it's a list, and return it.
        """
        if hasattr(self, "_subset_types"):
            output = self._subset_types
        elif hasattr(self, "metadata"):
            if isinstance(self.metadata, dict) and "subset_types" in self.metadata:
                output = self.metadata["subset_types"]
            elif hasattr(self.metadata, "subset_types"):
                output = self.metadata.subset_types  # noqa
            else:
                output = self._default_subset_types
        else:
            output = self._default_subset_types
        assert all([isinstance(s, str) for s in output])
        subset_types = [s for s in output]
        return subset_types

    def prepare_data(self) -> None:
        """Override this function to download and prepare data for the dataloaders.

        Downloading and saving data with multiple processes (distributed settings) will result in
        corrupted data. Lightning ensures this method is called only within a single process, so
        you can safely add your downloading logic within.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Called at the beginning of `fit` (train + validation), `validate`, `test`, or `predict`.

        This is where the metadata, size, and other high-level info of the already-downloaded
        and prepared dataset should be parsed. The outcome of this parsing should be a "state"
        inside the data module itself, likely in a data parser (e.g. an instance derived from
        `torch.utils.data.Dataset`). With a distributed training strategy, this will be called on
        each node.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass

    def teardown(self, stage: typing.Optional[str] = None) -> None:
        """Called at the end of `fit` (training + validation), `validate`, `test`, or `predict`.

        When called, the "state" of the downloaded/prepared dataset parsed in `setup` should be
        cleared (if needed).

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#teardown

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass

    def train_dataloader(self) -> pl_types.TRAIN_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for training based on the parsed dataset.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader

        Returns:
            A data loader (or a collection of them) that provides training samples.
        """
        raise NotImplementedError

    def test_dataloader(self) -> pl_types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for testing based on the parsed dataset.

        Note:
            In the case where this returns multiple test dataloaders, the LightningModule `test_step`
            method will have an argument `dataloader_idx` which matches the order here.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test-dataloader

        Returns:
            A data loader (or a collection of them) that provides testing samples.
        """
        raise NotImplementedError

    def val_dataloader(self) -> pl_types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for validation based on the parsed dataset.

        Note:
            During training, the returned dataloader(s) will not be reloaded between epochs unless
            you set the `reload_dataloaders_every_n_epochs` argument (in the trainer configuration)
            to a positive integer.

            In the case where this returns multiple dataloaders, the LightningModule `validation_step`
            method will have an argument `dataloader_idx` which matches the order here.

        For more information, see:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader

        Returns:
            A data loader (or a collection of them) that provides validation samples.
        """
        raise NotImplementedError

    def valid_dataloader(self) -> pl_types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for validation based on the parsed dataset.

        This function simply redirects to the `val_dataloader` function. Why? Just because using
        'val' instead of 'valid' is not everyone's cup of tea.
        """
        return self.val_dataloader()

    def predict_dataloader(self) -> pl_types.EVAL_DATALOADERS:
        """Instantiates one or more pytorch dataloaders for prediction runs.

        Note: in contrast with typical validation or test data loaders, this loader is typically
        assumed NOT to load groundtruth (target) annotations along with data samples.

        Note:
            In the case where this returns multiple dataloaders, the LightningModule `predict_step`
            method will have an argument `dataloader_idx` which matches the order here.

        Return:
            A data loader (or a collection of them) that provides prediction samples.
        """
        raise NotImplementedError

    @property
    def dataloader_types(self) -> typing.List[typing.AnyStr]:
        """Types of dataloaders that this particular implementation supports.

        Note: by default, we assume that these 'types' are linked with the data subsets.
        """
        return [t for t in self.dataloader_configs.keys() if not t.startswith("_")]

    def get_dataloader(
        self,
        loader_type: typing.AnyStr,
    ) -> typing.Union[pl_types.TRAIN_DATALOADERS, pl_types.EVAL_DATALOADERS]:
        """Returns a data loader object (or a collection of) for a given loader type.

        This function will verify that the specified loader type exists and is supported, and
        redirect the getter to the correct function that prepares the dataloader(s). By default, we
        assume that dataloader types are linked to data subsets.
        """
        assert loader_type in self.dataloader_types, f"invalid loader type: {loader_type}"
        expected_getter_name = f"{loader_type}_dataloader"
        assert hasattr(self, expected_getter_name), f"invalid getter attrib: {expected_getter_name}"
        getter = getattr(self, expected_getter_name)
        assert callable(getter), f"invalid getter type: {type(getter)}"
        dataloader = getter()
        return dataloader

    @classmethod
    def _get_subset_idxs_with_stratified_sampling(
        cls,
        labels: np.ndarray,  # should be 1d label index array (0-based)
        class_idx_to_count_map: typing.Dict[int, int],
        split_ratios: typing.Tuple[float, ...],
    ) -> typing.Tuple[typing.List[int], ...]:
        """Returns the sample idxs to use in train/valid/test sets with stratified sampling."""
        assert sum(class_idx_to_count_map.values()) == len(labels)
        rng = torch.Generator()
        rng.manual_seed(cls.split_seed)
        output_sample_idxs = [[] for _ in range(len(split_ratios))]
        for class_idx, sample_count in class_idx_to_count_map.items():
            split_sample_idxs = torch.utils.data.random_split(
                np.nonzero(labels == class_idx)[0],
                lengths=split_ratios,
                generator=rng,
            )
            for subset_idx, sample_idxs in enumerate(split_sample_idxs):
                output_sample_idxs[subset_idx].extend([idx for idx in sample_idxs])
        assert sum(len(s) for s in output_sample_idxs) == len(labels)
        assert len(np.unique(np.concatenate(output_sample_idxs))) == len(labels)
        output_sample_idxs = [np.sort(s).tolist() for s in output_sample_idxs]
        return tuple(output_sample_idxs)  # noqa

    @staticmethod
    def _get_subconfig_for_subset(
        configs: omegaconf.DictConfig,
        subset_type: str,
    ) -> omegaconf.DictConfig:
        """Returns an omegaconf DictConfig object for the given subset type.

        This function will use defaults if available, and then override with specific values.
        """
        default_subset_settings = configs.get("_default_", omegaconf.OmegaConf.create())
        if default_subset_settings is None:  # in case it was specified as an empty group, same as default
            default_subset_settings = omegaconf.OmegaConf.create()
        target_subset_settings = configs.get(subset_type, omegaconf.OmegaConf.create())
        if target_subset_settings is None:  # in case it was specified as an empty group, same as default
            target_subset_settings = omegaconf.OmegaConf.create()
        # instead of a global merge, we update subset-level settings key by key to replace groups
        output_settings = copy.deepcopy(default_subset_settings)
        for setting_name in target_subset_settings:
            omegaconf.OmegaConf.update(
                cfg=output_settings,
                key=setting_name,
                value=target_subset_settings.get(setting_name),
                merge=False,
            )
        return output_settings

    def _create_dataloader(
        self,
        parser: torch.utils.data.Dataset,
        subset_type: typing.AnyStr,
    ) -> torch.utils.data.DataLoader:
        """Creates and returns a new data loader object for a particular data subset.

        This wrapper allows us to redirect the data loader creation function to a fully-initialized
        partial function config that is likely provided via Hydra.

        The provided parser is forwarded directly to the dataloader creation function.
        """
        logger.debug(f"Instantiating a new '{subset_type}' dataloader...")
        config = self._get_subconfig_for_subset(self.dataloader_configs, subset_type)
        if os.getenv("PL_SEED_WORKERS"):
            if config.get("worker_init_fn", None) is not None:
                logger.warning(
                    "Using a custom worker init function with `seed_workers=True`! "
                    "(cannot use the lightning seed function here, make sure you use/call it yourself!)"
                )
            else:
                with omegaconf.open_dict(config):
                    # open_dict allows us to write through hydra's omegaconf struct
                    config.worker_init_fn = omegaconf.OmegaConf.create(
                        {
                            "_partial_": True,
                            "_target_": "lightning_fabric.utilities.seed.pl_worker_init_function",
                        }
                    )
        if "_target_" not in config or config["_target_"] is None:
            # if the type of dataloader is not specified, use PyTorch's DataLoader class
            with omegaconf.open_dict(config):
                # open_dict allows us to write through hydra's omegaconf struct
                config._target_ = "torch.utils.data.DataLoader"
        assert not config.get(
            "_partial_", False
        ), "this function should not return a partial function, it's time to create the loader!"
        if config["_target_"] == use_deeplake_loaders_key:
            import deeplake

            import ssl4rs.data.parsers.utils.deeplake

            assert isinstance(
                parser, ssl4rs.data.parsers.utils.deeplake.DeepLakeParser
            ), f"need a deeplake data parser, but got: {type(parser)}"
            assert isinstance(
                parser.dataset, deeplake.Dataset
            ), f"need a deeplake dataset attrib in parser, but got: {type(parser.dataset)}"
            dataloader = hydra.utils.instantiate(config, parser)  # UPDATE ME W PARSER's DATALOADER GETTER
        else:
            dataloader = hydra.utils.instantiate(config, parser)
        assert isinstance(
            dataloader, torch.utils.data.DataLoader
        ), f"invalid dataloader type: {type(dataloader)} (...should be DataLoader-derived)"
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
