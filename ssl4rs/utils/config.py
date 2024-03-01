"""Contains utilities related to configuration tagging, parsing, and processing."""
import hashlib
import logging
import os
import pathlib
import platform
import re
import sys
import time
import typing
import warnings

import dotenv
import hydra
import hydra.conf
import hydra.core.hydra_config
import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
import omegaconf
import torch
import torch.cuda
import torch.distributed

DictConfig = typing.Union[typing.Dict[typing.AnyStr, typing.Any], omegaconf.DictConfig]
"""Type for configuration dictionaries that might be regular dicts or omegaconf dicts."""

cfg: typing.Optional[omegaconf.DictConfig] = None
"""Global reference to the app-level config dictionary; set in the `extra_inits` function."""


def extra_inits(
    config: omegaconf.DictConfig,
    logger: typing.Optional[logging.Logger] = None,
    set_as_global_cfg: bool = True,
    logging_captures_warnings: bool = True,
    output_dir: typing.Optional[typing.Union[typing.AnyStr, pathlib.Path]] = None,
) -> None:
    """Runs optional utilities initializations, controlled by config flags."""
    import ssl4rs.utils.logging  # used here to avoid circular dependencies

    logging.captureWarnings(logging_captures_warnings)
    if logger is None:
        logger = ssl4rs.utils.logging.get_logger(__name__)

    # optionally disable python warnings
    if config.utils.get("ignore_warnings"):
        logger.info("Disabling python warnings... (utils.ignore_warnings=True)")
        warnings.filterwarnings("ignore")

    # optionally pretty print config tree using Rich library
    if config.utils.get("print_config"):
        logger.info("Printing config tree with rich... (utils.print_config=True)")
        ssl4rs.utils.logging.print_config(config=config, resolve=True)

    # optionally create some logs in the output directory
    if output_dir is not None:
        log_extension = ssl4rs.utils.logging.get_log_extension_slug(config=config)
        if config.utils.get("log_installed_pkgs"):
            ssl4rs.utils.logging.log_installed_packages(output_dir, log_extension=log_extension)
        if config.utils.get("log_runtime_tags"):
            ssl4rs.utils.logging.log_runtime_tags(output_dir, log_extension=log_extension)
        if config.utils.get("log_interpolated_config"):
            ssl4rs.utils.logging.log_interpolated_config(config, output_dir, log_extension=log_extension)

    # we might reseed again elsewhere, but we'll at least do it here to make sure
    seed_everything(config)
    torch.use_deterministic_algorithms(config.utils.use_deterministic_algorithms)

    # finally, set the global config reference
    if set_as_global_cfg:
        global cfg
        cfg = config


def clear_global_config() -> None:
    """Clears (replaces the reference) to the global configuration dictionary.

    If one did not exist, this function does nothing. This function should likely be called in
    scripts where we 'initialize' experiments inside a single process repeatedly, or when doing so
    inside unit tests.
    """
    global cfg
    if cfg is not None:
        cfg = None


def seed_everything(config: omegaconf.DictConfig) -> int:
    """Pulls the seed from the config and resets the RNGs with it using Lightning.

    If the seed is not set (i.e. its value is `None`), a new seed will be picked randomly and set
    inside the config dictionary. In any case, the seed that is set is returned by the function.
    """
    set_seed = pl.seed_everything(config.utils.seed, workers=config.utils.seed_workers)
    if config.utils.seed is None:
        config.utils.seed = set_seed
    return set_seed


def get_hydra_config() -> hydra.conf.HydraConf:
    """Returns the hydra configuration dictionary for the current app.

    Will probably throw is the app was not started from a proper hydra entrypoint...
    """
    return hydra.core.hydra_config.HydraConfig.get()


def get_package_root_dir() -> pathlib.Path:
    """Returns the path to this package's root directory (i.e. where its modules are located)."""
    import ssl4rs.utils.filesystem  # used here to avoid circular dependencies

    return ssl4rs.utils.filesystem.get_package_root_dir()


def get_framework_root_dir() -> typing.Optional[pathlib.Path]:
    """Returns the path to this framework's root directory (i.e. where the source code is located).

    If the package was NOT installed from source, this function will return `None`.
    """
    import ssl4rs.utils.filesystem  # used here to avoid circular dependencies

    return ssl4rs.utils.filesystem.get_framework_root_dir()


def get_platform_name() -> str:
    """Returns a print-friendly platform name that can be used for logs / data tagging."""
    return str(platform.node())


def get_timestamp() -> str:
    """Returns a print-friendly timestamp (year, month, day, hour, minute, second) for logs."""
    return time.strftime("%Y%m%d-%H%M%S")


def get_failsafe_rank(group: typing.Optional[torch.distributed.ProcessGroup] = None) -> int:
    """Returns the result of torch.distributed.get_rank, or zero if not in a process group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(group)
    return 0


def get_failsafe_worldsize(group: typing.Optional[torch.distributed.ProcessGroup] = None) -> int:
    """Returns the result of torch.distributed.get_world_size, or -1 if not in a process group."""
    if torch.distributed.is_initialized():
        torch.distributed.get_world_size(group)
    return -1


def get_failsafe_backend(group: typing.Optional[torch.distributed.ProcessGroup] = None) -> str:
    """Returns the result of torch.distributed.get_backend, or "n/a" if not in a process group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_backend(group)
    return "n/a"


def get_git_revision_hash() -> str:
    """Returns a print-friendly hash (SHA1 signature) for the underlying git repository (if found).

    If a git repository is not found, the function will return a static string.
    """
    try:
        import git
    except (ImportError, AttributeError):
        return "git-import-error"
    try:
        repo = git.Repo(path=os.path.abspath(__file__), search_parent_directories=True)
        sha = repo.head.object.hexsha
        return str(sha)
    except (AttributeError, ValueError, git.InvalidGitRepositoryError):
        return "git-revision-unknown"


def get_runtime_tags(
    with_gpu_info: bool = False,
    with_distrib_info: bool = False,
) -> typing.Mapping[str, typing.Any]:
    """Returns a map (dictionary) of tags related to the current runtime."""
    import ssl4rs  # used here to avoid circular dependencies  # noqa
    import ssl4rs.utils.filesystem

    curr_time = time.time()
    curr_local_time = time.localtime(curr_time)
    tags = {
        "framework_name": "ssl4rs",
        "framework_version": ssl4rs.__version__,
        "framework_dir": str(ssl4rs.utils.filesystem.get_framework_root_dir()),
        "package_dir": str(ssl4rs.utils.filesystem.get_package_root_dir()),
        "curr_work_dir": os.getcwd(),
        "platform_name": get_platform_name(),
        "git_hash": get_git_revision_hash(),
        "time_since_epoch": curr_time,
        "local_timestamp": time.strftime("%Y-%m-%d_%H-%M-%S", curr_local_time),
        "runtime_hash": hashlib.sha1(str(curr_time).encode(), usedforsecurity=False).hexdigest(),
        "sys_argv": sys.argv,
    }
    if with_gpu_info:
        dev_count = torch.cuda.device_count()
        tags["cuda"] = {
            "is_available": torch.cuda.is_available(),
            "arch_list": torch.cuda.get_arch_list(),
            "device_count": dev_count,
            "device_names": [torch.cuda.get_device_name(i) for i in range(dev_count)],
            "device_capabilities": [torch.cuda.get_device_capability(i) for i in range(dev_count)],
        }
    if with_distrib_info:
        tags["distrib"] = {
            "is_available": torch.distributed.is_available(),
            "is_initialized": torch.distributed.is_initialized(),
            "backend": get_failsafe_backend(),
            "rank": get_failsafe_rank(),
            "world_size": get_failsafe_worldsize(),
        }
    return tags


def get_installed_packages() -> typing.List[str]:
    """Returns a list of all packages installed in the current environment.

    If the required packages cannot be imported, the returned list will be empty. Note that some
    packages may not be properly detected by this approach, and it is pretty hacky, so use it with a
    grain of salt (i.e. just for logging is fine).
    """
    try:
        import importlib.metadata

        pkgs = [f"{pkg.name}=={pkg.version}" for pkg in importlib.metadata.distributions()]
    except (ImportError, AttributeError):
        try:
            import pip  # noqa

            # noinspection PyUnresolvedReferences
            pkgs = [f"{pkg.key}=={pkg.version}" for pkg in pip.get_installed_distributions()]
        except (ImportError, AttributeError):
            pkgs = []
    return sorted(pkgs, key=str.casefold)


def get_params_hash(*args, **kwargs):
    """Computes and returns the hash (md5 checksum) of a given set of parameters.

    Args:
        Any combination of parameters that are hashable via their string representation.

    Returns:
        The hashing result as a string of hexadecimal digits.
    """
    # by default, will use the repr of all params but remove the 'at 0x00000000' addresses
    clean_str = re.sub(r" at 0x[a-fA-F\d]+", "", str(args) + str(kwargs))
    return hashlib.sha1(clean_str.encode(), usedforsecurity=False).hexdigest()


def get_framework_dotenv_path(
    framework_dir: typing.Optional[typing.Union[typing.AnyStr, pathlib.Path]] = None,
) -> typing.Optional[pathlib.Path]:
    """Returns the path to the framework's dotenv config file (if any).

    Args:
        framework_dir: the path to the framework directory that should contain a dotenv config file.
            If not specified, it will be automatically deduced from the package directory.

    Returns:
        The path to the dotenv file, if it exists.
    """
    if framework_dir is None:
        framework_dir = get_framework_root_dir()
    assert framework_dir is not None, "cannot auto-locate framework directory!"
    framework_dir = pathlib.Path(framework_dir)
    assert framework_dir.is_dir(), f"invalid framework directory: {framework_dir}"
    dotenv_path = framework_dir / ".env"
    if dotenv_path.is_file():
        return dotenv_path
    return None


def get_data_root_dir() -> pathlib.Path:
    """Returns the data root directory for the current environment/config setup.

    This function will first check if a config dictionary is registered inside the module, and
    return its `data_root_dir` value if possible. If not, it will try to get the data root
    directory directly from the already-loaded environment variables. If that fails, it will try to
    load the framework's local dotenv config file to see if a local environment variable can be
    used. If all attempts fail, it will throw an exception.
    """
    # first, check the globally registered cfg object
    global cfg
    if cfg is not None:
        try:
            return pathlib.Path(cfg.utils.data_root_dir)
        except omegaconf.errors.MissingMandatoryValue:
            pass
    # check the already-loaded environment variables
    data_root_dir = os.getenv("DATA_ROOT")
    if data_root_dir is not None:
        return pathlib.Path(data_root_dir)
    # check the framework directory for a local env file and load it manually
    framework_dir = get_framework_root_dir()
    assert framework_dir is not None and framework_dir.is_dir(), "could not locate framework dir!"
    framework_dotenv_path = get_framework_dotenv_path(framework_dir)
    assert framework_dotenv_path is not None, f"no dotenv config file found at: {framework_dir}"
    dotenv_config = dotenv.dotenv_values(dotenv_path=framework_dotenv_path)
    data_root_dir = dotenv_config.get("DATA_ROOT", None)
    assert data_root_dir is not None, "could not find the data root dir anywhere!"
    return pathlib.Path(data_root_dir)


def get_output_root_dir() -> pathlib.Path:
    """Returns the output (log) root directory for the current environment/config setup.

    This function will first check if a config dictionary is registered inside the module, and
    return its `output_root_dir` value if possible. If not, it will try to get the output root
    directory directly from the already-loaded environment variables. If that fails, it will try to
    load the framework's local dotenv config file to see if a local environment variable can be
    used. If all attempts fail, it will throw an exception.
    """
    # first, check the globally registered cfg object
    global cfg
    if cfg is not None:
        try:
            return pathlib.Path(cfg.utils.output_root_dir)
        except omegaconf.errors.MissingMandatoryValue:
            pass
    # check the already-loaded environment variables
    output_root_dir = os.getenv("OUTPUT_ROOT")
    if output_root_dir is not None:
        return pathlib.Path(output_root_dir)
    # check the framework directory for a local env file and load it manually
    framework_dir = get_framework_root_dir()
    assert framework_dir is not None and framework_dir.is_dir(), "could not locate framework dir!"
    framework_dotenv_path = get_framework_dotenv_path(framework_dir)
    assert framework_dotenv_path is not None, f"no dotenv config file found at: {framework_dir}"
    dotenv_config = dotenv.dotenv_values(dotenv_path=framework_dotenv_path)
    output_root_dir = dotenv_config.get("OUTPUT_ROOT", None)
    assert output_root_dir is not None, "could not find the output root dir anywhere!"
    return pathlib.Path(output_root_dir)


def get_latest_checkpoint(config: omegaconf.DictConfig) -> typing.Optional[pathlib.Path]:
    """Returns the path to the latest checkpoint for the current run.

    If no checkpoint exists, the function will return `None`.
    """
    checkpoint_dir_path = pathlib.Path(config.utils.checkpoint_dir_path)
    if not checkpoint_dir_path.is_dir():
        return None
    expected_last_ckpt_path = checkpoint_dir_path / "last.ckpt"
    if expected_last_ckpt_path.is_file():
        return expected_last_ckpt_path
    # otherwise, assume checkpoint names are sortable, and the last will be the latest
    available_ckpts = list(checkpoint_dir_path.glob("*.ckpt"))
    if len(available_ckpts) == 0:
        return None
    return sorted(available_ckpts)[-1]


def get_callbacks(config: omegaconf.DictConfig) -> typing.List[pl.Callback]:
    """Returns the list of callbacks to pass to the trainer for the current run.

    If no callbacks are specified, the list will be empty.
    """
    import ssl4rs.utils.logging  # used here to avoid circular dependencies

    logger = ssl4rs.utils.logging.get_logger(__name__)
    callbacks: typing.List[pl.Callback] = []
    if "callbacks" in config:
        for cb_name, cb_config in config.callbacks.items():
            logger.info(f"Instantiating '{cb_name}' callback: {cb_config._target_}")  # noqa
            callbacks.append(hydra.utils.instantiate(cb_config))
    return callbacks


def get_loggers(config: omegaconf.DictConfig) -> typing.List[pl_loggers.Logger]:
    """Returns the list of callbacks to pass to the trainer for the current run.

    If no callbacks are specified, the list will be empty.
    """
    import ssl4rs.utils.logging  # used here to avoid circular dependencies

    logger = ssl4rs.utils.logging.get_logger(__name__)
    loggers: typing.List[pl_loggers.Logger] = []
    if "logger" in config:
        for lg_name, lg_config in config.logger.items():
            logger.info(f"Instantiating '{lg_name}' logger: {lg_config._target_}")  # noqa
            loggers.append(hydra.utils.instantiate(lg_config))
    return loggers


def get_model(config: omegaconf.DictConfig) -> torch.nn.Module:
    """Returns the instantiated and potentially compiled model to pass to the trainer."""
    import ssl4rs.utils.logging  # used here to avoid circular dependencies

    logger = ssl4rs.utils.logging.get_logger(__name__)
    logger.info(f"Instantiating model: {config.model._target_}")  # noqa
    model: pl.LightningModule = hydra.utils.instantiate(config.model)
    if config.get("compile_model", False):
        logger.debug("Compiling model...")
        compile_model_kwargs = config.get("compile_model_kwargs", None)
        if not compile_model_kwargs:
            compile_model_kwargs = {}
        assert isinstance(compile_model_kwargs, (dict, omegaconf.DictConfig))
        model = torch.compile(model, **compile_model_kwargs)
        logger.debug("Compilation complete")
    return model


def init_hydra_and_compose_config(
    version_base: typing.Optional[typing.AnyStr] = None,
    configs_dir: typing.Optional[typing.Union[typing.AnyStr, pathlib.Path]] = None,
    config_name: typing.AnyStr = "ssl4rs-profiler.yaml",
    data_root_dir: typing.Optional[typing.Union[typing.AnyStr, pathlib.Path]] = None,
    output_root_dir: typing.Optional[typing.Union[typing.AnyStr, pathlib.Path]] = None,
    overrides: typing.List[typing.AnyStr] = None,
    set_as_global_cfg: bool = True,
    caller_stack_depth: int = 2,
) -> omegaconf.DictConfig:
    """Initializes hydra and returns a config as a composition output.

    This function is meant to be used by local entrypoints that are not the 'main' scripts used in
    the framework (such as `train.py` and `test.py`) in order to allow them to access a full hydra
    config. Unit tests and analysis scripts will likely rely a lot on this...

    Args:
        version_base: hydra version argument to forward to the initialization function (if any).
        configs_dir: Path to the `configs` directory that contains all the config files for the
            framework. If not specified, we'll try to detect/find it automatically using the
            relative path between the cwd and the framework directory (which is not super safe!).
        config_name: name of the configuration file to load by default as the compose target.
        data_root_dir: path to the data root directory, if it needs to be specified or modified.
            If not specified, the default will be used based on the environment variable.
        output_root_dir: path to the output root directory, if it needs to be specified or modified.
            If not specified, the default will be used based on the environment variable.
        overrides: list of overrides to be provided to hydra's compose method.
        set_as_global_cfg: defines whether to store the loaded config as the global config or not.
        caller_stack_depth: depth of the parent caller function; might need to be adjusted if
            called from outside the framework itself.

    Returns:
        The result of the config composition.
    """
    # first, try to load dotenv file in case we have any mandatory cfg values derived from there
    dotenv_path = get_framework_dotenv_path()
    if dotenv_path is not None:
        dotenv.load_dotenv(dotenv_path=str(dotenv_path), override=True, verbose=True)
    # next, if the config dir is not provided, find it and get the relative path to it
    if configs_dir is None:
        pkg_root_dir = get_package_root_dir()
        assert pkg_root_dir is not None, "cannot auto-locate package directory"
        configs_dir = pkg_root_dir / "configs"
        configs_dir = pathlib.Path(os.path.relpath(str(configs_dir), str(pathlib.Path.cwd())))
        assert configs_dir.is_dir(), f"invalid configs dir: {configs_dir}"
        base_config_files = [f.name for f in configs_dir.iterdir() if f.is_file()]
        assert all(
            [f in base_config_files for f in ["ssl4rs-train.yaml", "ssl4rs-test.yaml"]]
        ), f"found invalid root config directory using relpath: {configs_dir}"
    # setup overrides based on provided args
    overrides = [] if overrides is None else [o for o in overrides]
    if data_root_dir is not None:
        overrides.append(f"utils.data_root_dir={str(data_root_dir)}")
    if output_root_dir is not None:
        overrides.append(f"utils.output_root_dir={str(output_root_dir)}")
    # initialize hydra and return the resulting config
    with hydra.initialize(
        version_base=version_base,
        config_path=str(configs_dir),
        caller_stack_depth=caller_stack_depth,
    ):
        config = hydra.compose(config_name=config_name, overrides=overrides)
        extra_inits(config, set_as_global_cfg=set_as_global_cfg)
    return config
