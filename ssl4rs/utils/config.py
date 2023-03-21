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
import hydra.core.hydra_config
import omegaconf
import pytorch_lightning
import torch.cuda

import ssl4rs.utils.logging

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
        log_extension = ssl4rs.utils.logging.get_log_extension_slug()
        if config.utils.get("log_installed_pkgs"):
            ssl4rs.utils.logging.log_installed_packages(output_dir, log_extension=log_extension)
        if config.utils.get("log_runtime_tags"):
            ssl4rs.utils.logging.log_runtime_tags(output_dir, log_extension=log_extension)
        if config.utils.get("log_interpolated_config"):
            ssl4rs.utils.logging.log_interpolated_config(config, output_dir, log_extension=log_extension)

    # we might reseed again elsewhere, but we'll at least do it here to make sure
    seed_everything(config)

    # finally, set the global config reference
    if set_as_global_cfg:
        global cfg
        cfg = config


def seed_everything(config: omegaconf.DictConfig) -> int:
    """Pulls the seed from the config and resets the RNGs with it using pytorch-lightning.

    If the seed is not set (i.e. its value is `None`), a new seed will be picked randomly and set
    inside the config dictionary. In any case, the seed that is set is returned by the function.
    """
    seed, seed_workers = config.get("seed", None), config.get("seed_workers")
    assert isinstance(seed_workers, bool)
    set_seed = pytorch_lightning.seed_everything(seed, workers=seed_workers)
    if seed is None:
        config["seed"] = set_seed
    return set_seed


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


def get_runtime_tags(with_gpu_info: bool = False) -> typing.Mapping[str, typing.Any]:
    """Returns a map (dictionary) of tags related to the current runtime."""
    import ssl4rs  # used here to avoid circular dependencies
    import ssl4rs.utils.filesystem

    tags = {
        "framework_name": "ssl4rs",
        "framework_version": ssl4rs.__version__,
        "framework_dir": str(ssl4rs.utils.filesystem.get_framework_root_dir()),
        "package_dir": str(ssl4rs.utils.filesystem.get_package_root_dir()),
        "platform_name": get_platform_name(),
        "git_hash": get_git_revision_hash(),
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
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
    return tags


def get_installed_packages() -> typing.List[str]:
    """Returns a list of all packages installed in the current environment.

    If the required packages cannot be imported, the returned list will be empty. Note that some
    packages may not be properly detected by this approach, and it is pretty hacky, so use it with
    a grain of salt (i.e. just for logging is fine).
    """
    try:
        import pip

        # noinspection PyUnresolvedReferences
        pkgs = pip.get_installed_distributions()
        return list(sorted(f"{pkg.key} {pkg.version}" for pkg in pkgs))
    except (ImportError, AttributeError):
        try:
            import pkg_resources as pkgr

            return list(sorted(str(pkg) for pkg in pkgr.working_set))
        except (ImportError, AttributeError):
            return []


def get_params_hash(*args, **kwargs):
    """Computes and returns the hash (md5 checksum) of a given set of parameters.

    Args:
        Any combination of parameters that are hashable via their string representation.

    Returns:
        The hashing result as a string of hexadecimal digits.
    """
    # by default, will use the repr of all params but remove the 'at 0x00000000' addresses
    clean_str = re.sub(r" at 0x[a-fA-F\d]+", "", str(args) + str(kwargs))
    return hashlib.sha1(clean_str.encode()).hexdigest()


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


def init_hydra_and_compose_config(
    version_base: typing.Optional[typing.AnyStr] = None,
    configs_dir: typing.Optional[typing.Union[typing.AnyStr, pathlib.Path]] = None,
    config_name: typing.AnyStr = "train.yaml",
    data_root_dir: typing.Optional[typing.Union[typing.AnyStr, pathlib.Path]] = None,
    output_root_dir: typing.Optional[typing.Union[typing.AnyStr, pathlib.Path]] = None,
    overrides: typing.List[typing.AnyStr] = None,
    set_as_global_cfg: bool = True,
) -> omegaconf.DictConfig:
    """Initializes hydra and returns a config as a composition output.

    This function is meant to be used by local entrypoints that are not the 'main' scripts used in
    the framework (such as `train.py` and `test.py`) in order to allow them to access a full hydra
    config. Unit tests will likely rely a lot on this...

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

    Returns:
        The result of the config composition.
    """
    if configs_dir is None:
        framework_dir = get_framework_root_dir()
        assert framework_dir is not None, "cannot auto-locate framework directory!"
        configs_dir = framework_dir / "configs"
        configs_dir = pathlib.Path(os.path.relpath(str(configs_dir), str(pathlib.Path.cwd())))
        assert configs_dir.is_dir(), f"invalid configs dir: {configs_dir}"
        base_config_files = [f.name for f in configs_dir.iterdir() if f.is_file()]
        assert all(
            [f in base_config_files for f in ["train.yaml", "test.yaml"]]
        ), f"found invalid root config directory using relpath: {configs_dir}"
    overrides = [] if overrides is None else [o for o in overrides]
    if data_root_dir is not None:
        overrides.append(f"++utils.data_root_dir={str(data_root_dir)}")
    if output_root_dir is not None:
        overrides.append(f"++utils.output_root_dir={str(output_root_dir)}")
    with hydra.initialize(version_base=version_base, config_path=str(configs_dir), caller_stack_depth=2):
        config = hydra.compose(config_name=config_name, overrides=overrides)
    extra_inits(config, set_as_global_cfg=set_as_global_cfg)
    return config
