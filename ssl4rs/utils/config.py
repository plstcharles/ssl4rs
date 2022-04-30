"""Contains utilities related to configuration tagging, parsing, and processing."""
import hashlib
import os
import pathlib
import platform
import re
import time
import typing
import warnings

import omegaconf
import pytorch_lightning

import ssl4rs

logger = ssl4rs.utils.get_logger(__name__)


def extra_inits(config: omegaconf.DictConfig) -> None:
    """Runs optional utilities initializations, controlled by config flags."""

    potential_output_dir = config.get("hydra.runtime.output_dir", None)

    # optionally disable python warnings
    if config.get("ignore_warnings"):
        logger.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # optionally pretty print config tree using Rich library
    if config.get("print_config"):
        logger.info("Printing config tree with Rich! <config.print_config=True>")
        ssl4rs.utils.logging.print_config(
            config=config,
            resolve=True,
            output_dir=potential_output_dir,
        )

    # optionally create some logs in the output directory
    if potential_output_dir is not None:
        if config.get("log_installed_pkgs"):
            ssl4rs.utils.logging.log_installed_packages(potential_output_dir)
        if config.get("log_runtime_tags"):
            ssl4rs.utils.logging.log_runtime_tags(potential_output_dir)

    # we might reseed again elsewhere, but we'll at least do it here to make sure
    seed_everything(config)


def seed_everything(config: omegaconf.DictConfig) -> None:
    """Pulls the seed from the config and resets the RNGs with it using pytorch-lightning."""
    seed, seed_workers = config.get("seed", None), config.get("seed_workers")
    assert isinstance(seed_workers, bool)
    pytorch_lightning.seed_everything(seed, workers=seed_workers)


def get_framework_dir() -> typing.Optional[pathlib.Path]:
    """Returns the root directory of the framework (i.e. the parent of the package directory).

    If the framework root does not exist (i.e. if the package was installed directly via pip),
    this function will return None.
    """
    package_root_dir = pathlib.Path(ssl4rs.__file__).parent  # there's no way this one does not exist
    framework_dir = package_root_dir.parent  # this is a candidate that might be wrong
    # we'll validate that we landed in the right place by checking for some files we expect to see...
    expected_framework_files = ["setup.py", "environment.yaml", ".gitignore", "train.py"]
    found_all_files = all([(framework_dir / f).is_file() for f in expected_framework_files])
    if not found_all_files:
        return None  # we did not manage to guess the framework dir location...
    return framework_dir


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


def get_runtime_tags() -> typing.Mapping[str, typing.Any]:
    """Returns a map (dictionary) of tags related to the current runtime."""
    tags = {
        "framework_name": "ssl4rs",
        "framework_version": ssl4rs.__version__,
        "framework_dir": get_framework_dir(),
        "platform_name": get_platform_name(),
        "git_hash": get_git_revision_hash(),
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
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
        return list(sorted(["%s %s" % (pkg.key, pkg.version) for pkg in pkgs]))
    except (ImportError, AttributeError):
        try:
            import pkg_resources as pkgr
            return list(sorted([str(pkg) for pkg in pkgr.working_set]))
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
