"""Contains utilities related to logging data to terminal or filesystem."""
import logging
import pathlib
import typing

import omegaconf
import pytorch_lightning
import pytorch_lightning.loggers
import pytorch_lightning.utilities
import rich.syntax
import rich.tree

import ssl4rs

default_print_configs = (
    "utils",
    "output",
    "data",
    "model",
    "callbacks",
    "logger",
    "trainer",
)
"""This is the (ordered) list of configs that we'll print when asked to, by default."""


def get_logger(*args, **kwargs) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger.

    This call will ensure that all logging levels are set according to the rank zero decorator so
    that only log calls made from a single GPU process (the rank-zero one) will be kept.
    """
    logger = logging.getLogger(*args, **kwargs)
    possible_log_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in possible_log_levels:
        setattr(logger, level, pytorch_lightning.utilities.rank_zero_only(getattr(logger, level)))
    return logger


logger = get_logger(__name__)


@pytorch_lightning.utilities.rank_zero_only
def print_config(
    config: omegaconf.DictConfig,
    print_configs: typing.Sequence[str] = default_print_configs,
    resolve: bool = True,
    output_dir: typing.Optional[typing.Union[typing.AnyStr, pathlib.Path]] = None,
) -> None:
    """Prints the content of the given config and its tree structure to the console using Rich.

    If an output directory is specified, the result will also be logged to a file there.

    Args:
        config: the configuration composed by Hydra to be printed.
        print_configs: the name and order of the config components to print.
        resolve: toggles whether to resolve reference fields inside the config or not.
        output_dir: the output directory (for e.g. the experiment) where logs can be saved.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    quee = []
    for config_name in print_configs:
        if config_name in config:
            quee.append(config_name)
        else:
            logger.warning(f"Could not found (sub)config with name '{config_name}'")
    for field in config:
        if field not in quee:
            quee.append(field)
    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = config[field]
        if isinstance(config_group, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_log_path = output_dir / "config_tree.log"
        with open(str(output_log_path), "w") as fd:
            rich.print(tree, file=fd)


@pytorch_lightning.utilities.rank_zero_only
def log_hyperparameters(
    config: omegaconf.DictConfig,
    model: pytorch_lightning.LightningModule,
    datamodule: pytorch_lightning.LightningDataModule,
    trainer: pytorch_lightning.Trainer,
    callbacks: typing.List[pytorch_lightning.Callback],
    loggers: typing.List[pytorch_lightning.loggers.LightningLoggerBase],
) -> None:
    """Forwards all notable/interesting/important hyperparameters to the model logger.

    If the model does not have a logger that implements the `log_hyperparams`, this function
    does nothing. Note that hyperparameters (at least, those define via config files) will
    always be automatically logged in `${hydra.runtime.output_dir}`.
    """
    if not trainer.logger:
        return  # no logger to use, nothing to do...

    hparams = {}  # we'll fill this dict with all the hyperparams we want to log
    hparams["model"] = config["model"]  # all model-related stuff is going in for sure
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    # data and training configs should also go in for sure (they should also always exist)
    hparams["data"] = config["data"]
    hparams["trainer"] = config["trainer"]
    # the following hyperparameters are individually picked and might be missing (no big deal)
    optional_hyperparams_to_log = (
        "callbacks",
        "data_root_dir",
        "output_root_dir",
        "experiment_name",
        "run_and_job_name",
        "run_type",
        "seed",
    )
    for hyperparam_name in optional_hyperparams_to_log:
        if hyperparam_name in config:
            hparams[hyperparam_name] = config[hyperparam_name]

    # send hparams to the trainer's logger(s)
    trainer.logger.log_hyperparams(hparams)  # type: ignore


def log_runtime_tags(
    output_dir: typing.Union[typing.AnyStr, pathlib.Path],
) -> None:
    """Saves a list of all runtime tags to a log file.

    Args:
        output_dir: the output directory inside which we should be saving the package log.
    """
    output_dir = pathlib.Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_log_path = output_dir / "runtime_tags.log"
    tag_dict = omegaconf.OmegaConf.create(ssl4rs.utils.config.get_runtime_tags())  # type: ignore
    with open(str(output_log_path), "w") as fd:
        fd.write(omegaconf.OmegaConf.to_yaml(tag_dict))


def log_installed_packages(
    output_dir: typing.Union[typing.AnyStr, pathlib.Path],
) -> None:
    """Saves a list of all packages installed in the current environment to a log file.

    Args:
        output_dir: the output directory inside which we should be saving the package log.
    """
    output_dir = pathlib.Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_log_path = output_dir / "installed_pkgs.log"
    with open(str(output_log_path), "w") as fd:
        for pkg_name in ssl4rs.utils.config.get_installed_packages():
            fd.write(f"{pkg_name}\n")
