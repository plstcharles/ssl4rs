"""Contains utilities related to logging data to terminal or filesystem."""
import datetime
import logging
import pathlib
import sys
import typing

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch.utilities as pl_utils
import omegaconf
import rich.syntax
import rich.tree
import torch
import torch.distributed
import yaml

default_print_configs = (
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
        setattr(logger, level, pl_utils.rank_zero_only(getattr(logger, level)))
    return logger


logger = get_logger(__name__)


def setup_logging_for_analysis_script(level: int = logging.DEBUG) -> logging.Logger:
    """Sets up logging with some console-only verbose settings for analysis scripts.

    THIS SHOULD NEVER BE USED IN GENERIC CODE OR OUTSIDE AN ENTRYPOINT; in other words, the only
    place you should ever see this function get called is close to a `if __name__ == "__main__":`
    statement in standalone analysis scripts. It should also never be called more than once, and
    it will reset the handlers attached to the root logger.

    The function returns a logger with the framework name which may be used/ignored as needed.
    """
    root = logging.getLogger()
    for h in root.handlers:  # reset all root handlers, in case this is called multiple times
        root.removeHandler(h)
    root.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    return get_logger("ssl4rs")


@pl_utils.rank_zero_only
def print_config(
    config: omegaconf.DictConfig,
    print_configs: typing.Sequence[str] = default_print_configs,
    resolve: bool = True,
) -> None:
    """Prints the content of the given config and its tree structure to the console using Rich.

    Args:
        config: the configuration composed by Hydra to be printed.
        print_configs: the name and order of the config components to print.
        resolve: toggles whether to resolve reference fields inside the config or not.
    """
    tree = rich.tree.Tree("CONFIG")
    queue = []
    for config_name in print_configs:
        if config_name in config:
            queue.append(config_name)
    for field in config:
        if field not in queue:
            queue.append(field)
    for field in queue:
        branch = tree.add(field)
        config_group = config[field]
        if isinstance(config_group, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)


@pl_utils.rank_zero_only
def log_hyperparameters(
    config: omegaconf.DictConfig,
    model: typing.Union[pl.LightningModule, torch.nn.Module],
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: typing.List[pl.Callback],
    loggers: typing.List[pl_loggers.Logger],
) -> None:
    """Forwards all notable/interesting/important hyperparameters to the trainer's logger.

    If the trainer does not have a logger that implements the `log_hyperparams`, this function does
    nothing. Note that hyperparameters (at least, those defined via config files) will always be
    automatically logged in `${hydra:runtime.output_dir}`.
    """
    if not trainer.logger:
        return  # no logger to use, nothing to do...

    hparams = dict()  # we'll fill this dict with all the hyperparams we want to log
    hparams["model"] = config["model"]  # all model-related stuff is going in for sure
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    # data and training configs should also go in for sure (they should also always exist)
    hparams["data"] = config["data"]
    hparams["trainer"] = config["trainer"]
    # the following hyperparameters are individually picked and might be missing (no big deal)
    optional_hyperparams_to_log = (
        "experiment_name",
        "run_type",
        "run_name",
        "job_name",
        "seed",
        "seed_workers",
    )
    for hyperparam_name in optional_hyperparams_to_log:
        if hyperparam_name in config:
            hparams[hyperparam_name] = config[hyperparam_name]
    for lgger in trainer.loggers:
        if hasattr(lgger, "log_hyperparams"):
            trainer.logger.log_hyperparams(hparams)  # type: ignore
    for hparam_key, hparam_val in hparams.items():
        logger.debug(f"{hparam_key}: {hparam_val}")


def get_log_extension_slug(config: omegaconf.DictConfig) -> str:
    """Returns a log file extension that includes a timestamp (for non-overlapping, sortable logs).

    The 'rounded seconds since epoch' portion will be computed when this function is called, whereas
    the timestamp will be derived from the hydra config's `utils.curr_timestamp` value. This will
    help make sure that similar logs saved within the same run will not overwrite each other.

    The output format is:
        `.{TIMESTAMP_WITH_DATE_AND_TIME}.{ROUNDED_SECS_SINCE_EPOCH}.rank{RANK_ID}.log`
    """
    import ssl4rs.utils.config  # doing it here to avoid circular imports

    curr_time = datetime.datetime.now()
    epoch_time_sec = int(curr_time.timestamp())  # for timezone independence
    timestamp = config.utils.curr_timestamp
    rank_id = ssl4rs.utils.config.get_failsafe_rank()
    return f".{timestamp}.{epoch_time_sec}.rank{rank_id:02d}.log"


def log_runtime_tags(
    output_dir: typing.Union[typing.AnyStr, pathlib.Path],
    with_gpu_info: bool = True,
    with_distrib_info: bool = True,
    log_extension: typing.AnyStr = ".log",
) -> None:
    """Saves a list of all runtime tags to a log file.

    Note: this may run across multiple processes simultaneously as long as the rank ID is used
    inside the log file extension.

    Args:
        output_dir: the output directory inside which we should be saving the package log.
        with_gpu_info: defines whether to log available GPU device info or not.
        with_distrib_info: defines whether to log available distribution backend/rank info or not.
        log_extension: extension to use in the log's file name.
    """
    import ssl4rs.utils.config  # doing it here to avoid circular imports

    output_dir = pathlib.Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_log_path = output_dir / f"runtime_tags{log_extension}"
    tag_dict = ssl4rs.utils.config.get_runtime_tags(
        with_gpu_info=with_gpu_info,
        with_distrib_info=with_distrib_info,
    )
    tag_dict = omegaconf.OmegaConf.create(tag_dict)  # type: ignore
    with open(str(output_log_path), "w") as fd:
        fd.write(omegaconf.OmegaConf.to_yaml(tag_dict))


def log_installed_packages(
    output_dir: typing.Union[typing.AnyStr, pathlib.Path],
    log_extension: typing.AnyStr = ".log",
) -> None:
    """Saves a list of all packages installed in the current environment to a log file.

    Note: this may run across multiple processes simultaneously as long as the rank ID is used
    inside the log file extension.

    Args:
        output_dir: the output directory inside which we should be saving the package log.
        log_extension: extension to use in the log's file name.
    """
    import ssl4rs.utils.config  # doing it here to avoid circular imports

    output_dir = pathlib.Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_log_path = output_dir / f"installed_pkgs{log_extension}"
    with open(str(output_log_path), "w") as fd:
        for pkg_name in ssl4rs.utils.config.get_installed_packages():
            fd.write(f"{pkg_name}\n")


def log_interpolated_config(
    config: omegaconf.DictConfig,
    output_dir: typing.Union[typing.AnyStr, pathlib.Path],
    log_extension: typing.AnyStr = ".log",
) -> None:
    """Saves the interpolated configuration file content to a log file in YAML format.

    If a configuration parameter cannot be interpolated because it is missing, this will throw
    an exception.

    Note: this may run across multiple processes simultaneously as long as the rank ID is used
    inside the log file extension.

    Note: this file should never be used to try to reload a completed run! (refer to hydra
    documentation on how to do that instead)

    Args:
        config: the not-yet-interpolated omegaconf dictionary that contains all parameters.
        output_dir: the output directory inside which we should be saving the package log.
        log_extension: extension to use in the log's file name.
    """
    output_dir = pathlib.Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_log_path = output_dir / f"config{log_extension}"
    with open(str(output_log_path), "w") as fd:
        yaml.dump(omegaconf.OmegaConf.to_object(config), fd)


@pl_utils.rank_zero_only
def finalize_logs(
    config: omegaconf.DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: typing.List[pl.Callback],
    loggers: typing.List[pl_loggers.Logger],
) -> None:
    """Makes sure everything is logged and closed properly before ending the session."""
    for lg in loggers:
        if isinstance(lg, pl_loggers.wandb.WandbLogger):
            # without this, sweeps with wandb logger might crash!
            import wandb

            wandb.finish()
