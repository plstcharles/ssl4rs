"""Contains utilities related to logging data to terminal or filesystem."""
import datetime
import functools
import logging
import pathlib
import pickle
import sys
import time
import typing

import dotenv
import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch.utilities as pl_utils
import omegaconf
import pandas as pd
import rich.syntax
import rich.tree
import torch
import yaml

default_print_configs = (
    "data",
    "model",
    "callbacks",
    "logger",
    "trainer",
)
"""This is the (ordered) list of configs that we'll print when asked to, by default."""

_root_logger_is_setup_for_analysis_script = False
"""Defines whether the setup_logging_for_analysis_script function has been called or not."""


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


def setup_logging_for_analysis_script(level: int = logging.INFO) -> logging.Logger:
    """Sets up logging with some console-only verbose settings for analysis scripts.

    THIS SHOULD NEVER BE USED IN GENERIC CODE OR OUTSIDE AN ENTRYPOINT; in other words, the only
    place you should ever see this function get called is close to a `if __name__ == "__main__":`
    statement in standalone analysis scripts, or in the first cell of a notebook. It should also
    never be called more than once, and it will reset the handlers attached to the root logger.

    The function returns a logger with the framework name which may be used/ignored as needed.
    """
    import ssl4rs.utils.config

    global _root_logger_is_setup_for_analysis_script
    if not _root_logger_is_setup_for_analysis_script:
        root = logging.getLogger()
        for h in root.handlers:  # reset all root handlers, in case this is called multiple times
            root.removeHandler(h)
        root.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
        dotenv_path = ssl4rs.utils.config.get_framework_dotenv_path()
        if dotenv_path is not None:
            dotenv.load_dotenv(dotenv_path=str(dotenv_path), override=True, verbose=True)
        _root_logger_is_setup_for_analysis_script = True
    logger_ = get_logger("ssl4rs")
    logger_.info("Logging set up for analysis script; runtime info:")
    for key, value in ssl4rs.utils.config.get_runtime_tags(with_gpu_info=True).items():
        logger_.info(f"{key}: {value}")
    return logger_


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


class DebugLogger(pl_loggers.Logger):
    """Implements a logger class used to test/check what is getting logged and when.

    Everything given to this logger with be dumped to disk in PICKLE format, with a separate pickle
    for each call. This is probably going to be a huge bottleneck in practice, so it should NEVER be
    used in real experiments, and only for debugging and testing.
    """

    def __init__(self, output_root_path: typing.Union[typing.AnyStr, pathlib.Path]) -> None:
        """Logs some metadata at the initialization of the logger and picks a subdirectory name."""
        import ssl4rs.utils.config  # doing it here to avoid circular imports

        super().__init__()
        self._output_root_path = pathlib.Path(output_root_path)
        self._output_root_path.mkdir(exist_ok=True)
        tag_dict = ssl4rs.utils.config.get_runtime_tags(
            with_gpu_info=True,
            with_distrib_info=True,
        )
        self._output_path = self._output_root_path / f"debug-logs-{tag_dict['runtime_hash']}"
        self._output_path.mkdir(exist_ok=False)
        self._log_stuff(**tag_dict)

    @property
    def root_dir(self) -> str:
        """Return the root directory where pickled logs get saved."""
        return str(self._output_root_path)

    @property
    def log_dir(self) -> str:
        """Return the directory where pickled logs for this particular logger get saved."""
        return str(self._output_path)

    @property
    def save_dir(self) -> str:
        """Return the directory where pickled logs for this particular logger get saved."""
        return self.log_dir

    @property
    def experiment(self) -> None:
        """Returns the experiment object associated with this logger."""
        return None

    @property
    def name(self) -> str:
        """Name of the logger."""
        return ""  # empty to bypass default ckpt subfolder structure

    @property
    def version(self) -> str:
        """Version of the logger."""
        return ""  # empty to bypass default ckpt subfolder structure

    @pl_utils.rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs) -> None:
        """Records experiment hyperparameters."""
        self._log_stuff(_log_func_name="log_hyperparams", params=params, *args, **kwargs)

    @pl_utils.rank_zero_only
    def log_metrics(self, metrics, step: typing.Optional[int] = None) -> None:
        """Records metric values (a dict of names-to-value pairs) at a specific experiment step."""
        self._log_stuff(_log_func_name="log_metrics", metrics=metrics, step=step)

    @pl_utils.rank_zero_only
    def log_graph(self, model, input_array: typing.Optional[torch.Tensor] = None) -> None:
        """Records model graph."""
        self._log_stuff(_log_func_name="log_graph", model=model, input_array=input_array)

    def __getitem__(self, idx: int) -> "DebugLogger":
        """Returns self to enable `self.logger[0].log_(...)`-style uses."""
        return self

    def __getattr__(self, name: str) -> typing.Callable:
        """Allows the logger to be called with arbitrary logging methods."""
        if name.startswith("log_"):
            return functools.partial(self._log_stuff, _log_func_name=name)
        else:
            raise AttributeError

    def _log_stuff(self, *args: typing.Any, _log_func_name=None, **kwargs: typing.Any) -> None:
        """Logs any kind of stuff (passed as args and kwargs) to disk using pickle.

        This will make sure that the log exists by selecting a proper log index suffix if needed.
        """
        curr_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_log_idx = 0
        expected_output_log_path = None
        while expected_output_log_path is None or expected_output_log_path.is_file():
            assert base_log_idx < 9999, "how is this even possible"
            file_name = f"log.{curr_timestamp}.{base_log_idx:04}.pkl"
            expected_output_log_path = pathlib.Path(self.log_dir) / file_name
            base_log_idx += 1
        with open(expected_output_log_path, "wb") as fd:
            pickle.dump({"func_name": _log_func_name, "args": args, "kwargs": kwargs}, fd)

    @staticmethod
    def parse_metric_logs(logs_path: typing.Union[typing.AnyStr, pathlib.Path]) -> pd.DataFrame:
        """Parses and returns the METRICS logs dumped as pickles in a particular directory.

        All logged outputs that were NOT metrics will NOT be reloaded by this function. Here, we
        assume all metrics are actually numerical values, and the ones that are not available at a
        specific step will be filled with NaNs. The output is returned as a pandas dataframe,
        indexed by the timestamps, with both metrics and steps as columns.
        """
        logs_path = pathlib.Path(logs_path)
        assert logs_path.is_dir(), "invalid debug logs root dir"
        logged_files = list(logs_path.glob("*.pkl"))
        assert len(logged_files) > 0, "not logs found"
        logged_metrics_data = []
        for logged_file in logged_files:
            with open(logged_file, "rb") as fd:
                data = pickle.load(fd)
                if data["func_name"] == "log_metrics":
                    logged_metrics_data.append(
                        {
                            "timestamp": "_".join(logged_file.name.split(".")[1:3]),
                            "step": data["kwargs"]["step"],
                            **data["kwargs"]["metrics"],
                        }
                    )
        assert len(logged_metrics_data) > 0, "no metrics logged"
        output = pd.DataFrame(logged_metrics_data).sort_values("timestamp").set_index("timestamp")
        return output
