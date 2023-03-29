import functools
import logging
import pathlib
import typing

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch.utils.data

import ssl4rs

logger = ssl4rs.utils.get_logger(__name__)


def _common_init(
    config: omegaconf.DictConfig,
) -> typing.Tuple[
    typing.Tuple[str, str, str, str],
    ssl4rs.data.DataModule,
]:  # returns ((exp_name, run_name, run_type, job_name), datamodule)
    """Runs the common (for-all-profiling-types) initialization stuff."""
    exp_name, run_name, run_type, job_name = config.experiment_name, config.run_name, config.run_type, config.job_name
    logger.info(f"Launching ({exp_name}: {run_name}, '{run_type}', job={job_name})")

    hydra_config = ssl4rs.utils.config.get_hydra_config()
    output_dir = pathlib.Path(hydra_config.runtime.output_dir)
    logger.info(f"Output directory: {output_dir.absolute()}")
    ssl4rs.utils.config.extra_inits(config, output_dir=output_dir)

    logger.info(f"Instantiating datamodule: {config.data.datamodule._target_}")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)
    assert isinstance(
        datamodule, ssl4rs.data.DataModule
    ), "invalid datamodule base class (need `ssl4rs.data.DataModule` for getters below!)"

    return (exp_name, run_name, run_type, job_name), datamodule


def _get_dataloader(
    config: omegaconf.DictConfig,
    datamodule: ssl4rs.data.DataModule,
) -> typing.Union[torch.utils.data.DataLoader, ssl4rs.data.DataParser]:
    """Returns the dataloader object we'll be profiling.

    The object that's actually returned will depend on which dataloader was originally targeted
    (e.g. the 'train', 'valid', 'test', or 'predict' load), and whether we'll be profiling the
    full dataloader object or just its dataset parser attribute.

    The PyTorch DataLoader and the framework's base data parser classes should have compatible
    interfaces in terms of how to fetch data batches.
    """
    assert "profiler" in config, "missing mandatory 'profiler' (sub)config!"
    target_dataloader_type = config.profiler.get("dataloader_type", "train")
    assert target_dataloader_type in datamodule.dataloader_types, (
        f"invalid target dataloader type: {target_dataloader_type}" f" (should be in {datamodule.dataloader_types})"
    )

    dataloader = datamodule.get_dataloader(target_dataloader_type)
    assert isinstance(
        dataloader, torch.utils.data.DataLoader
    ), f"current data profiler impl does not support this loader type: {type(dataloader)}"

    if not config.profiler.get("use_parser", False):
        return dataloader

    dataparser = dataloader.dataset
    assert isinstance(
        dataparser, ssl4rs.data.DataParser
    ), f"current data profiler impl does not support this parser type: {type(dataparser)}"

    return dataparser


def data_profiler(config: omegaconf.DictConfig) -> None:
    """Runs the data (module, loader, and/or parser) profiling pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    stopwatch_creator = functools.partial(
        ssl4rs.utils.Stopwatch,
        log_level=logging.INFO,
        logger=logger,
    )
    with stopwatch_creator(name="initialization and datamodule creation"):
        (exp_name, run_name, run_type, job_name), datamodule = _common_init(config)
    with stopwatch_creator(name="datamodule.prepare_data()"):
        datamodule.prepare_data()
    with stopwatch_creator(name="datamodule.setup()"):
        datamodule.setup()
    with stopwatch_creator(name="dataloader creation"):
        dataloader = _get_dataloader(config, datamodule)
    max_batch_count = config.profiler.get("batch_count", -1)
    assert max_batch_count is None or max_batch_count == -1 or max_batch_count >= 0
    if max_batch_count is None:
        max_batch_count = -1
    if max_batch_count != 0:
        batch_stopwatch = stopwatch_creator(
            log_message_format="batch{idx:04d} elapsed time: {:0.4f} seconds",
        )
        loop_count = config.profiler.get("loop_count", 1)
        assert loop_count > 0
        loop_times = []
        tot_batch_count = 0
        for loop_idx in range(loop_count):
            with stopwatch_creator(name=f"loop{loop_idx:03d}") as loop_sw:
                batch_stopwatch.start()
                for batch_idx, batch in enumerate(dataloader):
                    curr_elapsed_time = batch_stopwatch.stop()
                    logger.debug(f"batch{batch_idx:04d} elapsed time: {curr_elapsed_time:0.4f} seconds")
                    tot_batch_count += ssl4rs.data.get_batch_size(batch)
                    if max_batch_count != -1 and batch_idx + 1 == max_batch_count:
                        break
                    batch_stopwatch.start()
            loop_times.append(loop_sw.total())
        logger.info(f"loop time min: {np.min(loop_times)}")
        logger.info(f"loop time max: {np.max(loop_times)}")
        logger.info(f"loop time avg: {np.mean(loop_times)}")
        logger.info(f"loop time std: {np.std(loop_times)}")
        tot_loop_time = np.sum(loop_times)
        avg_batch_time = tot_loop_time / tot_batch_count
        logger.info(f"average time per data sample: {avg_batch_time:0.6f} seconds")
    with stopwatch_creator(name="datamodule.teardown()"):
        datamodule.teardown()
    logger.info(f"Done ({exp_name}: {run_name}, '{run_type}', job={job_name})")


# todo: add model inference profiler?
