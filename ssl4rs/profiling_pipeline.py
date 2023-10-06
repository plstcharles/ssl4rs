import copy
import functools
import logging
import pathlib
import typing

import cv2 as cv
import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import torch.utils.data
import tqdm.rich

import ssl4rs

logger = ssl4rs.utils.get_logger(__name__)
stopwatch_creator = functools.partial(ssl4rs.utils.Stopwatch, log_level=logging.INFO, logger=logger)


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

    logger.info(f"Instantiating datamodule: {config.data.datamodule._target_}")  # noqa
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)
    assert isinstance(
        datamodule, ssl4rs.data.DataModule
    ), "invalid datamodule base class (need `ssl4rs.data.DataModule` for getters below!)"

    return (exp_name, run_name, run_type, job_name), datamodule


def _get_dataloader(
    datamodule: ssl4rs.data.DataModule,
    target_dataloader_type: typing.AnyStr,
    return_parser: bool,
) -> typing.Union[torch.utils.data.DataLoader, ssl4rs.data.DataParser]:
    """Returns the dataloader object we'll be profiling.

    The object that's actually returned will depend on which dataloader was originally targeted
    (e.g. the 'train', 'valid', 'test', or 'predict' load), and whether we'll be profiling the full
    dataloader object or just its dataset parser attribute.

    The PyTorch DataLoader and the framework's base data parser classes should have compatible
    interfaces in terms of how to fetch data batches.
    """
    avail_dataloader_types = datamodule.dataloader_types
    assert (
        target_dataloader_type in avail_dataloader_types
    ), f"invalid dataloader type: {target_dataloader_type}\n\t(should be in {avail_dataloader_types})"

    dataloader = datamodule.get_dataloader(target_dataloader_type)
    assert isinstance(
        dataloader, torch.utils.data.DataLoader
    ), f"current data profiler impl does not support this loader type: {type(dataloader)}"

    if not return_parser:
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
    assert "profiler" in config, "missing mandatory 'profiler' (sub)config!"
    with stopwatch_creator(name="initialization and datamodule creation"):
        (exp_name, run_name, run_type, job_name), datamodule = _common_init(config)
    with stopwatch_creator(name="datamodule.prepare_data()"):
        datamodule.prepare_data()
    with stopwatch_creator(name="datamodule.setup()"):
        datamodule.setup()
    with stopwatch_creator(name="dataloader creation"):
        dataloader = _get_dataloader(
            datamodule=datamodule,
            target_dataloader_type=config.profiler.get("default_dataloader_type", "train"),
            return_parser=config.profiler.get("use_parser", False),
        )
    max_batch_count = config.profiler.get("batch_count", -1)
    assert max_batch_count is None or max_batch_count == -1 or max_batch_count >= 0
    if max_batch_count is None:
        max_batch_count = -1
    display_key = config.profiler.get("display_key", None)
    display_wait_time = config.profiler.get("display_wait_time", 0)  # in milliseconds
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
                if max_batch_count == -1:
                    max_iters = len(dataloader)
                else:
                    max_iters = min(len(dataloader), max_batch_count)
                pbar = tqdm.rich.tqdm(
                    desc=f"loop{loop_idx:03d}",
                    total=max_iters,
                )
                batch_stopwatch.start()
                for batch_idx, batch in enumerate(dataloader):
                    curr_elapsed_time, _ = batch_stopwatch.stop(), loop_sw.stop()
                    logger.debug(f"batch{batch_idx:04d} elapsed time: {curr_elapsed_time:0.4f} seconds")
                    pbar.update(1)
                    tot_batch_count += ssl4rs.data.get_batch_size(batch)
                    if display_key:
                        _display_batched_tensor(batch, display_key, display_wait_time)
                    if max_batch_count != -1 and batch_idx + 1 == max_batch_count:
                        break
                    batch_stopwatch.start(), loop_sw.start()
                pbar.close()
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


def _display_batched_tensor(
    batch: ssl4rs.data.BatchDictType,
    display_key: typing.AnyStr,
    display_wait_time: int = 0,  # in msec
) -> None:
    """Renders and displays the images given via a batched tensor using opencv."""
    batch_size = ssl4rs.data.get_batch_size(batch)
    if batch_size == 0:
        return
    assert display_key is not None and display_key in batch, f"missing tensor key: {display_key}"
    display_tensor = batch[display_key]
    if len(display_tensor) == batch_size:
        display_images = [ssl4rs.utils.drawing.get_displayable_image(t) for t in display_tensor]
    else:
        assert batch_size == 1, "mismatch between tensor dim0 length & batch size for auto-display"
        display_images = [ssl4rs.utils.drawing.get_displayable_image(display_tensor)]
    for img in display_images:
        cv.imshow(f"{display_key}", img)
        cv.waitKey(display_wait_time)


def _get_trainer_override_settings(max_batch_count: int) -> omegaconf.DictConfig:
    """Returns the set of overriding trainer settings used for model train/valid profiling."""
    return omegaconf.OmegaConf.create(
        {
            "max_epochs": 1,
            "limit_train_batches": max_batch_count if max_batch_count > 0 else None,
            "limit_val_batches": max_batch_count if max_batch_count > 0 else None,
            "num_sanity_val_steps": 0,
            "barebones": True,
        }
    )


def model_profiler(config: omegaconf.DictConfig) -> None:
    """Runs the model (1x training + validation epoch) profiling pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    assert "profiler" in config, "missing mandatory 'profiler' (sub)config!"
    with stopwatch_creator(name="initialization and datamodule creation"):
        (exp_name, run_name, run_type, job_name), datamodule = _common_init(config)
        datamodule.prepare_data()
        datamodule.setup()
        train_dataloader = _get_dataloader(
            datamodule=datamodule,
            target_dataloader_type=config.profiler.get("default_dataloader_type", "train"),
            return_parser=config.profiler.get("use_parser", False),
        )
        valid_dataloader = _get_dataloader(
            datamodule=datamodule,
            target_dataloader_type=config.profiler.get("valid_dataloader_type", "train"),
            return_parser=config.profiler.get("use_parser", False),
        )
    with stopwatch_creator(name="model and trainer creation"):
        model = ssl4rs.utils.config.get_model(config)
    max_batch_count = config.profiler.get("batch_count", -1)
    assert max_batch_count is None or max_batch_count == -1 or max_batch_count >= 0
    if max_batch_count is None:
        max_batch_count = -1
    trainer_config = copy.deepcopy(config.trainer)
    trainer_override_config = _get_trainer_override_settings(max_batch_count)
    for key, val in trainer_override_config.items():
        if key in trainer_config:
            omegaconf.OmegaConf.update(cfg=trainer_config, key=key, value=val, merge=False)
        else:
            with omegaconf.open_dict(trainer_config):
                trainer_config[key] = val
    logger.info("Final trainer settings for model profiling:")
    for key, val in trainer_config.items():
        logger.info(f"\t{key}: {val}")
    if max_batch_count != 0:
        loop_count = config.profiler.get("loop_count", 1)
        assert loop_count > 0
        loop_times = []
        for loop_idx in range(loop_count):
            trainer: pl.Trainer = hydra.utils.instantiate(trainer_config)
            with stopwatch_creator(name=f"loop{loop_idx:03d}") as loop_sw:
                trainer.fit(
                    model=model,  # noqa
                    train_dataloaders=train_dataloader,
                    val_dataloaders=valid_dataloader,
                )
            loop_times.append(loop_sw.total())
        logger.info(f"epoch time min: {np.min(loop_times)}")
        logger.info(f"epoch time max: {np.max(loop_times)}")
        logger.info(f"epoch time avg: {np.mean(loop_times)}")
        logger.info(f"epoch time std: {np.std(loop_times)}")
    datamodule.teardown()
    logger.info(f"Done ({exp_name}: {run_name}, '{run_type}', job={job_name})")
