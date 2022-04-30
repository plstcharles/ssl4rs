import pathlib
import typing

import hydra
import hydra.core.hydra_config
import omegaconf
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_log

import ssl4rs

logger = ssl4rs.utils.get_logger(__name__)


def test(config: omegaconf.DictConfig) -> None:
    """Runs the testing pipeline based on a specified model checkpoint path.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    exp_name, run_name, run_type, job_name = config.experiment_name, config.run_name, config.run_type, config.job_name
    logger.info(f"Launching ({exp_name}: {run_name}, '{run_type}', job={job_name})")

    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir.absolute()}")
    ssl4rs.utils.config.extra_inits(config, output_dir=output_dir)

    logger.info(f"Instantiating datamodule: {config.data.datamodule._target_}")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)

    logger.info(f"Instantiating model: {config.model._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    loggers: typing.List[pl_log.Logger] = []
    if "logger" in config:
        for lg_name, lg_config in config.logger.items():
            logger.info(f"Instantiating '{lg_name}' logger: {lg_config._target_}")
            loggers.append(hydra.utils.instantiate(lg_config))

    logger.info(f"Instantiating trainer: {config.trainer._target_}")
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, logger=loggers)

    logger.info("Logging hyperparameters...")
    ssl4rs.utils.logging.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=[],
        loggers=loggers,
    )

    logger.info("Running trainer.test()...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)

    if hasattr(model, "compute_metrics") and callable(model.compute_metrics):
        metrics = model.compute_metrics(loop_type="test")
        for metric_name, metric_val in metrics.items():
            logger.info(f"{metric_name}: {metric_val}")

    logger.info("Finalizing logs...")
    ssl4rs.utils.logging.finalize_logs(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=[],
        loggers=loggers,
    )

    logger.info(f"Done ({exp_name}: {run_name}, '{run_type}', job={job_name})")
