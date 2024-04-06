import pathlib

import hydra
import lightning.pytorch as pl
import omegaconf

import ssl4rs

logger = ssl4rs.utils.get_logger(__name__)


def test(config: omegaconf.DictConfig) -> None:
    """Runs the testing pipeline based on a specified model checkpoint path.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    exp_name, run_name, run_type, job_name = config.experiment_name, config.run_name, config.run_type, config.job_name
    logger.info(f"Launching ({exp_name}: {run_name}, '{run_type}', job={job_name})")

    hydra_config = ssl4rs.utils.config.get_hydra_config()
    output_dir = pathlib.Path(hydra_config.runtime.output_dir)
    logger.info(f"Output directory: {output_dir.absolute()}")
    ssl4rs.utils.config.extra_inits(config, output_dir=output_dir)

    logger.info(f"Instantiating datamodule: {config.data.datamodule._target_}")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)

    model = ssl4rs.utils.config.get_model(config)
    callbacks = ssl4rs.utils.config.get_callbacks(config)
    loggers = ssl4rs.utils.config.get_loggers(config)

    logger.info(f"Instantiating trainer: {config.trainer._target_}")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    logger.info("Logging hyperparameters...")
    ssl4rs.utils.logging.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
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
        output_dir=output_dir,
    )

    logger.info(f"Done ({exp_name}: {run_name}, '{run_type}', job={job_name})")
