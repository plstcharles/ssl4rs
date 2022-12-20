import pathlib
import typing

import hydra
import hydra.core.hydra_config
import omegaconf
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_log

import ssl4rs

logger = ssl4rs.utils.get_logger(__name__)


def train(config: omegaconf.DictConfig) -> typing.Optional[float]:
    """Runs the training pipeline, and possibly tests the model as well following that.

    If testing is enabled, the 'best' model weights found during training will be reloaded
    automatically inside this function.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Value obtained for the targeted metric (for hyperparam optimization).
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

    callbacks: typing.List[pl.Callback] = []
    if "callbacks" in config:
        for cb_name, cb_config in config.callbacks.items():
            logger.info(f"Instantiating '{cb_name}' callback: {cb_config._target_}")
            callbacks.append(hydra.utils.instantiate(cb_config))

    loggers: typing.List[pl_log.LightningLoggerBase] = []
    if "logger" in config:
        for lg_name, lg_config in config.logger.items():
            logger.info(f"Instantiating '{lg_name}' logger: {lg_config._target_}")
            loggers.append(hydra.utils.instantiate(lg_config))

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

    completed_training = False
    if "train" in run_type:
        logger.info("Running trainer.fit()...")
        trainer.fit(model=model, datamodule=datamodule)
        completed_training = not trainer.interrupted

    target_metric_val: typing.Optional[float] = None
    if not trainer.interrupted:
        target_metric_name = config.get("target_metric")
        if target_metric_name is not None:
            assert target_metric_name in trainer.callback_metrics, (
                f"target metric {target_metric_name} for hyperparameter optimization not found! "
                "make sure the `target_metric` field in the config is correct!"
            )
            target_metric_val = trainer.callback_metrics.get(target_metric_name)
        if "test" in run_type:
            ckpt_path = "best"
            if not completed_training or config.trainer.get("fast_dev_run"):
                ckpt_path = None
            logger.info("Running trainer.test()...")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            if hasattr(model, "compute_metrics") and callable(model.compute_metrics):
                metrics = model.compute_metrics(loop_type="test")
                for metric_name, metric_val in metrics.items():
                    logger.info(f"best {metric_name}: {metric_val}")

    logger.info("Finalizing logs...")
    ssl4rs.utils.logging.finalize_logs(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        loggers=loggers,
    )

    if completed_training and not config.trainer.get("fast_dev_run"):
        logger.info(f"Best model ckpt at: {trainer.checkpoint_callback.best_model_path}")

    logger.info(f"Done ({exp_name}: {run_name}, '{run_type}', job={job_name})")
    return target_metric_val
