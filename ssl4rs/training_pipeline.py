import pathlib
import typing

import hydra
import hydra.core.hydra_config
import hydra.types
import lightning.pytorch as pl
import omegaconf

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

    hydra_config = ssl4rs.utils.config.get_hydra_config()
    output_dir = pathlib.Path(hydra_config.runtime.output_dir)
    logger.info(f"Output directory: {output_dir.absolute()}")
    ssl4rs.utils.config.extra_inits(config, output_dir=output_dir)

    logger.info(f"Instantiating datamodule: {config.data.datamodule._target_}")  # noqa
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)

    model = ssl4rs.utils.config.get_model(config)
    callbacks = ssl4rs.utils.config.get_callbacks(config)
    loggers = ssl4rs.utils.config.get_loggers(config)

    logger.info(f"Instantiating trainer: {config.trainer._target_}")  # noqa
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
    resume_from_ckpt_path = None
    if "train" in run_type:
        if config.resume_from_latest_if_possible:
            assert (
                hydra_config.mode == hydra.types.RunMode.RUN
            ), "cannot resume training from a checkpoint in multi-run mode!"
            resume_from_ckpt_path = ssl4rs.utils.config.get_latest_checkpoint(config)
            if resume_from_ckpt_path is not None:
                resume_from_ckpt_path = str(resume_from_ckpt_path)
                logger.info(f"Will resume from 'latest' checkpoint at: {resume_from_ckpt_path}")
        logger.info("Running trainer.fit()...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_from_ckpt_path)
        completed_training = not trainer.interrupted

    target_metric_val: typing.Optional[float] = None
    if not trainer.interrupted:
        has_ckpt_callback = isinstance(trainer.checkpoint_callback, pl.callbacks.ModelCheckpoint)
        if not completed_training or config.trainer.get("fast_dev_run") or not has_ckpt_callback:
            best_ckpt_path = None
        else:
            assert hasattr(trainer.checkpoint_callback, "best_model_path"), "missing callback attrib?"
            best_ckpt_path = trainer.checkpoint_callback.best_model_path  # noqa
            logger.info(f"Best model ckpt at: {best_ckpt_path}")
        target_metric_name = config.get("target_metric")
        if target_metric_name is not None and not config.trainer.get("fast_dev_run"):
            assert model.has_metric(target_metric_name), (
                f"target metric {target_metric_name} for hyperparameter optimization not found! "
                "make sure the `target_metric` field in the config is correct!"
            )
            if best_ckpt_path:
                best_model = model.load_from_checkpoint(best_ckpt_path)
                target_metric_val = best_model.compute_metric(target_metric_name)
                logger.info(f"Best target metric: {target_metric_name}: {target_metric_val}")
        if "test" in run_type:
            logger.info("Running trainer.test()...")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt_path)
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

    logger.info(f"Done ({exp_name}: {run_name}, '{run_type}', job={job_name})")
    return target_metric_val
