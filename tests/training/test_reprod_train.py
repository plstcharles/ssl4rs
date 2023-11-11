import pathlib

import pandas as pd
import pytest

import ssl4rs.utils.logging
import tests.helpers.module_runner as module_runner

base_cli_args = [
    "python",
    "train.py",
    "experiment=example_mnist_classif_fast",
    "callbacks=[]",  # to remove the cpu monitor that logs its measurements, used by default
    "logger=debug",  # to use the debug logger instead of the default csv logger
    "++trainer.enable_checkpointing=False",  # no need for this here
    "++trainer.enable_progress_bar=False",  # no need for this here
    # these settings will make the run fast-enough
    "++trainer.max_steps=20",
    "++trainer.max_epochs=2",
    "++trainer.limit_train_batches=10",
    "++trainer.limit_val_batches=10",
    "++trainer.limit_test_batches=10",
    # and these *should* make it reproducible
    "++trainer.benchmark=False",
    "++trainer.deterministic=True",
    "resume_from_latest_if_possible=False",
    "utils.seed=42",
    "utils.seed_workers=True",
    "utils.use_deterministic_algorithms=True",
]


@pytest.mark.slow
def test_reprod_with_2nd_run(tmpdir):
    """Checks that training sessions can be fully reproduced under a 2nd run configuration."""
    cli_args = base_cli_args + [f"utils.output_root_dir='{tmpdir}'"]
    output = module_runner.run(cli_args + ["run_name=_pytest_run_A"])
    if output.returncode != 0:
        pytest.fail(output.stderr)
    old_out_dir = pathlib.Path(tmpdir) / "runs" / "mnist_with_micro_mlp" / "_pytest_run_A"
    assert old_out_dir.is_dir()
    old_logs_dir = list(old_out_dir.glob("debug-logs-*"))[0]
    old_metrics = ssl4rs.utils.logging.DebugLogger.parse_metric_logs(old_logs_dir)
    output = module_runner.run(cli_args + ["run_name=_pytest_run_B"])
    if output.returncode != 0:
        pytest.fail(output.stderr)
    new_out_dir = pathlib.Path(tmpdir) / "runs" / "mnist_with_micro_mlp" / "_pytest_run_B"
    assert new_out_dir.is_dir()
    new_logs_dir = list(new_out_dir.glob("debug-logs-*"))[0]
    new_metrics = ssl4rs.utils.logging.DebugLogger.parse_metric_logs(new_logs_dir)
    assert old_logs_dir != new_logs_dir  # although the two paths are NOT the same...
    # ...all metrics & steps & everything should be 100% equal, up to timestamps & column order
    new_metrics = new_metrics.set_index("step")
    old_metrics = old_metrics.set_index("step")
    pd.testing.assert_frame_equal(old_metrics, new_metrics, check_like=True)
