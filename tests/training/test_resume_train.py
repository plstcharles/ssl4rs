import os
import pathlib

import pytest

import ssl4rs.utils.logging
import tests.helpers.module_runner as module_runner


@pytest.mark.slow
def test_resume_after_completion(tmpdir):
    """Test resuming a training session after it was actually completed."""
    command = [
        "python",
        "train.py",
        "experiment=example_mnist_classif_fast",
        "callbacks=[]",  # to remove the cpu monitor that logs its measurements, used by default
        "logger=debug",  # to use the debug logger instead of the default csv logger
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_resume_after_completion",
        "++trainer.max_steps=10",
        "++trainer.max_epochs=2",
        "++trainer.limit_train_batches=5",
        "++trainer.limit_val_batches=5",
        "++trainer.limit_test_batches=5",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    exp_dir = pathlib.Path(tmpdir) / "runs" / "mnist_with_micro_mlp"
    expected_out_dir = exp_dir / "_pytest_resume_after_completion"
    assert expected_out_dir.is_dir()
    logs_dirs = list(expected_out_dir.glob("debug-logs-*"))
    assert len(logs_dirs) == 1
    old_logs_dir = list(expected_out_dir.glob("debug-logs-*"))[0]
    old_metrics = ssl4rs.utils.logging.DebugLogger.parse_metric_logs(old_logs_dir)
    assert "test/accuracy" in old_metrics.columns and "step" in old_metrics.columns
    # 2nd run with same args, plus the resume token:
    command.append("resume_from_latest_if_possible=True")
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    logs_dirs = list(expected_out_dir.glob("debug-logs-*"))
    assert len(logs_dirs) == 2
    new_logs_dir = next(d for d in logs_dirs if d != old_logs_dir)
    new_metrics = ssl4rs.utils.logging.DebugLogger.parse_metric_logs(new_logs_dir)
    # should still have the same test metrics, and no new steps recorded beyond the last one
    new_max_step = new_metrics["step"].max()
    old_max_step = old_metrics["step"].max()
    assert new_max_step == old_max_step
    new_final_output = new_metrics[new_metrics["step"] == new_max_step]
    old_final_output = old_metrics[old_metrics["step"] == old_max_step]
    assert len(new_final_output) == 1 and len(old_final_output) == 1
    assert new_final_output["test/accuracy"].iloc[0] == old_final_output["test/accuracy"].iloc[0]


@pytest.mark.slow
def test_resume_after_interruption(tmpdir):
    """Test resuming a training session after it was interrupted mid-epoch."""
    command = [
        "python",
        "train.py",
        "experiment=example_mnist_classif_fast",
        "logger=debug",  # to use the debug logger instead of the default csv logger
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_resume_after_interruption",
        "++trainer.max_steps=8",
        "++trainer.max_epochs=5",
        "++trainer.limit_train_batches=5",
        "++trainer.limit_val_batches=5",
        "++trainer.limit_test_batches=5",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    exp_dir = pathlib.Path(tmpdir) / "runs" / "mnist_with_micro_mlp"
    expected_out_dir = exp_dir / "_pytest_resume_after_interruption"
    assert expected_out_dir.is_dir()
    with open(os.path.join(expected_out_dir, "console.log")) as fd:
        old_console_log = fd.read()
    assert "Will resume from 'latest' checkpoint" not in old_console_log
    logs_dirs = list(expected_out_dir.glob("debug-logs-*"))
    assert len(logs_dirs) == 1
    old_logs_dir = list(expected_out_dir.glob("debug-logs-*"))[0]
    old_metrics = ssl4rs.utils.logging.DebugLogger.parse_metric_logs(old_logs_dir)
    # 2nd run with same args, plus the resume token and increased step limits:
    command.append("resume_from_latest_if_possible=True")
    command.append("++trainer.max_steps=12")  # should get to 3rd epoch
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    with open(os.path.join(expected_out_dir, "console.log")) as fd:
        new_console_log = fd.read()
    assert len(old_console_log) < len(new_console_log)
    assert new_console_log.startswith(old_console_log)
    assert "Will resume from 'latest' checkpoint" in new_console_log
    logs_dirs = list(expected_out_dir.glob("debug-logs-*"))
    assert len(logs_dirs) == 2
    new_logs_dir = next(d for d in logs_dirs if d != old_logs_dir)
    new_metrics = ssl4rs.utils.logging.DebugLogger.parse_metric_logs(new_logs_dir)
    assert old_metrics["step"].max() == new_metrics["step"].min()
    assert new_metrics["step"].max() == 12
