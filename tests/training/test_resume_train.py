import os

import pandas as pd
import pytest

import tests.helpers.module_runner as module_runner


@pytest.mark.slow
def test_resume_after_completion(tmpdir):
    """Test resuming a training session after it was actually completed."""
    command = [
        "python",
        "train.py",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_resume_after_completion",
        "++trainer.max_steps=10",
        "++trainer.max_epochs=2",
        "++trainer.limit_train_batches=5",
        "++trainer.limit_val_batches=5",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_resume_after_completion",
    )
    assert os.path.isdir(expected_out_dir)
    expected_csv_log = os.path.join(expected_out_dir, "csv", "metrics.csv")
    assert os.path.isfile(expected_csv_log)
    old_csv_data = pd.read_csv(expected_csv_log)
    assert "test/accuracy" in old_csv_data.columns and "step" in old_csv_data.columns
    # just to convince ourselves we won't be reading the SAME, un-updated file, let's delete it
    os.remove(expected_csv_log)
    assert not os.path.isfile(expected_csv_log)
    # 2nd run with same args, plus the resume token:
    command.append("++resume_from_latest_if_possible=True")
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    new_csv_data = pd.read_csv(expected_csv_log)
    # should still have the same test metrics, and no new steps recorded beyond the last one
    new_max_step = new_csv_data["step"].max()
    old_max_step = old_csv_data["step"].max()
    assert new_max_step == old_max_step
    new_final_output = new_csv_data[new_csv_data["step"] == new_max_step]
    old_final_output = old_csv_data[old_csv_data["step"] == old_max_step]
    assert len(new_final_output) == 1 and len(old_final_output) == 1
    assert new_final_output["test/accuracy"].iloc[0] == old_final_output["test/accuracy"].iloc[0]


@pytest.mark.slow
def test_resume_after_interruption(tmpdir):
    """Test resuming a training session after it was interrupted mid-epoch."""
    command = [
        "python",
        "train.py",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_resume_after_interruption",
        "++trainer.max_steps=8",
        "++trainer.max_epochs=5",
        "++trainer.limit_train_batches=5",
        "++trainer.limit_val_batches=5",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_resume_after_interruption",
    )
    assert os.path.isdir(expected_out_dir)
    with open(os.path.join(expected_out_dir, "console.log")) as fd:
        old_console_log = fd.read()
    assert "Will resume from 'latest' checkpoint" not in old_console_log
    expected_csv_log = os.path.join(expected_out_dir, "csv", "metrics.csv")
    assert os.path.isfile(expected_csv_log)
    old_csv_data = pd.read_csv(expected_csv_log)
    # 2nd run with same args, plus the resume token and increased step limits:
    command.append("++resume_from_latest_if_possible=True")
    command.append("++trainer.max_steps=12")  # should get to 3rd epoch
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    with open(os.path.join(expected_out_dir, "console.log")) as fd:
        new_console_log = fd.read()
    assert len(old_console_log) < len(new_console_log)
    assert new_console_log.startswith(old_console_log)
    assert "Will resume from 'latest' checkpoint" in new_console_log
    new_csv_data = pd.read_csv(expected_csv_log)
    assert old_csv_data["step"].max() == new_csv_data["step"].min()
    assert new_csv_data["step"].max() == 12
