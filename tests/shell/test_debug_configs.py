import glob
import os
import typing

import pandas as pd
import pytest
import yaml

import tests.helpers.module_runner as module_runner


def _get_base_command(tmpdir, test_name) -> typing.List[typing.AnyStr]:
    return [
        "python",
        "train.py",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        f"run_name=_pytest_debug_{test_name}",
    ]


@pytest.mark.slow
def test_debug_fast_dev_run(tmpdir):
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#fast-dev-run
    command = _get_base_command(tmpdir, "fast_dev_run")
    command.extend(["debug=fast_dev_run", "logger=csv"])
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "debug",
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_debug_fast_dev_run",
    )
    assert os.path.isdir(expected_out_dir)
    expected_console_log = os.path.join(expected_out_dir, "console.log")
    assert os.path.isfile(expected_console_log)
    # there should not be a CSV logger dir, as fast dev runs disable logging
    expected_csv_log = os.path.join(expected_out_dir, "csv", "metrics.csv")
    assert not os.path.isfile(expected_csv_log)


@pytest.mark.slow
def test_debug_limit_batches(tmpdir):
    command = _get_base_command(tmpdir, "limit_batches")
    command.append("debug=limit_batches")
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "debug",
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_debug_limit_batches",
    )
    expected_config_logs = glob.glob(os.path.join(expected_out_dir, "config.*.log"))
    assert len(expected_config_logs) == 1
    expected_config = expected_config_logs[0]
    with open(expected_config) as fd:
        config = yaml.safe_load(fd)
    expected_step_count = config["trainer"]["max_epochs"] * config["trainer"]["limit_train_batches"]
    # in this case, there should be a CSV logger dir, as we did not disable logging
    expected_csv_log = os.path.join(expected_out_dir, "csv", "metrics.csv")
    assert os.path.isfile(expected_csv_log)
    csv_data = pd.read_csv(expected_csv_log)
    # we'll just make sure that the last event (at the last step) is indeed at the expected max step
    assert csv_data["step"].max() == expected_step_count


@pytest.mark.slow
def test_debug_overfit(tmpdir):
    command = _get_base_command(tmpdir, "overfit")
    command.extend(
        [
            "debug=overfit",
            "trainer.limit_val_batches=1",
            "trainer.limit_test_batches=1",
        ]
    )
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "debug",
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_debug_overfit",
    )
    expected_config_logs = glob.glob(os.path.join(expected_out_dir, "config.*.log"))
    assert len(expected_config_logs) == 1
    with open(expected_config_logs[0]) as fd:
        config = yaml.safe_load(fd)
    expected_step_count = config["trainer"]["max_epochs"] * config["trainer"]["overfit_batches"]
    # in this case, there should be a CSV logger dir, as we did not disable logging
    expected_csv_log = os.path.join(expected_out_dir, "csv", "metrics.csv")
    assert os.path.isfile(expected_csv_log)
    csv_data = pd.read_csv(expected_csv_log)
    # we'll just make sure that the last event (at the last step) is indeed at the expected max step
    assert csv_data["step"].max() == expected_step_count


@pytest.mark.slow
def test_debug_pl_profiler(tmpdir):
    command = _get_base_command(tmpdir, "pl_profiler")
    command.extend(
        [
            "debug=pl_profiler",
            "trainer.limit_train_batches=3",
            "trainer.limit_val_batches=1",
            "trainer.limit_test_batches=1",
        ]
    )
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "debug",
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_debug_pl_profiler",
    )
    with open(os.path.join(expected_out_dir, "console.log")) as fd:
        console_log = fd.read()
    assert "Profiler Report" in console_log
