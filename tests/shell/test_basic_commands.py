import os

import pytest

import tests.helpers.module_runner as module_runner
import tests.helpers.runif


def test_help():
    """Test just executing the train script to get the help message."""
    command = ["python", "train.py", "--help"]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    assert "Powered by Hydra" in output.stdout


def test_fast_dev_run(tmpdir):
    """Test running a 'fast dev run' (1x iter/loader, and most trainer stuff disabled)."""
    command = [
        "python",
        "train.py",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_fast_dev_run",
        "++trainer.fast_dev_run=true",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(tmpdir, "runs", "mnist_with_micro_mlp", "_pytest_fast_dev_run")
    assert os.path.isdir(expected_out_dir)
    expected_console_log = os.path.join(expected_out_dir, "console.log")
    assert os.path.isfile(expected_console_log)


def test_basic_run(tmpdir):
    """Test running a 'regular' (but still fast) run with the CSV logger."""
    command = [
        "python",
        "train.py",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_basic_run",
        "logger=csv",
        "++trainer.max_epochs=2",
        "++trainer.limit_train_batches=5",
        "++trainer.limit_val_batches=5",
        "++trainer.limit_test_batches=5",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    out_dir = os.path.join(tmpdir, "runs", "mnist_with_micro_mlp", "_pytest_basic_run")
    assert os.path.isdir(out_dir)
    # the mnist example config has the 'classification' callbacks enabled with model checkpointing
    expected_ckpt = os.path.join(out_dir, "ckpts", "last.ckpt")
    assert os.path.isfile(expected_ckpt)
    # and we enabled the csv logger, so we should get that output as well
    expected_csv_log = os.path.join(out_dir, "csv", "metrics.csv")
    assert os.path.isfile(expected_csv_log)
    # note: contents will be validated in 'loggers' utests


@pytest.mark.slow
def test_debug(tmpdir):
    """Test running 1 epoch on CPU."""
    command = [
        "python",
        "train.py",
        "debug=default",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_debug",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "debug",
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_debug",
    )
    assert os.path.isdir(expected_out_dir)
    expected_console_log = os.path.join(expected_out_dir, "console.log")
    assert os.path.isfile(expected_console_log)
    expected_csv_log = os.path.join(expected_out_dir, "csv", "metrics.csv")
    assert os.path.isfile(expected_csv_log)
    expected_ckpt = os.path.join(expected_out_dir, "ckpts", "last.ckpt")
    assert os.path.isfile(expected_ckpt)


@tests.helpers.runif.RunIf(min_gpus=1)
@pytest.mark.slow
def test_debug_gpu(tmpdir):
    """Test running 1 epoch on GPU."""
    command = [
        "python",
        "train.py",
        "debug=default",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_debug_gpu",
        "trainer.accelerator=gpu",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "debug",
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_debug_gpu",
    )
    assert os.path.isdir(expected_out_dir)
    expected_console_log = os.path.join(expected_out_dir, "console.log")
    assert os.path.isfile(expected_console_log)
    expected_csv_log = os.path.join(expected_out_dir, "csv", "metrics.csv")
    assert os.path.isfile(expected_csv_log)
    expected_ckpt = os.path.join(expected_out_dir, "ckpts", "last.ckpt")
    assert os.path.isfile(expected_ckpt)


@tests.helpers.runif.RunIf(min_gpus=1)
@pytest.mark.slow
def test_debug_gpu_halfprec(tmpdir):
    """Test running 1 epoch on GPU with half (16-bit) float precision."""
    command = [
        "python",
        "train.py",
        "debug=default",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_debug_gpu_halfprec",
        "trainer.accelerator=gpu",
        "trainer.precision=16",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "debug",
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_debug_gpu_halfprec",
    )
    assert os.path.isdir(expected_out_dir)
    expected_console_log = os.path.join(expected_out_dir, "console.log")
    assert os.path.isfile(expected_console_log)
    expected_csv_log = os.path.join(expected_out_dir, "csv", "metrics.csv")
    assert os.path.isfile(expected_csv_log)
    expected_ckpt = os.path.join(expected_out_dir, "ckpts", "last.ckpt")
    assert os.path.isfile(expected_ckpt)
