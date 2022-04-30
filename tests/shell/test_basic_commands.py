import os

import pytest

import tests.helpers.module_runner as module_runner
from tests.helpers.runif import RunIf


def test_help():
    """Test just executing the train script to get the help message."""
    command = ["train.py", "--help"]
    output = module_runner.run(command)
    assert output.returncode == 0 and "Powered by Hydra" in output.stdout


def test_fast_dev_run(tmpdir):
    """Test running for 1 train, val and test batch."""
    command = [
        "train.py",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_fast_dev_run",
        "++trainer.fast_dev_run=true",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "runs",
        "mnist_with_micro_mlp",
        "_pytest_fast_dev_run",
    )
    assert os.path.isdir(expected_out_dir)
    expected_console_log = os.path.join(expected_out_dir, "console.log")
    assert os.path.isfile(expected_console_log)


@pytest.mark.slow
def test_debug(tmpdir):
    """Test running 1 epoch on CPU."""
    command = [
        "train.py",
        "trainer=debug",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        "run_name=_pytest_debug",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
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


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_debug_gpu(tmpdir):
    """Test running 1 epoch on GPU."""
    command = [
        "train.py",
        "trainer=debug",
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


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_debug_gpu_halfprec(tmpdir):
    """Test running 1 epoch on GPU with half (16-bit) float precision."""
    command = [
        "train.py",
        "trainer=debug",
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
