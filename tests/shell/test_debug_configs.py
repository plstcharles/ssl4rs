import typing

import pytest

import tests.helpers.module_runner as module_runner


def _get_base_command(tmpdir, test_name) -> typing.List[typing.AnyStr]:
    return [
        "train.py",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        f"run_name=_pytest_debug_{test_name}",
    ]


@pytest.mark.slow
def test_debug_fast_dev_run(tmpdir):
    command = _get_base_command(tmpdir, "fast_dev_run")
    command.append("debug=fast_dev_run")
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)


@pytest.mark.slow
def test_debug_limit_batches(tmpdir):
    command = _get_base_command(tmpdir, "limit_batches")
    command.append("debug=limit_batches")
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)


@pytest.mark.slow
def test_debug_overfit(tmpdir):
    command = _get_base_command(tmpdir, "overfit")
    command.extend([
        "debug=overfit",
        "trainer.limit_val_batches=1",
        "trainer.limit_test_batches=1",
    ])
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)


@pytest.mark.slow
def test_debug_pl_profiler(tmpdir):
    command = _get_base_command(tmpdir, "pl_profiler")
    command.extend([
        "debug=pl_profiler",
        "trainer.limit_train_batches=3",
        "trainer.limit_val_batches=1",
        "trainer.limit_test_batches=1",
    ])
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
