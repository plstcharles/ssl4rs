import os

import pandas as pd
import pytest

import tests.helpers.module_runner as module_runner

base_cli_args = [
    "python",
    "train.py",
    "experiment=example_mnist_classif_fast",
    # these settings will make the run fast-enough
    "++trainer.max_steps=20",
    "++trainer.max_epochs=2",
    "++trainer.limit_train_batches=10",
    "++trainer.limit_val_batches=10",
    # and these *should* make it reproducible
    "++seed=42",
    "++seed_workers=True",
    "++trainer.benchmark=False",
    "++trainer.deterministic=True",
]


import lightning.pytorch.utilities.seed


@pytest.mark.slow
def test_reprod_with_2nd_run(tmpdir):
    """Checks that training sessions can be fully reproduced under a 2nd run configuration."""
    cli_args = base_cli_args + [f"utils.output_root_dir='{tmpdir}'"]
    output = module_runner.run(cli_args + ["run_name=_pytest_run_A"])
    if output.returncode != 0:
        pytest.fail(output.stderr)
    old_out_dir = os.path.join(tmpdir, "runs", "mnist_with_micro_mlp", "_pytest_run_A")
    old_csv_path = os.path.join(old_out_dir, "csv", "metrics.csv")
    old_csv_data = pd.read_csv(old_csv_path)
    output = module_runner.run(cli_args + ["run_name=_pytest_run_B"])
    if output.returncode != 0:
        pytest.fail(output.stderr)
    new_out_dir = os.path.join(tmpdir, "runs", "mnist_with_micro_mlp", "_pytest_run_B")
    new_csv_path = os.path.join(new_out_dir, "csv", "metrics.csv")
    new_csv_data = pd.read_csv(new_csv_path)
    assert old_csv_path != new_csv_path  # although the two paths are NOT the same...
    assert old_csv_data.equals(new_csv_data)  # ...all metrics & steps & everything should be 100% equal!
