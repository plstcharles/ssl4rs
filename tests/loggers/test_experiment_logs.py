import glob
import os

import pandas as pd
import pytest
import yaml

import tests.helpers.module_runner as module_runner


def _launch_experiment_and_return_out_dir(tmpdir, run_suffix):
    command = [
        "train.py",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmpdir}'",
        f"run_name=_pytest_{run_suffix}",
        f"logger={run_suffix}",
        "++trainer.max_epochs=2",
        "++trainer.limit_train_batches=5",
        "++trainer.limit_val_batches=5",
        "++trainer.limit_test_batches=5",
    ]
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
    expected_out_dir = os.path.join(
        tmpdir,
        "runs",
        "mnist_with_micro_mlp",
        f"_pytest_{run_suffix}",
    )
    assert os.path.isdir(expected_out_dir)
    return expected_out_dir


def test_tboard_and_csv(tmpdir):
    """Test tboard+csv logger outputs after running 2 epoch on CPU with the fast config."""
    out_dir = _launch_experiment_and_return_out_dir(tmpdir, run_suffix="tboard_and_csv")
    expected_ckpt = os.path.join(out_dir, "ckpts", "last.ckpt")
    assert os.path.isfile(expected_ckpt)
    ckpt_paths = os.listdir(os.path.join(out_dir, "ckpts"))
    assert len(ckpt_paths) == 2

    expected_csv_log = os.path.join(out_dir, "csv", "metrics.csv")
    assert os.path.isfile(expected_csv_log)
    csv_data = pd.read_csv(expected_csv_log)
    assert all([metric in csv_data.columns for metric in ["train/accuracy", "valid/accuracy", "test/accuracy"]])
    assert csv_data["step"].max() == 10  # 10th step with 0-based = test time
    assert csv_data["epoch"].max() == 1  # 0-based
    assert csv_data[~csv_data["train/accuracy"].isna()]["step"].max() == 9
    assert csv_data[~csv_data["valid/accuracy"].isna()]["step"].max() == 9
    assert csv_data[~csv_data["test/accuracy"].isna()]["step"].max() == 10

    expected_config_logs = glob.glob(os.path.join(out_dir, "config.*.log"))
    assert len(expected_config_logs) == 1
    expected_config_log = expected_config_logs[0]
    with open(expected_config_log) as fd:
        config = yaml.safe_load(fd)
    run_and_job_name = config["utils"]["run_and_job_name"]

    expected_tboard_dir = os.path.join(tmpdir, "tensorboard")
    assert os.path.isdir(expected_tboard_dir)
    expected_tboard_exp_dir = os.path.join(expected_tboard_dir, "mnist_with_micro_mlp")
    assert os.path.isdir(expected_tboard_exp_dir)
    expected_tboard_run_dir = os.path.join(expected_tboard_exp_dir, run_and_job_name)
    assert os.path.isdir(expected_tboard_run_dir)
    tboard_files = os.listdir(expected_tboard_run_dir)
    assert any([file_name.startswith("events.out") for file_name in tboard_files])


@pytest.mark.slow
def test_mlflow(tmpdir):
    """Test mlflow logger outputs after running 2 epoch on CPU with the fast config."""
    out_dir = _launch_experiment_and_return_out_dir(tmpdir, run_suffix="mlflow")

    unexpected_mlflow_dir = os.path.join(out_dir, "mlruns")
    assert not os.path.isdir(unexpected_mlflow_dir)
    expected_mlflow_dir = os.path.join(tmpdir, "mlruns")
    assert os.path.isdir(expected_mlflow_dir)

    ignored_mflow_dirs = ["0", ".trash"]  # not sure what '0' is... but it's always there
    mlflow_exp_dirs = [
        os.path.join(expected_mlflow_dir, dir)
        for dir in os.listdir(expected_mlflow_dir)
        if os.path.isdir(os.path.join(expected_mlflow_dir, dir)) and dir not in ignored_mflow_dirs
    ]
    assert len(mlflow_exp_dirs) == 1
    mlflow_exp_dir = mlflow_exp_dirs[0]
    assert os.path.isfile(os.path.join(mlflow_exp_dir, "meta.yaml"))
    mlflow_run_dirs = [dir for dir in os.listdir(mlflow_exp_dir) if os.path.isdir(os.path.join(mlflow_exp_dir, dir))]
    # we only launched one run in the tmpdir, so there should only be one subdirectory
    assert len(mlflow_run_dirs) == 1
    mlflow_run_dir = os.path.join(mlflow_exp_dir, mlflow_run_dirs[0])
    assert os.path.isfile(os.path.join(mlflow_run_dir, "meta.yaml"))
    assert os.path.isdir(os.path.join(mlflow_run_dir, "artifacts"))
    assert os.path.isdir(os.path.join(mlflow_run_dir, "metrics"))
    assert os.path.isdir(os.path.join(mlflow_run_dir, "params"))
    assert os.path.isdir(os.path.join(mlflow_run_dir, "tags"))
    assert os.path.isfile(os.path.join(mlflow_run_dir, "metrics", "train", "accuracy"))
    assert os.path.isfile(os.path.join(mlflow_run_dir, "metrics", "epoch"))

    # re-launch a 2nd run; it should be located in the same directory in the output folder
    out_dir2 = _launch_experiment_and_return_out_dir(tmpdir, run_suffix="mlflow")
    assert out_dir2 == out_dir
    expected_config_logs = glob.glob(os.path.join(out_dir, "config.*.log"))
    assert len(expected_config_logs) == 2
    # ...but it should have a 2nd run directory in the mlflow folder
    mlflow_exp_dirs2 = [
        os.path.join(expected_mlflow_dir, dir)
        for dir in os.listdir(expected_mlflow_dir)
        if os.path.isdir(os.path.join(expected_mlflow_dir, dir)) and dir not in ignored_mflow_dirs
    ]
    assert len(mlflow_exp_dirs2) == 1
    assert mlflow_exp_dirs2[0] == mlflow_exp_dir
    mlflow_run_dirs2 = [dir for dir in os.listdir(mlflow_exp_dir) if os.path.isdir(os.path.join(mlflow_exp_dir, dir))]
    assert len(mlflow_run_dirs2) == 2
