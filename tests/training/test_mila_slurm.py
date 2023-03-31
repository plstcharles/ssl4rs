import pathlib
import time
import typing

import pytest

import ssl4rs.utils.config
import ssl4rs.utils.filesystem
import tests.helpers.module_runner as module_runner
import tests.helpers.runif


def _get_sbatch_script_header(
    job_name,
    out_dir,
    use_gpu=False,
) -> str:
    """Returns the 'header' (top portion) of an sbatch script for the mila cluster."""
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f'#SBATCH --output="{str(out_dir)}/%j.log"',
            f'#SBATCH --error="{str(out_dir)}/%j.log"',
            "#SBATCH --mem-per-cpu=4GB",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --time=00:03:00",  # should be enough...
            "#SBATCH --gres=gpu:1" if use_gpu else "",
            'echo    "Arguments: $@"',
            'echo -n "Date:      "; date',
            'echo    "JobId:     $SLURM_JOBID"',
            'echo    "JobName:   $SLURM_JOB_NAME"',
            'echo    "Node:      $HOSTNAME"',
            'echo    "Nodelist:  $SLURM_JOB_NODELIST"',
            "export PYTHONUNBUFFERED=1",
            "module purge",
        ]
    )


def _get_mila_cluster_tmp_shared_dir() -> pathlib.Path:
    """Returns a temporary directory for logs on the mila cluster."""
    # note: this assumes we've defined the `OUTPUT_ROOT` environment variable somewhere
    output_root_dir = ssl4rs.utils.config.get_output_root_dir()
    test_tmp_shared_dir = output_root_dir / "_pytest_tmp"
    test_tmp_shared_dir.mkdir(parents=True, exist_ok=True)
    return test_tmp_shared_dir


def _launch_sbatch_and_wait_for_completion(
    script_path,
    pool_rate_sec=5,
    timeout_sec=60,
    should_cancel_on_timeout=True,
) -> typing.Union[str, bool]:  # output = job_id, job_finished
    output = module_runner.run(["sbatch", str(script_path)])
    assert output.returncode == 0
    expected_job_id_prefix = "Submitted batch job "
    assert str(output.stdout).startswith(expected_job_id_prefix), f"unexpected sbatch launch output: {output.stdout}"
    job_id = str(output.stdout)[len(expected_job_id_prefix) :].strip()
    start_time = time.time()
    job_finished = False
    while not job_finished and (time.time() - start_time) < timeout_sec:
        time.sleep(pool_rate_sec)
        output = module_runner.run(["squeue", "--job", job_id])
        if output.returncode != 0:
            # if squeue for that job id fails, it's done
            job_finished = True
        elif output.returncode == 0:
            # otherwise, if it returned without the job id in the output, it's done
            job_finished = job_id not in output.stdout
        if job_finished:
            break
    if should_cancel_on_timeout and not job_finished:
        module_runner.run(["scancel", job_id])
    return job_id, job_finished


@tests.helpers.runif.RunIf(only_on_mila_cluster=True)
@pytest.mark.slow
def test_mila_cluster_hello_world():
    tmp_shared_dir = _get_mila_cluster_tmp_shared_dir()
    script_header = _get_sbatch_script_header(
        job_name="ssl4rs-utest-hello-world",
        out_dir=tmp_shared_dir,
        use_gpu=False,
    )
    script_exec = "echo 'Hello world!'\n"
    script_full = script_header + "\n" + script_exec
    script_path = tmp_shared_dir / "hello_world.sh"
    with open(script_path, "w") as fd:
        fd.write(script_full)
    job_id, job_finished = _launch_sbatch_and_wait_for_completion(script_path, timeout_sec=30)
    assert job_finished, f"job {job_id} did not finish before timeout"
    output_log_path = tmp_shared_dir / f"{job_id}.log"
    assert output_log_path.is_file()
    with open(output_log_path) as fd:
        output_log = fd.read()
    assert "Hello world!" in output_log


@tests.helpers.runif.RunIf(only_on_mila_cluster=True)
@pytest.mark.slow
def test_mila_cluster_train_1gpu():
    tmp_shared_dir = _get_mila_cluster_tmp_shared_dir()
    script_header = _get_sbatch_script_header(
        job_name="ssl4rs-utest-train-fast-1gpu",
        out_dir=tmp_shared_dir,
        use_gpu=True,
    )
    command = [
        "srun",
        '"$CONDA_PREFIX/bin/python"',
        "train.py",
        "debug=default",
        "experiment=example_mnist_classif_fast",
        f"utils.output_root_dir='{tmp_shared_dir}'",
        "run_name=_pytest_mila_cluster_train_1gpu",
        "trainer.accelerator=gpu",
        # note that we need to disable colorlog to get easier-to-read logs on the cluster
        "hydra/hydra_logging=default",
        "hydra/job_logging=default",
    ]
    script_full = script_header + "\n" + " ".join(command)
    script_path = tmp_shared_dir / "train_1gpu.sh"
    with open(script_path, "w") as fd:
        fd.write(script_full)
    expected_out_dir = tmp_shared_dir / "debug" / "runs" / "mnist_with_micro_mlp" / "_pytest_mila_cluster_train_1gpu"
    if expected_out_dir.is_dir():
        ssl4rs.utils.filesystem.recursively_remove_all(expected_out_dir)
    job_id, job_finished = _launch_sbatch_and_wait_for_completion(script_path, timeout_sec=180)
    assert job_finished, f"job {job_id} did not finish before timeout"
    assert expected_out_dir.is_dir()
    last_line = "Done (mnist_with_micro_mlp: _pytest_mila_cluster_train_1gpu, 'train-test', job=0)"
    output_log_path = tmp_shared_dir / f"{job_id}.log"
    assert output_log_path.is_file()
    with open(output_log_path) as fd:
        output_log = fd.read()
    assert last_line in output_log
    console_log_path = expected_out_dir / "console.log"
    with open(console_log_path) as fd:
        console_log = fd.read()
    assert last_line in console_log
