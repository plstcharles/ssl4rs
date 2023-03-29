import typing

import pytest

import tests.helpers.module_runner as module_runner


def _get_base_command(tmpdir, exp_name, test_name) -> typing.List[typing.AnyStr]:
    return [
        "train.py",
        "-m",
        f"experiment={exp_name}",
        f"utils.output_root_dir='{tmpdir}'",
        f"run_name=_pytest_debug_{test_name}",
        "++resume_from_latest_if_possible=False",  # we CANNOT resume w/ hydra multiruns
    ]


@pytest.mark.slow
def test_sweep_mnist_experiments(tmpdir):
    command = _get_base_command(tmpdir, "glob(example_mnist_*)", "mnist_experiments")
    command.append("++trainer.fast_dev_run=true")
    command.append("++trainer.accelerator=cpu")
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)


@pytest.mark.slow
def test_sweep_mnist_fast_hparams(tmpdir):
    command = _get_base_command(tmpdir, "example_mnist_classif_fast", "mnist_fast_hparams")
    command.extend(
        [
            "model.encoder_config.hidden_channels='[5]','[8]'",
            "data.datamodule.dataloader_fn_map._default_.batch_size=16,24",
            "++trainer.fast_dev_run=true",
        ]
    )
    output = module_runner.run(command)
    if output.returncode != 0:
        pytest.fail(output.stderr)
