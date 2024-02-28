import hydra
import pytest
import torch

import ssl4rs.utils.config
from ssl4rs.data import batch_size_key
from ssl4rs.data.datamodules.mnist import DataModule as MNISTDataModule


def _check_minibatch_content(minibatch, expected_batch_size: int = 0):
    # minibatch loaded by the new mnist datamodule should always be a dict w/ metadata!
    assert isinstance(minibatch, dict)
    batch_size = minibatch[batch_size_key].sum().item()
    assert isinstance(batch_size, int) and batch_size > 0
    if expected_batch_size != 0:
        assert batch_size == expected_batch_size
    assert isinstance(minibatch["batch_id"], list)
    assert len(minibatch["batch_id"]) == batch_size
    assert len(set(minibatch["batch_id"])) == batch_size
    # the original data/target fields from the tensor remap wrapper should also be there
    assert all([key in minibatch for key in ["data", "target"]])
    assert isinstance(minibatch["data"], torch.Tensor) and minibatch["data"].dtype == torch.float32
    assert minibatch["data"].shape == (batch_size, 1, 28, 28)  # B x C x H x W
    assert isinstance(minibatch["target"], torch.Tensor) and minibatch["target"].dtype == torch.int64
    assert minibatch["target"].shape == (batch_size,)  # B
    assert (minibatch["target"] >= 0).all().item() and (minibatch["target"] < 10).all().item()


def test_mnist_datamodule_via_hydra(tmpdir, global_cfg_cleaner):
    config = ssl4rs.utils.config.init_hydra_and_compose_config(
        configs_dir="../../../ssl4rs/configs",
        output_root_dir=tmpdir,
        overrides=["data=mnist.yaml"],
    )
    datamodule = hydra.utils.instantiate(config.data.datamodule)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    minibatch = next(iter(train_dataloader))
    _check_minibatch_content(minibatch)


@pytest.mark.parametrize("batch_size", [32, 64])
def test_mnist_datamodule(batch_size):
    dataparser_configs = dict(
        valid=dict(
            dataset_name="potato",
        ),
    )
    dataloader_configs = dict(
        _default_=dict(
            _target_="torch.utils.data.DataLoader",
            batch_size=batch_size,
        )
    )
    mnist_data_dir = ssl4rs.utils.config.get_data_root_dir() / "mnist"
    datamodule = MNISTDataModule(
        data_dir=mnist_data_dir,
        dataparser_configs=dataparser_configs,
        dataloader_configs=dataloader_configs,
    )
    datamodule.prepare_data()
    assert not datamodule.data_train and not datamodule.data_valid and not datamodule.data_test
    datamodule.setup()
    assert datamodule.data_train and datamodule.data_valid and datamodule.data_test
    assert len(datamodule.data_train) + len(datamodule.data_valid) + len(datamodule.data_test) == 70_000
    assert datamodule.data_valid.dataset_name == "potato"
    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()
    minibatch = next(iter(datamodule.train_dataloader()))
    _check_minibatch_content(minibatch)
