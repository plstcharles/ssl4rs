import os

import pytest
import torch

import ssl4rs.utils.config
from ssl4rs.data.datamodules.mnist import DataModule as MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    dataloader_fn_map = dict(
        _default_=dict(
            _target_="torch.utils.data.DataLoader",
            batch_size=batch_size,
        )
    )
    mnist_data_dir = ssl4rs.utils.config.get_data_root_dir() / "mnist"
    datamodule = MNISTDataModule(data_dir=mnist_data_dir, dataloader_fn_map=dataloader_fn_map)
    datamodule.prepare_data()
    assert not datamodule.data_train and not datamodule.data_valid and not datamodule.data_test
    datamodule.setup()
    assert datamodule.data_train and datamodule.data_valid and datamodule.data_test
    assert len(datamodule.data_train) + len(datamodule.data_valid) + len(datamodule.data_test) == 70_000
    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    assert isinstance(batch, dict)
    assert "data" in batch and "target" in batch
    x, y = batch["data"], batch["target"]
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert (y >= 0).all() and (y < 10).all()

    assert "batch_size" in batch
    assert isinstance(batch["batch_size"], torch.Tensor)
    assert batch["batch_size"].sum().item() == batch_size

    assert "batch_id" in batch
    assert len(batch["batch_id"]) == batch_size
    assert len(set(batch["batch_id"])) == batch_size
