import pathlib

import numpy as np
import pytest
import torch.utils.data

import ssl4rs.data.parsers.utils
import ssl4rs.data.repackagers.utils


class _FakeDatasetRepackager(ssl4rs.data.repackagers.utils.DeepLakeRepackager):
    @property
    def image_shape(self):
        return self.dataset.shape[1:]

    @property
    def class_names(self):
        return [f"{idx}" for idx in range(self.dataset.num_classes)]

    @property
    def tensor_info(self):
        return dict(
            image=dict(htype="image", dtype=np.float32, sample_compression=None),
            label=dict(htype="class_label", dtype=np.int16, class_names=self.class_names),
        )

    @property
    def dataset_info(self):
        return dict(
            name=self.dataset_name,
            class_names=self.class_names,
            image_shape=list(self.image_shape),
        )

    @property
    def dataset_name(self) -> str:
        return "fake_image_dataset"

    def __len__(self) -> int:
        return len(self.dataset)

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
    ):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, item: int):
        batch = self.dataset[item]
        batch = {t: batch[t] for t in self.tensor_names}
        return batch


@pytest.fixture
def fake_on_disk_dataset_path(tmpdir):
    return pathlib.Path(tmpdir) / "fake_dataset"


@pytest.fixture
def deeplake_on_disk_image_dataset(fake_image_dataset, fake_on_disk_dataset_path):
    repackager = _FakeDatasetRepackager(fake_image_dataset)
    dataset = repackager.export(fake_on_disk_dataset_path, num_workers=0)
    return dataset


def test_export_and_reopen_dataset(
    deeplake_on_disk_image_dataset,
    fake_on_disk_dataset_path,
    fake_image_dataset,
):
    assert len(deeplake_on_disk_image_dataset) == len(fake_image_dataset)
    on_disk_batch = deeplake_on_disk_image_dataset[0]
    orig_batch = fake_image_dataset[0]
    assert np.array_equal(
        on_disk_batch["image"].numpy(),
        orig_batch["image"],
    )
    assert on_disk_batch["label"].numpy() == orig_batch["label"]
    deeplake_reopened_dataset = ssl4rs.data.parsers.utils.DeepLakeParser(fake_on_disk_dataset_path)
    reopened_batch = deeplake_reopened_dataset[0]
    assert np.array_equal(
        reopened_batch["image"],
        orig_batch["image"],
    )
    assert reopened_batch["label"] == orig_batch["label"]


def test_get_dataloader(
    deeplake_on_disk_image_dataset,
):
    dataparser = ssl4rs.data.parsers.utils.deeplake.DeepLakeParser(deeplake_on_disk_image_dataset)
    dataloader = ssl4rs.data.parsers.utils.deeplake.get_dataloader(
        dataparser,
        num_workers=0,
        batch_size=8,
        drop_last=False,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    assert all([t in batch for t in ["batch_id", "batch_size", "index", "label", "image"]])
    assert ssl4rs.data.get_batch_size(batch) == 8
    images, labels = batch["image"], batch["label"]
    assert images.shape == (8, 512, 512, 3)
    assert labels.shape == (8, 1)
    batch7 = dataparser[7]
    assert torch.equal(
        images[7],
        torch.as_tensor(batch7["image"]),
    )
