import numpy as np
import torch
import torch.utils.data

import ssl4rs.data.parsers.utils.wrappers


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.fake_data = [[1, 2, 3, 4], [5, 6], [7], [], [8, 9, 10]]

    def __len__(self) -> int:
        return len(self.fake_data)

    def __getitem__(self, idx: int) -> dict:
        return {
            "some_constant": "hello!",
            "orig_idx": idx,
            "some_array": self.fake_data[idx],
            "unused": int,
        }


def test_wrap_fake_dataset_without_shuffle():
    dataset = _FakeDataset()
    wrapped_dataset = ssl4rs.data.parsers.utils.wrappers.IterableDataset(
        dataset_to_wrap=dataset,
        target_array_keys=["some_array"],
        constant_copy_keys=["some_constant", "orig_idx"],
        shuffle=False,
    )
    loaded_data = list(wrapped_dataset)
    assert len(loaded_data) == 10
    assert all(["unused" not in b for b in loaded_data])
    assert all([b["some_constant"] == "hello!" for b in loaded_data])
    loaded_values = [b["some_array"] for b in loaded_data]
    assert np.array_equal(loaded_values, np.arange(1, 11))
    loaded_idxs = [b["orig_idx"] for b in loaded_data]
    assert np.array_equal(loaded_idxs, [0, 0, 0, 0, 1, 1, 2, 4, 4, 4])


def test_wrap_fake_dataset_with_shuffle():
    dataset = _FakeDataset()
    wrapped_dataset = ssl4rs.data.parsers.utils.wrappers.IterableDataset(
        dataset_to_wrap=dataset,
        target_array_keys=["some_array"],
        shuffle=True,
        buffer_size=5,
        shuffle_seed=0,
    )
    loaded_data = list(wrapped_dataset)
    assert len(loaded_data) == 10
    loaded_values = [b["some_array"] for b in loaded_data]
    assert not np.array_equal(loaded_values, np.arange(1, 11))
    assert np.array_equal(np.unique(loaded_values), np.arange(1, 11))


def test_wrap_fake_dataset_with_shuffle_and_workers():
    dataset = _FakeDataset()
    wrapped_dataset = ssl4rs.data.parsers.utils.wrappers.IterableDataset(
        dataset_to_wrap=dataset,
        target_array_keys=["some_array"],
        shuffle=True,
        buffer_size=5,
        shuffle_seed=0,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=wrapped_dataset,
        batch_size=3,
        shuffle=False,
        num_workers=2,
    )
    loaded_batches = [b for b in dataloader]
    assert len(loaded_batches) == 4
    loaded_arrays = [b["some_array"] for b in loaded_batches]
    assert np.array_equal([len(a) for a in loaded_arrays], [3, 3, 3, 1])
    loaded_values = torch.cat(loaded_arrays).numpy()
    assert not np.array_equal(loaded_values, np.arange(1, 11))
    assert np.array_equal(np.unique(loaded_values), np.arange(1, 11))
