import typing

import numpy as np
import pytest
import torch.utils.data

import ssl4rs.data.transforms.batch


class FakeDatasetParser(torch.utils.data.Dataset):
    """Fake dataset parser class that returns data in the expected batch dict format."""

    def __init__(
        self,
        data_tensor_shape: typing.Tuple[int, ...],
        num_samples: int = 35,
        use_variable_data_tensor_shapes: bool = False,
        num_classes: int = 10,
        data_tensor_name: str = "data",
        label_tensor_name: str = "label",
    ):
        super().__init__()
        self._data_tensor_shape = data_tensor_shape
        self._num_samples = num_samples
        self._use_variable_data_tensor_shapes = use_variable_data_tensor_shapes
        self._num_classes = num_classes
        self._data_tensor_name = data_tensor_name
        self._label_tensor_name = label_tensor_name

    @property
    def shape(self):
        return len(self), *self._data_tensor_shape

    @property
    def num_classes(self):
        return self._num_classes

    def __len__(self):
        return self._num_samples

    def __getitem__(self, item):
        assert isinstance(item, int)
        assert 0 <= item < len(self)
        rng = np.random.RandomState(seed=item)
        if self._use_variable_data_tensor_shapes:
            data_tensor_shape = [rng.randint(1, max_n + 1) for max_n in self._data_tensor_shape]
        else:
            data_tensor_shape = self._data_tensor_shape
        return {
            self._data_tensor_name: rng.randn(*data_tensor_shape).astype(np.float32),
            self._label_tensor_name: rng.randint(0, self._num_classes),
            ssl4rs.data.transforms.batch.batch_index_key: item,
        }


@pytest.fixture
def fake_image_dataset() -> torch.utils.data.Dataset:
    return FakeDatasetParser(
        data_tensor_shape=(512, 512, 3),
        data_tensor_name="image",
    )


@pytest.fixture
def fake_dyn_shape_image_dataset() -> torch.utils.data.Dataset:
    return FakeDatasetParser(
        data_tensor_shape=(512, 512),
        use_variable_data_tensor_shapes=True,
        data_tensor_name="image",
    )
