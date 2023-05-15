import pytest

import ssl4rs.data.transforms.batch as batch_transforms
from ssl4rs.data import batch_size_key


def test_batch_sizer__fixed():
    t = batch_transforms.BatchSizer(128)

    empty_batch = t({})
    assert len(empty_batch) == 1
    assert empty_batch[batch_size_key] == 0

    not_empty_batch = t({"hello": "something"})
    assert len(not_empty_batch) == 2
    assert not_empty_batch[batch_size_key] == 128
    assert not_empty_batch["hello"] == "something"

    already_sized_batch = t({"hello": "something", batch_size_key: 128})
    assert len(already_sized_batch) == 2
    assert already_sized_batch[batch_size_key] == 128
    assert already_sized_batch["hello"] == "something"

    with pytest.raises(AssertionError):
        _ = t({"hello": "something", batch_size_key: 64})


def test_batch_sizer__hint():
    t = batch_transforms.BatchSizer("magic_array")

    empty_batch = t({})
    assert len(empty_batch) == 1
    assert empty_batch[batch_size_key] == 0

    with pytest.raises(AssertionError):
        _ = t({"hello": "something"})

    magic_array = [0, 1, 2, 3]
    not_empty_batch = t({"magic_array": magic_array})
    assert len(not_empty_batch) == 2
    assert not_empty_batch[batch_size_key] == 4
    assert not_empty_batch["magic_array"] is magic_array

    already_sized_batch = t({"magic_array": magic_array, batch_size_key: 4})
    assert len(already_sized_batch) == 2
    assert already_sized_batch[batch_size_key] == 4
    assert not_empty_batch["magic_array"] is magic_array

    with pytest.raises(AssertionError):
        _ = t({"magic_array": magic_array, batch_size_key: 5})


def test_batch_identifier():
    t = batch_transforms.BatchIdentifier(
        batch_id_prefix="potato",
        batch_index_key="INDEX",
        dataset_name="hello",
    )
    batch_id_key = batch_transforms.batch_id_key
    # case 1: batch id already in batch dict
    for idx in range(10):
        out = t({batch_id_key: idx})
        assert out[batch_id_key] == idx
    # case 2: batch id missing, but index provided manually
    for idx in range(10):
        out = t({}, index=idx)
        assert f"{idx}" in out[batch_id_key]
    # case 3: batch id missing, but index inside batch
    for idx in range(10):
        out = t({"INDEX": idx})
        assert f"{idx}" in out[batch_id_key]
    # finally, check that batch ids include prefix/dataset and are unique
    ids = []
    for idx in range(1000):
        out = t({"INDEX": idx})
        assert f"{idx}" in out[batch_id_key]
        assert out[batch_id_key].startswith("potato")
        assert "hello" in out[batch_id_key]
        ids.append(out[batch_id_key])
    ids = set(ids)
    assert len(ids) == 1000
