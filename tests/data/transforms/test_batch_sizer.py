import pytest

import ssl4rs.data.transforms.batch_sizer as batch_sizer


def test_batch_sizer__fixed():
    t = batch_sizer.BatchSizer(128)

    empty_batch = t({})
    assert len(empty_batch) == 1
    assert empty_batch["batch_size"] == 0

    not_empty_batch = t({"hello": "something"})
    assert len(not_empty_batch) == 2
    assert not_empty_batch["batch_size"] == 128
    assert not_empty_batch["hello"] == "something"

    already_sized_batch = t({"hello": "something", "batch_size": 128})
    assert len(already_sized_batch) == 2
    assert already_sized_batch["batch_size"] == 128
    assert already_sized_batch["hello"] == "something"

    with pytest.raises(AssertionError):
        _ = t({"hello": "something", "batch_size": 64})


def test_batch_sizer__hint():
    t = batch_sizer.BatchSizer("magic_array")

    empty_batch = t({})
    assert len(empty_batch) == 1
    assert empty_batch["batch_size"] == 0

    with pytest.raises(AssertionError):
        _ = t({"hello": "something"})

    magic_array = [0, 1, 2, 3]
    not_empty_batch = t({"magic_array": magic_array})
    assert len(not_empty_batch) == 2
    assert not_empty_batch["batch_size"] == 4
    assert not_empty_batch["magic_array"] is magic_array

    already_sized_batch = t({"magic_array": magic_array, "batch_size": 4})
    assert len(already_sized_batch) == 2
    assert already_sized_batch["batch_size"] == 4
    assert not_empty_batch["magic_array"] is magic_array

    with pytest.raises(AssertionError):
        _ = t({"magic_array": magic_array, "batch_size": 5})
