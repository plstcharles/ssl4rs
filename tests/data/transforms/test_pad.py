import torch

from ssl4rs.data.transforms.pad import (  # Replace with the actual module name
    PadIfNeeded,
    pad_if_needed,
)


def test_pad_if_needed_fn():
    tensor = torch.randn((2, 5, 5))
    padded_tensor = pad_if_needed(tensor, 10, 10)
    assert padded_tensor.shape == (2, 10, 10)
    assert torch.equal(padded_tensor[:, 0:5, 0:5], tensor)
    padded_tensor = pad_if_needed(tensor, 10, 10, centered=True)
    assert padded_tensor.shape == (2, 10, 10)
    assert torch.equal(padded_tensor[:, 2:7, 2:7], tensor)

    padded_tensor = pad_if_needed(tensor, 5, 5)
    assert padded_tensor.shape == (2, 5, 5)
    assert torch.equal(padded_tensor, tensor)
    padded_tensor = pad_if_needed(tensor, 3, 3)
    assert padded_tensor.shape == (2, 5, 5)
    assert torch.equal(padded_tensor, tensor)


def test_pad_if_needed_transform():
    tensor = torch.randn((2, 5, 5))
    pad_module = PadIfNeeded(
        target_key="data",
        min_height=10,
        min_width=10,
    )
    output = pad_module({"data": tensor})
    assert output["data"].shape == (2, 10, 10)
    assert output["data/prepad_shape"] == (2, 5, 5)
    pad_module = PadIfNeeded(
        target_key="data",
        min_height=4,
        min_width=4,
    )
    output = pad_module({"data": tensor})
    assert output["data"].shape == (2, 5, 5)
    assert output["data/prepad_shape"] == (2, 5, 5)
