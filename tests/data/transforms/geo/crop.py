import math
import typing
from unittest import mock

import pytest
import torch

import ssl4rs.data.transforms.geo.crop as geo_crop


def test_geo_random_resized_crop():
    data = torch.randn(1, 1024, 2048)
    gsd = 0.1

    t = geo_crop.GroundSamplingDistanceAwareRandomResizedCrop(
        size=(256, 128),
        gsd_ratios=(0.5, 2),
    )
    for _ in range(100):
        out, new_gsd = t(data, gsd)
        assert out.shape == (1, 256, 128)
        assert 0.05 <= new_gsd <= 0.2

    t_with_dict = geo_crop.GroundSamplingDistanceAwareRandomResizedCrop(
        size=(128, 256),
        gsd_ratios=(0.5, 4),
        target_key="potato",
        gsd_key="carrot",
    )
    for _ in range(100):
        updated_dict = t_with_dict(
            {
                "potato": data,
                "carrot": gsd,
            }
        )
        assert not torch.equal(updated_dict["potato"], data)
        assert updated_dict["potato"].shape == (1, 128, 256)
        assert 0.05 <= updated_dict["carrot"] <= 0.4


def test_geo_random_resized_crop_with_bad_ratio():
    data = torch.randn(1, 512, 1024)
    gsd = 0.1
    t_ok = geo_crop.GroundSamplingDistanceAwareRandomResizedCrop(
        size=(256, 500),
        gsd_ratios=(2, 2),
    )
    out, new_gsd = t_ok(data, gsd)
    assert out.shape == (1, 256, 500)
    assert math.isclose(new_gsd, 0.2)
    t_bad = geo_crop.GroundSamplingDistanceAwareRandomResizedCrop(
        size=(256, 500),
        gsd_ratios=(2.001, 2.001),
    )
    with pytest.raises(ValueError):
        t_bad(data, gsd)


def test_geo_random_resized_crop_with_torchscript():
    data = torch.randn(1, 1024, 1024)
    gsd = 0.1
    t = geo_crop.GroundSamplingDistanceAwareRandomResizedCrop(
        size=(256, 256),
        gsd_ratios=(0.5, 4),
    )
    t_compiled = torch.compile(t)
    for _ in range(10):
        out, new_gsd = t_compiled(data, gsd)
        assert out.shape == (1, 256, 256)
        assert 0.05 <= new_gsd <= 0.4


def fake_resize(img: torch.Tensor, size: typing.Sequence[int], *args, **kwargs):
    assert img.ndim == 3 and img.shape[0] == 1
    assert (img == 1).all()
    input_cols = img.shape[-1]
    output_cols = size[-1]
    output_val = input_cols / output_cols
    return torch.ones(1, *size) * output_val


@mock.patch("torchvision.transforms.functional.resize")
def test_geo_random_resized_crop_interpolation(resize_mock):
    # note: this assumes that the base impl relies on torchvision's resize op
    resize_mock.side_effect = fake_resize
    data = torch.ones(1, 1024, 512)
    gsd = 1.0
    t = geo_crop.GroundSamplingDistanceAwareRandomResizedCrop(
        size=(128, 256),
        gsd_ratios=(0.5, 4),
    )
    for _ in range(10):
        out, new_gsd = t(data, gsd)
        assert out.shape == (1, 128, 256)
        assert 0.5 <= new_gsd <= 4.0
        assert len(torch.unique(out)) == 1
        assert math.isclose(out[0, 0, 0].item(), new_gsd)


def test_geo_center_fixed_crop():
    data = torch.randn(1, 1024, 2048)
    t_without_resize = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(256, 128),
        output_gsd=None,
    )
    t_with_resize = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(120, 150),
        output_gsd=0.5,
        target_key="potato",
        gsd_key="carrot",
    )
    for _ in range(100):
        out, new_gsd = t_without_resize(data, 0.1)
        assert out.shape == (1, 256, 128)
        assert new_gsd == 0.1
        out, new_gsd = t_with_resize(data, 0.1)
        assert out.shape == (1, 120, 150)
        assert new_gsd == 0.5
        updated_dict = t_with_resize(
            {
                "potato": data,
                "carrot": 0.1,
            }
        )
        assert not torch.equal(updated_dict["potato"], data)
        assert updated_dict["potato"].shape == (1, 120, 150)
        assert updated_dict["carrot"] == 0.5


def test_geo_center_fixed_crop_with_bad_crops():
    data = torch.randn(1, 512, 1024)
    gsd = 0.1
    t_ok = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(256, 500),
        output_gsd=0.2,
    )
    out, new_gsd = t_ok(data, gsd)
    assert out.shape == (1, 256, 500)
    assert math.isclose(new_gsd, 0.2)
    t_ok = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(512, 1024),
        output_gsd=None,
    )
    out, new_gsd = t_ok(data, gsd)
    assert out.shape == (1, 512, 1024)
    assert new_gsd == 0.1
    t_bad = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(256, 500),
        output_gsd=0.21,
    )
    with pytest.raises(ValueError):
        t_bad(data, gsd)
    t_bad = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(256, 1200),
        output_gsd=None,
    )
    with pytest.raises(ValueError):
        t_bad(data, gsd)


def test_geo_center_fixed_crop_with_padded_crops():
    data = torch.ones(1, 512, 1024)
    gsd = 0.1
    t_bad = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(600, 1024),
        output_gsd=0.1,
    )
    with pytest.raises(ValueError):
        t_bad(data, gsd)
    t_ok = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(600, 1024),
        output_gsd=0.1,
        allow_auto_padding=True,
    )
    out, new_gsd = t_ok(data, gsd)
    assert out.shape == (1, 600, 1024)
    assert new_gsd == 0.1
    assert torch.count_nonzero(out == 0).item() == (600 - 512) * 1024


def test_geo_center_fixed_crop_with_torchscript():
    data = torch.randn(1, 1024, 1024)
    gsd = 0.1
    t = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(256, 256),
        output_gsd=0.2,
    )
    t_compiled = torch.compile(t)
    for _ in range(10):
        out, new_gsd = t_compiled(data, gsd)
        assert out.shape == (1, 256, 256)
        assert new_gsd == 0.2


@mock.patch("torchvision.transforms.functional.resize")
def test_geo_center_fixed_crop_interpolation(resize_mock):
    # note: this assumes that the base impl relies on torchvision's resize op
    resize_mock.side_effect = fake_resize
    data = torch.ones(1, 1024, 512)
    gsd = 1.0
    t_downsample = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(128, 256),
        output_gsd=2.0,
    )
    out, new_gsd = t_downsample(data, gsd)
    assert out.shape == (1, 128, 256)
    assert new_gsd == 2.0
    assert len(torch.unique(out)) == 1
    assert math.isclose(out[0, 0, 0].item(), new_gsd)
    t_upsample = geo_crop.GroundSamplingDistanceAwareCenterFixedCrop(
        size=(256, 128),
        output_gsd=0.5,
    )
    out, new_gsd = t_upsample(data, gsd)
    assert out.shape == (1, 256, 128)
    assert new_gsd == 0.5
    assert len(torch.unique(out)) == 1
    assert math.isclose(out[0, 0, 0].item(), new_gsd)
