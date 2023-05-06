import math

import pytest
import torch

import ssl4rs.data.transforms.geo.crop as geo_crop


def test_geo_crop():
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


def test_geo_crop_with_bad_ratio():
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


def test_geo_crop_with_torchscript():
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
