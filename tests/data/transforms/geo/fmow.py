import pathlib
import typing
from unittest import mock

import cv2 as cv
import numpy as np
import torch

from ssl4rs.data.transforms.geo.fmow import JPEGDecoderWithRandomResizedCrop


def fake_noop_resize(img: torch.Tensor, size: typing.Sequence[int], *args, **kwargs):
    assert img.shape[-2] == size[0]
    assert img.shape[-1] == size[1]
    return img


@mock.patch("torchvision.transforms.functional.resize")
def test_jpeg_decode_with_fixed_crop(resize_mock, tmpdir):
    resize_mock.side_effect = fake_noop_resize
    t = JPEGDecoderWithRandomResizedCrop(
        min_crop_size=(512, 512),
        output_size=(512, 512),
        gsd_ratios=(1.0, 1.0),
    )
    image = (np.ones((400, 400, 3), dtype=np.float32) * 127).astype(np.uint8)
    image_path = pathlib.Path(tmpdir) / "tmp.jpg"
    cv.imwrite(str(image_path), image, [cv.IMWRITE_JPEG_QUALITY, 100])
    with open(image_path, "rb") as fd:
        image_bytes = fd.read()
    out = t({
        "image/rgb/jpg": image_bytes,
        "image/rgb/gsd": 1.0,
    })
    assert out["image/rgb/jpg"].shape == (3, 512, 512)
    assert out["image/rgb/gsd"] == 1.0
    output = out["image/rgb/jpg"]
    output_display = (torch.permute(output, (1, 2, 0)).numpy() * 255).astype(np.uint8)[..., ::-1]
    assert np.array_equal(image, output_display[0:image.shape[0], 0:image.shape[1]])
    assert resize_mock.call_count == 1


def fake_downscale_resize(img: torch.Tensor, size: typing.Sequence[int], *args, **kwargs):
    assert img.shape[-2] == size[0] * 2
    assert img.shape[-1] == size[1] * 2
    return img[..., 0:size[0], 0:size[1]]


@mock.patch("torchvision.transforms.functional.resize")
def test_jpeg_decode_with_downscaled_crop(resize_mock, tmpdir):
    resize_mock.side_effect = fake_downscale_resize
    t = JPEGDecoderWithRandomResizedCrop(
        min_crop_size=(1024, 1024),
        output_size=(512, 512),
        gsd_ratios=(2.0, 2.0),
    )
    image = (np.ones((1024, 1024, 3), dtype=np.float32) * 127).astype(np.uint8)
    image_path = pathlib.Path(tmpdir) / "tmp.jpg"
    cv.imwrite(str(image_path), image, [cv.IMWRITE_JPEG_QUALITY, 100])
    with open(image_path, "rb") as fd:
        image_bytes = fd.read()
    out = t({
        "image/rgb/jpg": image_bytes,
        "image/rgb/gsd": 1.0,
    })
    assert out["image/rgb/jpg"].shape == (3, 512, 512)
    assert out["image/rgb/gsd"] == 2.0
    output = out["image/rgb/jpg"]
    output_display = (torch.permute(output, (1, 2, 0)).numpy() * 255).astype(np.uint8)[..., ::-1]
    assert (output_display == 127).all()
