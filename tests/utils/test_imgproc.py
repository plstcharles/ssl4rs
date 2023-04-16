import pathlib

import numpy as np
import PIL.Image
import pytest

import ssl4rs.utils.imgproc as imgproc
import ssl4rs.utils.patch_coord as patches


def test_get_image_shape_from_file(tmpdir):
    img_dir = pathlib.Path(tmpdir) / "dummy_images_for_shape_getter"
    img_dir.mkdir(exist_ok=True)
    np.random.seed(0)
    for img_idx in range(50):
        img_h, img_w = np.random.randint(8, 1000), np.random.randint(8, 1000)
        img = np.random.randint(0, 256, size=(img_h, img_w, 3), dtype=np.uint8)
        img_path = img_dir / f"{img_idx}.jpg"
        PIL.Image.fromarray(img).save(img_path)
        found_h, found_w = imgproc.get_image_shape_from_file(img_path)
        assert found_h == img_h and found_w == img_w


def test_decode_jpg_without_rescale(tmpdir):
    img_dir = pathlib.Path(tmpdir) / "dummy_images_for_decoder_without_rescale"
    img_dir.mkdir(exist_ok=True)
    np.random.seed(0)
    for img_idx in range(50):
        img_h, img_w = np.random.randint(8, 1000), np.random.randint(8, 1000)
        img = np.random.randint(0, 256, size=(img_h, img_w, 3), dtype=np.uint8)
        img_path = img_dir / f"{img_idx}.jpg"
        PIL.Image.fromarray(img).save(img_path)
        decoded_img = imgproc.decode_jpg(image=img_path)
        assert decoded_img.ndim == 3 and decoded_img.shape[-1] == 3
        assert decoded_img.shape == img.shape


@pytest.mark.parametrize("downscale_ratio", [2, 4, 8])
def test_decode_jpg_with_rescale(tmpdir, downscale_ratio):
    # note: scales 1:2 + 1:4 are SIMD-enabled, 1:8 is not (according to libjpeg-turbo docs)
    img_dir = pathlib.Path(tmpdir) / "dummy_images_for_decoder_with_rescale"
    img_dir.mkdir(exist_ok=True)
    np.random.seed(0)
    for img_idx in range(50):
        img_h, img_w = np.random.randint(8, 1000), np.random.randint(8, 1000)
        img = np.random.randint(0, 256, size=(img_h, img_w, 3), dtype=np.uint8)
        img_path = img_dir / f"{img_idx}.jpg"
        PIL.Image.fromarray(img).save(img_path)
        decoded_img = imgproc.decode_jpg(
            image=img_path,
            to_bgr_format=False,
            use_fast_upsample=True,
            use_fast_dct=True,
            scaling_factor=(1, downscale_ratio),
        )
        assert decoded_img.ndim == 3 and decoded_img.shape[-1] == 3
        expected_shape = tuple(int(round(z * (1 / downscale_ratio))) for z in img.shape[:2])
        assert np.isclose(decoded_img.shape[:2], expected_shape, atol=1).all()


def test_flex_crop_regular_tensor():
    image = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))  # B x C x H x W
    patch = patches.PatchCoord((2, 3), bottom_right=(5, 6))
    crop = imgproc.flex_crop(
        image=image,
        patch=patch,
    )
    assert crop.shape == (3, 4, 3, 3)  # B x C x patchsize
    assert np.array_equal(crop, image[:, :, 2:5, 3:6])
    assert crop.base is not None  # i.e. it is a view
    crop_copy = imgproc.flex_crop(
        image=image,
        patch=patch,
        force_copy=True,
    )
    assert crop_copy.base is None  # in this case, it should now be a copy
    assert np.array_equal(crop_copy, crop)
    crop[:, :, -1, -1] = -1
    assert not np.array_equal(crop_copy, crop)
    assert (image[:, :, 4, 5] == -1).all()
    # now, let's test the oob behavior
    patch = patches.PatchCoord((2, 3), bottom_right=(7, 5))
    crop = imgproc.flex_crop(
        image=image,
        patch=patch,
    )
    assert crop.shape == (3, 4, 5, 2)
    assert np.array_equal(crop[:, :, :3, :], image[:, :, 2:, 3:5])
    assert crop.base is None  # we added padding, it cannot be a view


def test_flex_crop_opencv_image():
    image = np.arange(600, dtype=np.int16).reshape((10, 20, 3))
    patch = patches.PatchCoord((2, 3), bottom_right=(8, 6))
    crop = imgproc.flex_crop(
        image=image,
        patch=patch,
    )
    assert crop.shape == (6, 3, 3)  # patchsize x C
    assert np.array_equal(crop, image[2:8, 3:6, :])
    assert crop.base is not None  # i.e. it is a view
