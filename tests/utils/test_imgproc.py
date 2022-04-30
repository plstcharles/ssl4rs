import numpy as np
import pytest
import ssl4rs.utils.imgproc as imgproc
import ssl4rs.utils.patches as patches


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
