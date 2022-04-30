import numpy as np
import ssl4rs.data.transforms.patchify


def test_patchify_no_overlap_no_grid():
    img = np.arange(100).reshape(10, 10)
    patch_shape = (2, 2)
    patcher = ssl4rs.data.transforms.patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=False,
    )
    out = patcher(img)
    assert isinstance(out, list) and len(out) == 25
    assert np.array_equal(out[0], img[0:2, 0:2])
    assert np.array_equal(out[1], img[0:2, 2:4])
    assert np.array_equal(out[5], img[2:4, 0:2])
    assert np.array_equal(out[-1], img[8:10, 8:10])

    img = np.arange(100).reshape(10, 10)
    patch_shape = (2, 4)
    patcher = ssl4rs.data.transforms.patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=False,
    )
    out = patcher(img)
    assert isinstance(out, list) and len(out) == 10
    assert np.array_equal(out[0], img[0:2, 0:4])
    assert np.array_equal(out[1], img[0:2, 4:8])
    assert np.array_equal(out[2], img[2:4, 0:4])
    assert np.array_equal(out[-1], img[8:10, 4:8])


def test_patchify_no_overlap_with_grid():
    img = np.arange(100).reshape(10, 10)
    patch_shape = (2, 2)
    patcher = ssl4rs.data.transforms.patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=True,
    )
    out = patcher(img)
    assert isinstance(out, np.ndarray) and out.shape == (5, 5, 2, 2)
    assert np.array_equal(out[0, 0], img[0:2, 0:2])
    assert np.array_equal(out[0, 1], img[0:2, 2:4])
    assert np.array_equal(out[0, 2], img[0:2, 4:6])
    assert np.array_equal(out[1, 0], img[2:4, 0:2])
    assert np.array_equal(out[1, 1], img[2:4, 2:4])
    assert np.array_equal(out[-1, -1], img[8:10, 8:10])

    img = np.arange(100).reshape(10, 10)
    patch_shape = (2, 4)
    patcher = ssl4rs.data.transforms.patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=True,
    )
    out = patcher(img)
    assert isinstance(out, np.ndarray) and out.shape == (5, 2, 2, 4)
    assert np.array_equal(out[0, 0], img[0:2, 0:4])
    assert np.array_equal(out[0, 1], img[0:2, 4:8])
    assert np.array_equal(out[1, 0], img[2:4, 0:4])
    assert np.array_equal(out[1, 1], img[2:4, 4:8])
    assert np.array_equal(out[-1, -1], img[8:10, 4:8])


def test_patchify_overlap_with_grid():
    img = np.arange(1200).reshape(30, 40)
    patch_shape = (4, 8)
    patcher = ssl4rs.data.transforms.patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0.25,
        offset_overlap=False,
        make_contiguous=True,
    )
    out = patcher(img)
    assert isinstance(out, np.ndarray) and out.shape == (9, 6, 4, 8)
    assert np.array_equal(out[0, 0], img[0:4, 0:8])
    assert np.array_equal(out[0, 1], img[0:4, 6:14])
    assert np.array_equal(out[1, 0], img[3:7, 0:8])
    assert np.array_equal(out[1, 1], img[3:7, 6:14])
    assert np.array_equal(out[-1, -1], img[24:28, 30:38])


def test_patchify_overlap_with_grid_and_offset():
    img = np.arange(1200).reshape(30, 40)
    patch_shape = (4, 8)
    patcher = ssl4rs.data.transforms.patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0.25,
        offset_overlap=True,
        padding_val=-1,
        make_contiguous=True,
    )
    out = patcher(img)
    assert isinstance(out, np.ndarray) and out.shape == (10, 7, 4, 8)
    assert (out[0, 0][:, :2] == -1).all()
    assert (out[0, 0][:1, :] == -1).all()
    assert np.array_equal(out[0, 0, 1:, 2:], img[0:3, 0:6])
    assert (out[0, 1][:1, :] == -1).all()
    assert np.array_equal(out[0, 1][1:, :], img[0:3, 4:12])
    assert (out[1, 0][:, :2] == -1).all()
    assert np.array_equal(out[1, 0][:, 2:], img[2:6, 0:6])
    assert (out[-1, -1][:, 6:] == -1).all()
    assert np.array_equal(out[-1, -1][:, :6], img[26:30, 34:40])


def test_patchify_with_mask():
    img = np.arange(1200).reshape(30, 40)
    patch_shape = (4, 8)
    patcher = ssl4rs.data.transforms.patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=False,
    )
    mask = np.zeros((30, 40))
    mask[16:, 10:] = 1
    out = patcher(img, mask)
    assert isinstance(out, list) and len(out) == 9
    assert np.array_equal(out[0], img[16:20, 10:18])
    assert np.array_equal(out[1], img[16:20, 18:26])
    assert np.array_equal(out[2], img[16:20, 26:34])
    assert np.array_equal(out[3], img[20:24, 10:18])
    assert np.array_equal(out[-1], img[24:28, 26:34])
