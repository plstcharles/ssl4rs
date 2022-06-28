"""Contains utilities related to image loading/processing/drawing."""

import typing

import imagesize
import numpy as np
import torch


import ssl4rs.utils.patch_coord


def get_image_shape_from_file(file_path: typing.AnyStr) -> typing.Tuple[int, int]:
    """Returns the (height, width) of an image stored on disk.

    This function will parse the file header and try to deduct the image size without actually
    opening the image. This should make it much faster to quickly parse datasets to validate their
    contents without opening each image directly.
    """
    width, height = imagesize.get(file_path)
    assert width > 0 and height > 0, "invalid image shape!"
    return height, width


def flex_crop(
    image: typing.Union[np.ndarray, torch.Tensor],
    patch: "ssl4rs.utils.patch_coord.PatchCoord",
    padding_val: typing.Union[int, float] = 0,
    force_copy: bool = False,
) -> typing.Union[np.ndarray, torch.Tensor]:
    """Flexibly crops a region from within an OpenCV or PyTorch image, using padding if needed.

    If the image is provided as a numpy array with two or three dimensions, it will be cropped under
    the assumption that it is an OpenCV-like image with a number of channels of 1 to 4 and that the
    channel dimension is the last in the array shape (e.g. H x W x C), resulting in patches with
    a similar dimension ordering. Otherwise, the image will be cropped under the assumption that we
    are doing a 2D spatial crop and that the spatial dimensions are the last dimensions in the
    array shape (e.g. for a 4-dim tensor B x C x H x W and a 2D crop size, the output will be
    B x C x [patch.shape]).

    Args:
        image: the image to crop (provided as a numpy array or torch tensor).
        patch: a patch coordinates object describing the area to crop from the image.
        padding_val: border value to use when padding the image.
        force_copy: defines whether to force a copy of the target image region even when it can be
            avoided.

    Returns:
        The cropped image of the same object type as the input image, with the same dim arrangement.
    """
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise TypeError("expected input image to be numpy array or torch tensor")

    if isinstance(image, np.ndarray) and image.ndim in [2, 3] and patch.ndim == 2:
        # special handling for OpenCV-like arrays (where the channel is the last dimension)
        assert image.ndim == 2 or image.shape[2] in [1, 2, 3, 4], \
            "cannot handle channel counts outside [1, 2, 3, 4] with opencv crop!"
        image_region = ssl4rs.utils.patch_coord.PatchCoord((0, 0), shape=image.shape[:2])
        if patch not in image_region:
            # special handling for crop coordinates falling outside the image bounds...
            crop_out_shape = patch.shape if image.ndim == 2 else (patch.shape, image.shape[2])
            inters_coord = patch.intersection(image_region)
            # this forces an allocation, as we cannot get a view with the required padding
            crop_out = np.full(shape=crop_out_shape, fill_value=padding_val, dtype=image.dtype)
            if inters_coord is not None:  # if the intersection contains any pixels at all, copy...
                offset = [inters_coord.tl[0] - patch.tl[0], inters_coord.tl[1] - patch.tl[1]]
                crop_out[
                    offset[0]:(offset[0] + inters_coord.shape[0]),
                    offset[1]:(offset[1] + inters_coord.shape[1]),
                    ...
                ] = image[
                    inters_coord.tl[0]:inters_coord.br[0],
                    inters_coord.tl[1]:inters_coord.br[1],
                    ...
                ]
            return crop_out
        # if all crop coordinates are in-bounds, we can get a view directly from the image
        crop_view = image[patch.tl[0]:patch.br[0], patch.tl[1]:patch.br[1], ...]
        if not force_copy:
            return crop_view
        return np.copy(crop_view)

    # regular handling (we crop along the spatial dimensions located at the end of the array)
    assert patch.ndim <= image.ndim, \
        "patch dim count should be equal to or lower than image dimension count!"
    image_region = ssl4rs.utils.patch_coord.PatchCoord(
        top_left=[0] * patch.ndim,
        shape=image.shape[-patch.ndim:]
    )
    crop_out_shape = tuple(image.shape[:-patch.ndim]) + patch.shape

    # first check: figure out if there is anything to crop at all
    inters_coord = patch.intersection(image_region)
    if inters_coord is None or inters_coord.is_empty:
        # if not, we can return right away without any allocation (out shape will have a zero-dim)
        if isinstance(image, torch.Tensor):
            return torch.empty(crop_out_shape, dtype=image.dtype, device=image.device)
        else:
            return np.empty(crop_out_shape, dtype=image.dtype)

    # if there is an intersection, figure out if it's totally inside the image or not
    if patch not in image_region:
        # ...it's not totally in the image, so we'll have to allocate + fill
        if isinstance(image, torch.Tensor):
            crop_out = \
                torch.full(crop_out_shape, padding_val, dtype=image.dtype, device=image.device)
        else:
            crop_out = np.full(crop_out_shape, padding_val, dtype=image.dtype)
        offset = [inters_coord.tl[d] - patch.tl[d] for d in patch.dimrange]
        crop_inner_slice = tuple([slice(None)] * (image.ndim - patch.ndim) + [
            slice(offset[d], offset[d] + inters_coord.shape[d])
            for d in patch.dimrange
        ])
        crop_outer_slice = tuple([slice(None)] * (image.ndim - patch.ndim)) + inters_coord.slice
        crop_out[crop_inner_slice] = image[crop_outer_slice]
        return crop_out

    # if we get here, there is an intersection without any out-of-bounds element lookup
    crop_outer_slice = tuple([slice(None)] * (image.ndim - patch.ndim)) + inters_coord.slice
    crop_view = image[crop_outer_slice]
    if not force_copy:
        return crop_view
    elif isinstance(image, np.ndarray):
        return np.copy(crop_view)
    else:
        return crop_view.clone()  # note: this will not detach the tensor, just make a copy!
