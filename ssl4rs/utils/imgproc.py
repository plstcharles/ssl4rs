"""Contains utilities related to image loading/processing/drawing."""
import io
import pathlib
import typing
from unittest import mock

import numpy as np
import torch

import ssl4rs.utils.patch_coord

_turbojpeg_handler = None

try:
    import turbojpeg
except ImportError:
    turbojpeg = mock.Mock()


def _get_turbojpeg_handler() -> "turbojpeg.TurboJPEG":
    """Returns the global handler used to interact with the turbojpeg library."""
    global _turbojpeg_handler
    if _turbojpeg_handler is None:
        assert not isinstance(turbojpeg, mock.Mock), "missing package: PyTurboJPEG"
        _turbojpeg_handler = turbojpeg.TurboJPEG()
    return _turbojpeg_handler


def get_image_shape_from_file(
    file_path_or_data: typing.Union[io.BytesIO, typing.AnyStr, pathlib.Path],
    with_turbojpeg: bool = True,
) -> typing.Tuple[int, int]:
    """Returns the (height, width) of an image stored on disk.

    This function will parse the file header and try to deduct the image size without actually
    opening the image. This should make it much faster to quickly parse datasets to validate their
    contents without opening each image directly.

    This can be done via libjpeg-turbo (`with_turbojpeg=True`), or via the imagesize package.
    """
    if with_turbojpeg:
        turbojpeg_handler = _get_turbojpeg_handler()
        assert isinstance(file_path_or_data, (str, pathlib.Path)), "missing impl for bytes"
        jpeg_header = _read_jpeg_header_only(file_path_or_data)
        header = turbojpeg_handler.decode_header(jpeg_header)
        width, height, jpeg_subsample, jpeg_colorspace = header
    else:
        import imagesize

        width, height = imagesize.get(file_path_or_data)
    assert width > 0 and height > 0, "invalid image!"
    return height, width


def _read_jpeg_header_only(
    image: typing.Union[typing.AnyStr, pathlib.Path],
    chunk_size: int = 1024,
    body_padding_size: int = 32,
) -> bytes:
    """Reads and returns the header of a jpeg file using turbojpeg without reading the image data.

    If the file is not a valid JPEG, an exception will be thrown. For more information on the
    header markers, see:
        https://docs.fileformat.com/image/jpeg/

    Args:
        image: the path to the file to be parsed for a JPEG header.
        chunk_size: the size of the memory block to read in each iteration.
        body_padding_size: the size of the raw data to retain in the image body.

    Returns:
        The raw bytes representing the encoded header of the JPEG image. Every marker up to the
        start-of-signal (SOS) will be included; after the SOS, we will add some dummy body padding,
        and end the byte array with the end-of-image (EOI) marker.
    """
    image = pathlib.Path(image)
    assert image.is_file(), f"invalid image path: {image}"
    assert body_padding_size < chunk_size - 1
    with open(image, "rb") as fd:
        # read the first two bytes to check if it's a valid jpeg file
        first_two_bytes = fd.read(2)
        if first_two_bytes != b"\xFF\xD8":
            raise RuntimeError("Invalid jpeg file (unexpected header encoding)")
        header_data = [first_two_bytes]
        while True:  # while we have not reached the image data, keep ingesting...
            memblock = fd.read(chunk_size)
            if not memblock:
                raise RuntimeError("Invalid jpeg file (unexpected end-of-file in header)")
            # if found a fragmented SOS marker, just add the missing fragment
            if header_data[-1][-1] == b"\xFF" and memblock[0] == b"\xDA":
                header_data.append(memblock[: body_padding_size + 1])
                break
            # otherwise, if we hit the full SOS marker, add what's needed
            found_start_of_scan = memblock.find(b"\xFF\xDA")
            if found_start_of_scan != -1:
                if found_start_of_scan + 2 + body_padding_size > len(memblock):
                    header_data.append(memblock)
                    extra_to_read = (found_start_of_scan + 2 + body_padding_size) - len(memblock)
                    header_data.append(fd.read(extra_to_read))
                else:
                    header_data.append(memblock[: found_start_of_scan + 2 + body_padding_size])
                break
    header_data = b"".join(header_data + [b"\xFF\xD9"])
    return header_data


def decode_jpg(
    image: typing.Union[typing.AnyStr, pathlib.Path, bytes],
    to_bgr_format: bool = True,
    use_fast_upsample: bool = False,
    use_fast_dct: bool = False,
    scaling_factor: typing.Optional[typing.Tuple[int, int]] = None,
    crop_region: typing.Optional[typing.Tuple[int, int, int, int]] = None,
):
    """Decodes a JPEG from its (encoded) bytes data into an image.

    Note: when using the scaling factor, the expected image size might be off-by-one compared to
    when using a rounded or floored shape estimate; do not rely on those too much...

    Args:
        image: the path to the jpeg file to be parsed, or the encoded jpeg file data directly.
        to_bgr_format: defines whether to keep the image data in BGR format, or decode it to RGB.
        use_fast_upsample: allows faster decoding by skipping chrominance sample smoothing; see
            https://jpeg-turbo.dpldocs.info/libjpeg.turbojpeg.TJFLAG_FASTUPSAMPLE.html
        use_fast_dct: allows the use of the fastest DCT/IDCT algorithm available; see
            https://jpeg-turbo.dpldocs.info/libjpeg.turbojpeg.TJFLAG_FASTDCT.html
        scaling_factor: optional scaling factor that can be applied to resample the image as it is
            being decoded. Derived from IDCT scaling extensions in libjpeg-turbo decompressor.
            Only a handful of factors are supported, and only `(1, 2)` and `(1, 4)` are
            SIMD-enabled.
        crop_region: optional crop region parameters (top, left, height, width) to target only
            a region of the JPEG to be decoded and returned.

    Returns:
        The decoded jpeg image.
    """
    assert isinstance(image, (bytes, str, pathlib.Path)), f"invalid data type: {type(image)}"
    if isinstance(image, (str, pathlib.Path)):
        image = pathlib.Path(image)
        assert image.is_file(), f"missing jpeg image: {image}"
        with open(image, "rb") as fd:
            image = fd.read()
    assert isinstance(image, bytes)
    turbojpeg_handler = _get_turbojpeg_handler()
    flags = 0
    if use_fast_upsample:
        flags = flags | turbojpeg.TJFLAG_FASTUPSAMPLE
    if use_fast_dct:
        flags = flags | turbojpeg.TJFLAG_FASTDCT
    assert crop_region is None or scaling_factor is None, "cannot use scale+crop simultaneously"
    if crop_region is not None:
        assert isinstance(crop_region, typing.Sequence) and len(crop_region) == 4
        image = turbojpeg_handler.crop(
            image,
            x=crop_region[1],
            y=crop_region[0],
            w=crop_region[3],
            h=crop_region[2],
        )
    output = turbojpeg_handler.decode(
        image,
        pixel_format=turbojpeg.TJPF_BGR if to_bgr_format else turbojpeg.TJPF_RGB,
        scaling_factor=scaling_factor,
        flags=flags,
    )
    assert output.ndim == 3 and output.shape[-1] == 3
    return output


# check crop_multiple from turbojpeg @@@@@@
# also check regular crop and transform ops (flips, rotations)


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
        assert image.ndim == 2 or image.shape[2] in range(
            1, 5
        ), "cannot handle channel counts outside [1, 2, 3, 4] with opencv crop!"
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
                    offset[0] : (offset[0] + inters_coord.shape[0]),
                    offset[1] : (offset[1] + inters_coord.shape[1]),
                    ...,
                ] = image[inters_coord.tl[0] : inters_coord.br[0], inters_coord.tl[1] : inters_coord.br[1], ...]
            return crop_out
        # if all crop coordinates are in-bounds, we can get a view directly from the image
        crop_view = image[patch.tl[0] : patch.br[0], patch.tl[1] : patch.br[1], ...]
        if not force_copy:
            return crop_view
        return np.copy(crop_view)

    # regular handling (we crop along the spatial dimensions located at the end of the array)
    assert patch.ndim <= image.ndim, "patch dim count should be equal to or lower than image dimension count!"
    image_region = ssl4rs.utils.patch_coord.PatchCoord(top_left=[0] * patch.ndim, shape=image.shape[-patch.ndim :])
    crop_out_shape = tuple(image.shape[: -patch.ndim]) + patch.shape

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
            crop_out = torch.full(crop_out_shape, padding_val, dtype=image.dtype, device=image.device)
        else:
            crop_out = np.full(crop_out_shape, padding_val, dtype=image.dtype)
        offset = [inters_coord.tl[d] - patch.tl[d] for d in patch.dimrange]
        crop_inner_slice = tuple(
            [slice(None)] * (image.ndim - patch.ndim)
            + [slice(offset[d], offset[d] + inters_coord.shape[d]) for d in patch.dimrange]
        )
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
