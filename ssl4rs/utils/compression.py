"""Contains utilities related to raw data compression (lz4, jpg, png, ...)."""
import copy
import typing

import cv2 as cv
import lz4.frame
import numpy as np

no_compression_flags = [
    "None", "none", "raw", "", None,
]
"""List of flags used to specify when NOT to compress data."""

supported_compression_flags = [
    *no_compression_flags,
    "lz4",  # lossless, generic (byte-sequence-compatible), fast, but not great compression
    "jpg",  # lossy (by default), only for RGB images, fast, very good compression
    "png",  # lossless, only for RGB or RGBA images, medium speed, good compression for graphics
    # TODO: add turbojpeg encoding/decoding?
]
"""List of flags for the currently supported data compression approaches."""


def encode(
    data: np.ndarray,
    approach: typing.AnyStr = "lz4",
    **kwargs,
) -> typing.Union[bytes, bytearray, np.ndarray]:
    """Encodes a numpy array using a given encoding approach.

    Args:
        data: the numpy array to encode.
        approach: the encoding approach to use, see the `supported_compression_flags` list.
        kwargs: extra keyword arguments that will be forwarded to the compression implementation.

    Returns:
        The compressed (if at all) result. The type will depend on the approach, as
        different implementations are used under the hood.
    """

    assert approach in supported_compression_flags, f"unexpected approach '{approach}'"
    if approach in no_compression_flags:
        assert not kwargs
        return data
    elif approach == "lz4":
        return lz4.frame.compress(data, **kwargs)
    elif approach == "jpg" or approach == "jpeg":
        ret, buf = cv.imencode(".jpg", data, **kwargs)  # type: ignore
    elif approach == "png":
        ret, buf = cv.imencode(".png", data, **kwargs)  # type: ignore
    else:
        raise NotImplementedError
    assert ret, "failed to encode data"
    return buf


def decode(
    data: typing.Union[bytes, bytearray, np.ndarray],
    approach: typing.AnyStr = "lz4",
    **kwargs,
) -> np.ndarray:
    """Decodes a binary array using a given coding approach.

    Args:
        data: the binary array to decode.
        approach: the encoding approach to reverse, see the `supported_compression_flags` list.

    Returns:
        The decompressed (if at all) data, in its originally-encoded numpy array format.
    """
    assert approach in supported_compression_flags, f"unexpected approach '{approach}'"
    if approach in no_compression_flags:
        assert not kwargs
        return data
    elif approach == "lz4":
        return lz4.frame.decompress(data, **kwargs)
    elif approach in ["jpg", "jpeg", "png"]:
        kwargs = copy.deepcopy(kwargs)
        if isinstance(kwargs["flags"], str):  # required arg by opencv
            kwargs["flags"] = eval(kwargs["flags"])
        return cv.imdecode(data, **kwargs)  # type: ignore
    else:
        raise NotImplementedError
