import typing

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

default_pad_color = (218, 224, 237)  # BGR color of the default padding to use in displays
default_text_color = (32, 26, 26)  # BGR color of the text to render in the displays


def fig2array(fig: plt.Figure) -> np.ndarray:
    """Transforms a pyplot figure into a numpy-compatible RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf


def get_displayable_image(
    array: np.ndarray,
    grayscale: bool = False,
    mask: typing.Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns a 'displayable' image that has been normalized and padded to three channels."""
    assert array.ndim in [2, 3], "unexpected input array dim count"
    if array.ndim == 3:  # if image is already 3-dim
        if array.shape[2] == 2:  # if we only have two channels
            array = np.dstack((array, array[:, :, 0]))  # add an extra channel to make it RGB
        elif array.shape[2] > 3:  # if we have more than three channels
            array = array[..., :3]  # just grab the first three, should be OK for display purposes
    image = cv.normalize(array, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U, mask)
    if grayscale and array.ndim == 3 and array.shape[2] != 1:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif not grayscale and (array.ndim == 2 or array.shape[2] == 1):
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return image  # this should be a 3-ch BGR image in HxWxC format


def get_displayable_heatmap(array: np.ndarray) -> np.ndarray:
    """Returns a 'displayable' array that has been min-maxed and mapped to BGR triplets."""
    if array.ndim != 2:
        array = np.squeeze(array)
    assert array.ndim == 2, "unexpected input array dim count"
    array = cv.normalize(array, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    heatmap = cv.applyColorMap(array, cv.COLORMAP_VIRIDIS)
    return heatmap


def add_heatmap_on_base_image(
    base_image: np.ndarray,
    heat_values: np.ndarray,
    blend_weight: float = 0.7,  # blending weight for the heatmap (foreground), in [0, 1]
) -> np.ndarray:
    """Adds a colorful heatmap on top of a base image."""
    assert base_image.ndim == 2 or base_image.ndim == 3
    if base_image.ndim == 2 or (base_image.ndim == 3 and base_image.shape[-1] == 1):
        base_image = cv.cvtColor(base_image, cv.COLOR_GRAY2BGR)
    assert base_image.ndim == 3 and base_image.shape[-1] == 3
    assert base_image.dtype == np.uint8
    heatmap = get_displayable_heatmap(heat_values)
    assert 0 < blend_weight < 1
    image = cv.addWeighted(base_image, 1 - blend_weight, heatmap, blend_weight, 0)
    return image


def get_cv_colormap_from_class_color_map(
    class_color_list: typing.Sequence[
        typing.Union[np.ndarray, typing.Tuple[int, int, int]]
    ],
    default_color: np.ndarray = np.array([0xFF, 0xFF, 0xFF], dtype=np.uint8),
) -> np.ndarray:
    """Converts a list of color triplets into a 256-len array of color triplets for OpenCV."""
    assert (
        len(class_color_list) < 256
    ), "invalid class color list (should be less than 256 classes)"
    out_color_array = []
    for label_idx in range(256):
        if label_idx < len(class_color_list):
            assert (
                len(class_color_list[label_idx]) == 3
            ), f"invalid triplet for idx={label_idx}"
            out_color_array.append(
                class_color_list[label_idx][::-1]
            )  # RGB to BGR for opencv
        else:
            out_color_array.append(default_color)
    return np.asarray(out_color_array).astype(np.uint8)


def apply_cv_colormap(
    class_idx_map: np.ndarray,
    color_map: np.ndarray,
) -> np.ndarray:
    """Applies the OpenCV color map onto the class label index map, channel by channel."""
    assert np.issubdtype(class_idx_map.dtype, np.integer)
    min_label_idx, max_label_idx, _, _ = cv.minMaxLoc(class_idx_map)
    assert min_label_idx >= -1 and max_label_idx < 255, "invalid label index range"
    class_idx_map = class_idx_map.astype(np.uint8)  # dontcare label becomes 255
    assert color_map.shape == (256, 3), "invalid color map shape"
    output = np.zeros((*class_idx_map.shape, 3), dtype=np.uint8)
    for ch_idx in range(3):
        output[..., ch_idx] = cv.applyColorMap(class_idx_map, color_map[..., ch_idx])[
            ..., ch_idx
        ]
    return output


def get_html_color_code(rgb: typing.Tuple[int, int, int]) -> str:
    """Returns the HTML (hex) color code given a tuple of R,G,B values."""
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def resize_nn(
    image: np.ndarray,
    zoom_factor: float = 1,
) -> np.ndarray:
    """Performs nearest-neighbor resampling of a BGR image with a given scaling factor."""
    assert image.ndim == 3
    if zoom_factor != 1:
        return cv.resize(
            src=image,
            dsize=(-1, -1),
            fx=zoom_factor, fy=zoom_factor,
            interpolation=cv.INTER_NEAREST,
        )
    return image


def add_subtitle_to_image(
    image: np.ndarray,
    subtitle: str,
    extra_border_size: int = 0,
    extra_subtitle_padding: int = 2,
    scale: float = 2.0,
    thickness: typing.Optional[int] = 2,
    pad_color: typing.Tuple[int, int, int] = default_pad_color,
) -> np.ndarray:
    """Renders an image with a small subtitle string underneath it."""
    assert image.ndim == 3
    text_size, baseline = cv.getTextSize(
        text=subtitle,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=scale,
        thickness=thickness,
    )
    if text_size[0] > image.shape[1]:
        extra_x_padding = (text_size[0] - image.shape[1]) // 2
    else:
        extra_x_padding = 0
    image = cv.copyMakeBorder(
        src=image,
        top=extra_border_size,
        bottom=(text_size[1] + extra_border_size + extra_subtitle_padding * 2),
        left=extra_border_size + extra_x_padding,
        right=extra_border_size + extra_x_padding,
        borderType=cv.BORDER_CONSTANT,
        value=pad_color,
    )
    out_x = int(0.5 * image.shape[1] - text_size[0] // 2)
    out_y = image.shape[0] - extra_subtitle_padding
    cv.putText(
        img=image,
        text=subtitle,
        org=(out_x, out_y),  # X, Y, as expected by opencv
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=scale,
        color=default_text_color,
        thickness=thickness,
        lineType=None,
        bottomLeftOrigin=None,
    )
    return image


def append_image_to_string(
    image: np.ndarray,
    string: str,
    extra_border_size: int = 0,
    extra_string_padding: int = 20,
    scale: float = 3.0,
    thickness: typing.Optional[int] = 3,
    pad_color: typing.Tuple[int, int, int] = default_pad_color,
) -> np.ndarray:
    """Renders a text string followed by an image on its right with optional padding."""
    assert image.ndim == 3
    text_size, baseline = cv.getTextSize(
        text=string,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=scale,
        thickness=thickness,
    )
    assert text_size[1] < image.shape[0]
    image = cv.copyMakeBorder(
        src=image,
        top=extra_border_size,
        bottom=extra_border_size,
        left=extra_border_size + text_size[0] + extra_string_padding * 2,
        right=extra_border_size,
        borderType=cv.BORDER_CONSTANT,
        value=pad_color,
    )
    out_x = extra_string_padding
    out_y = (image.shape[0] + text_size[1]) // 2
    cv.putText(
        img=image,
        text=string,
        org=(out_x, out_y),  # X, Y, as expected by opencv
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=scale,
        color=default_text_color,
        thickness=thickness,
        lineType=None,
        bottomLeftOrigin=None,
    )
    return image


def vconcat_and_pad_if_needed(
    images: typing.Sequence[np.ndarray],
    extra_border_padding: int = 10,
    extra_vertical_padding: int = 20,
    pad_color: typing.Tuple[int, int, int] = default_pad_color,
) -> np.ndarray:
    """Concatenates a list of images vertically with optional auto-padding."""
    assert all([img.ndim == 3 for img in images])
    max_width = max([img.shape[1] for img in images])
    padded_images = []
    for img_idx, img in enumerate(images):
        req_padding = max_width - img.shape[1]
        padded_images.append(
            cv.copyMakeBorder(
                src=img,
                top=extra_border_padding,
                bottom=extra_border_padding + (extra_vertical_padding if img_idx < len(images) - 1 else 0),
                left=extra_border_padding + req_padding // 2,
                right=extra_border_padding + (req_padding - req_padding // 2),
                borderType=cv.BORDER_CONSTANT,
                value=pad_color,
            )
        )
    return cv.vconcat(padded_images)


def hconcat_and_pad_if_needed(
    images: typing.Sequence[np.ndarray],
    extra_border_padding: int = 10,
    extra_horizontal_padding: int = 20,
    pad_color: typing.Tuple[int, int, int] = default_pad_color,
) -> np.ndarray:
    """Concatenates a list of images horizontally with optional auto-padding."""
    assert all([img.ndim == 3 for img in images])
    max_height = max([img.shape[0] for img in images])
    padded_images = []
    for img_idx, img in enumerate(images):
        req_padding = max_height - img.shape[0]
        padded_images.append(
            cv.copyMakeBorder(
                src=img,
                top=extra_border_padding + req_padding // 2,
                bottom=extra_border_padding + (req_padding - req_padding // 2),
                left=extra_border_padding,
                right=extra_border_padding + (extra_horizontal_padding if img_idx < len(images) - 1 else 0),
                borderType=cv.BORDER_CONSTANT,
                value=pad_color,
            )
        )
    return cv.hconcat(padded_images)
