# @@@@@@@@ TODO import the tests for these functions
import typing

import cv2 as cv
import numpy as np
import torch


def generate_boundary_mask_from_class_label_map(
    class_label_map: typing.Union[np.ndarray, torch.Tensor],
    target_class_label: int,
    ignore_index: typing.Optional[int] = None,
    contour_dilate_kernel_size: typing.Optional[typing.Tuple[int, int]] = None,
) -> np.ndarray:
    """Generates a boundary mask from all regions of interest inside a given class label map.

    We will use the specified target class label to extract the regions of interest from the class
    label map (which should be a 2-dim, HxW array). If a label index to ignore is specified, we will
    make sure that the pixels assigned to that index will keep this assignment in the output
    boundary mask.

    The output will be provided as a 2-dim, HxW, int64 array with background (non-boundary) pixels
    set to zero, boundary pixels set to one, and ignored pixels set to the ignored index value.
    """
    if isinstance(class_label_map, torch.Tensor):
        class_label_map = class_label_map.detach().cpu().numpy()
    assert isinstance(class_label_map, np.ndarray)
    assert class_label_map.ndim == 2 and np.issubdtype(class_label_map.dtype, np.integer)
    assert isinstance(target_class_label, int) and target_class_label >= 0
    assert ignore_index is None or isinstance(ignore_index, int)
    if contour_dilate_kernel_size is not None:
        assert all([isinstance(i, int) and i > 0 for i in contour_dilate_kernel_size])
    dontcare_mask = None
    if ignore_index is not None:
        dontcare_mask = np.asarray(class_label_map == ignore_index)
    roi_mask = np.asarray(class_label_map == target_class_label)
    if not np.any(roi_mask):
        # no contours to be found, quick exit
        if dontcare_mask is None:
            return np.zeros(roi_mask.shape, dtype=np.int64)
        else:
            return dontcare_mask.astype(np.int64) * ignore_index
    roi_mask = roi_mask.astype(np.uint8) * 255  # to prep for opencv input
    contours, _ = cv.findContours(
        image=roi_mask,
        mode=cv.RETR_EXTERNAL,
        method=cv.CHAIN_APPROX_SIMPLE,
    )
    output_mask = np.zeros(roi_mask.shape, dtype=np.uint8)  # temporarily uint8, for opencv
    cv.drawContours(
        image=output_mask,
        contours=contours,
        contourIdx=-1,  # draw all contours
        color=255,
        thickness=1,
    )
    if contour_dilate_kernel_size is not None:
        dilate_struct_elem = cv.getStructuringElement(cv.MORPH_CROSS, contour_dilate_kernel_size)
        output_mask = cv.dilate(output_mask, dilate_struct_elem, iterations=1)
    # convert back from opencv format to our intended output format
    output_mask = output_mask.astype(np.int64) // 255
    if dontcare_mask is not None:
        output_mask[dontcare_mask] = ignore_index
    return output_mask
