import typing

import cv2 as cv
import numpy as np
import torch


def validate_inputs(class_label_map: np.ndarray, target_class_label: int, ignore_index: typing.Optional[int]):
    assert isinstance(class_label_map, np.ndarray)
    assert class_label_map.ndim == 2 and np.issubdtype(class_label_map.dtype, np.integer)
    assert isinstance(target_class_label, int) and target_class_label >= 0
    assert ignore_index is None or isinstance(ignore_index, int)


def generate_boundary_distance_mask(
        class_label_map: typing.Union[np.ndarray, torch.Tensor],
        target_class_label: int,
        ignore_index: typing.Optional[int] = None,
        pad_size: int = 1,  # Pad size to avoid boundary issues
) -> np.ndarray:
    """
    creates a mask from the boundary labels that for each pixel inside a contour/polygon,
    each pixel has a distance value from the nearest boundary
    Parameters
    ----------
    class_label_map
    target_class_label
    ignore_index
    pad_size

    Returns
    -------

    """
    if isinstance(class_label_map, torch.Tensor):
        class_label_map = class_label_map.detach().cpu().numpy()
    assert isinstance(class_label_map, np.ndarray)
    assert class_label_map.ndim == 2 and np.issubdtype(class_label_map.dtype, np.integer)
    assert isinstance(target_class_label, int) and target_class_label >= 0
    assert ignore_index is None or isinstance(ignore_index, int)

    # init relevant masks
    dontcare_mask = None
    if ignore_index is not None:
        dontcare_mask = np.asarray(class_label_map == ignore_index)

    # init Region of InterestN(RoI Mask)
    roi_mask = np.asarray(class_label_map == target_class_label)

    if not np.any(roi_mask):
        if dontcare_mask is None:
            return np.zeros(roi_mask.shape, dtype=np.int64)
        else:
            return dontcare_mask.astype(np.int64) * ignore_index

    roi_mask = roi_mask.astype(np.uint8) * 255  # to prep for opencv input

    # Pad the array to avoid boundary issues
    padded_roi_mask = np.pad(roi_mask, pad_size, mode='constant', constant_values=0)
    normalized_distances = np.zeros(padded_roi_mask.shape, dtype=np.float32)

    # find boundaries/countours
    # todo: do we need to dilate here? I dont think so tbh
    contours, _ = cv.findContours(
        image=padded_roi_mask,
        mode=cv.RETR_EXTERNAL,
        method=cv.CHAIN_APPROX_SIMPLE,
    )

    # process contours
    for contour in contours:
        contour_mask = np.zeros(padded_roi_mask.shape, dtype=np.uint8)
        cv.drawContours(
            image=contour_mask,
            contours=[contour],
            contourIdx=-1,
            color=255,
            thickness=cv.FILLED,
        )

        # Compute the centroid of the contour
        moments = cv.moments(contour)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            cx, cy = 0, 0

        # Compute the distance transform of the contour mask
        distance_transform = cv.distanceTransform(contour_mask, cv.DIST_L2, 3)

        # Identify interior pixels of the contour
        interior_indices = np.transpose(np.nonzero(contour_mask > 0))
        distances_to_centroid = np.sqrt((interior_indices[:, 1] - cx) ** 2 + (interior_indices[:, 0] - cy) ** 2)
        max_distance_to_centroid = distances_to_centroid.max()

        # add distances from boundary and centroid
        # todo: see if you want to divide instead or smth simimlar
        # maybe 1 - (dist from bound/ dist from centroid)
        if max_distance_to_centroid != 0:
            combined_distances = (1 * distance_transform[interior_indices[:, 0], interior_indices[:, 1]]
                                  + 0 * distances_to_centroid)
            max_combined_distance = combined_distances.max()
            normalized_distances[interior_indices[:, 0], interior_indices[:, 1]] = (
                    combined_distances / max_combined_distance
            )

    # Convert back from opencv format to our intended output format
    normalized_distances = normalized_distances[pad_size:-pad_size, pad_size:-pad_size]

    # bring back ignore index
    if dontcare_mask is not None:
        normalized_distances[dontcare_mask] = ignore_index

    return normalized_distances
