import typing

import torch
from torchvision.transforms import InterpolationMode

import ssl4rs.data.transforms.geo.crop
import ssl4rs.utils.imgproc

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType

GeoCenterCrop = ssl4rs.data.transforms.geo.crop.GroundSamplingDistanceAwareCenterFixedCrop


class InstanceCenterCrop(GeoCenterCrop):
    """Implements a crop function that will always be centered on top of fMoW instance bboxes.

    See the fMoW data module / data parser classes for more info on that dataset, and the
    `GroundSamplingDistanceAwareCenterFixedCrop` on the transform.
    """

    def __init__(
        self,
        size: typing.Tuple[int, int],
        output_gsd: typing.Optional[float] = None,  # None = do not resize
        allow_auto_padding: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        add_params_to_batch_dict: bool = True,
    ):
        """Validates cropping settings.

        Args:
            size: expected output size of the crop, for each edge, after resizing. This is expected
                to be height first, width second, and in pixel values.
            output_gsd: expected output GSD value for the generated crops. The area in the provided
                arrays that corresponds to the output crop will vary based on this setting; a
                larger GSD can be obtained by upsampling a smaller area in the original array, and
                vice-versa. For example, with an original GSD of 2m and a target output GSD of
                4m, given a target output size of 256x256 pixels, we will fetch a 512x512 area in
                the original array, downsample it to 256x256 pixels, and return it. If this value
                is `None`, this function should be equivalent to a classic `CenterCrop` function.
            allow_auto_padding: defines to auto-pad input arrays when they are too small for the
                specified crop size/GSD values.
            interpolation: Desired interpolation enum defined by
                `torchvision.transforms.InterpolationMode`. Default is bilinear.
            antialias: defines whether to apply antialiasing. It only affects tensors when the
                bilinear or bicubic interpolation modes are selected, and is ignored otherwise.
            add_params_to_batch_dict: toggles whether crop parameters should be also added to the
                batch dictionary when one is provided. These parameters will be stored
        """
        super().__init__(
            size=size,
            output_gsd=output_gsd,
            target_key="image/rgb/jpg",
            gsd_key="image/rgb/gsd",
            allow_auto_padding=allow_auto_padding,
            interpolation=interpolation,
            antialias=antialias,
            add_params_to_batch_dict=add_params_to_batch_dict,
        )

    @staticmethod
    def get_params(
        input_array: torch.Tensor,
        gsd: float,
        size: typing.Tuple[int, int],
        output_gsd: typing.Optional[float],
        allow_auto_padding: bool,
        batch_dict: typing.Optional["BatchDictType"] = None,
    ) -> typing.Tuple[int, int, int, int, float]:
        """Returns the parameters for a crop-and-resize operation on the input array.

        Note: in contrast with the base class implementation, this derived version will make sure
        that the crop is centered on the bounding box of the instance located inside the image. If
        the crop clips the image boundaries, it will be slightly moved off-center to not have to
        add any padding to the image.

        Args:
            input_array: the input array from which a crop should be generated.
            gsd: the ground sample distance associated with the above array.
            size: expected output size of the crop, for each edge, after resizing. This is expected
                to be height first, width second, and in pixel values.
            output_gsd: expected output GSD value post-resizing. If `None`, no resizing should
                occur, and the `size` corresponds to the crop area in the original array directly.
            allow_auto_padding: defines to auto-pad input arrays when they are too small for the
                specified crop size/GSD values.
            batch_dict: input batch dictionary provided to the transform op; might be used in
                derived classes to process other batch attributes simultaneously.

        Returns:
            A tuple of parameters (top-idx, left-idx, height, width) to be passed to a crop
            function to return a crop, and the ground sampling distance (GSD) of the resulting
            crop (after resizing, if necessary).
        """
        # first, get the crop height/width and expected GSD from the base class impl
        _, _, crop_height, crop_width, output_gsd = GeoCenterCrop.get_params(
            input_array=input_array,
            gsd=gsd,
            size=size,
            output_gsd=output_gsd,
            allow_auto_padding=allow_auto_padding,
        )
        # now, relocate the crop to the bounding box location...
        assert batch_dict is not None, "missing batch dict (must be called with full fMoW batch data)"
        assert "image/rgb/bbox" in batch_dict, "missing bbox in batch dict?"
        bbox = batch_dict["image/rgb/bbox"]
        bbox_left, bbox_top, bbox_width, bbox_height = bbox
        bbox_center = (bbox_top + bbox_height // 2, bbox_left + bbox_width // 2)
        crop_top = bbox_center[0] - crop_height // 2
        crop_left = bbox_center[1] - crop_width // 2
        input_height, input_width = input_array.shape[-2:]
        # and move the crop within bounds, if needed
        if not allow_auto_padding:
            if crop_top < 0:
                crop_top = 0
            elif crop_top + crop_height > input_height:
                crop_top = input_height - crop_height
            if crop_left < 0:
                crop_left = 0
            elif crop_left + crop_width > input_width:
                crop_left = input_width - crop_width
            if (
                crop_top < 0
                or crop_left < 0
                or crop_top + crop_height > input_height
                or crop_left + crop_width > input_width
            ):
                raise ValueError(
                    f"cannot create a {crop_width}x{crop_height} crop "
                    f"in a {input_width}x{input_height} image without padding"
                )
        return crop_top, crop_left, crop_height, crop_width, output_gsd
