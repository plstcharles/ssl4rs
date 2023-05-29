import functools
import io
import typing

import torch
import torchvision.transforms.functional
from torchvision.transforms import InterpolationMode

import ssl4rs.data.transforms.geo.crop
import ssl4rs.utils.imgproc

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType

GeoCenterCrop = ssl4rs.data.transforms.geo.crop.GroundSamplingDistanceAwareCenterFixedCrop


def _decode_crop_and_resize(
    crop_param_getter: typing.Callable,
    input_array: bytes,
    input_gsd: float,
    output_size: typing.Tuple[int, int],
    min_input_size: typing.Optional[typing.Tuple[int, int]],
    use_fast_upsample: bool,
    use_fast_dct: bool,
    interpolation: InterpolationMode,
    antialias: typing.Optional[typing.Union[str, bool]],
) -> typing.Tuple[torch.Tensor, typing.Tuple]:  # output = crop, crop_params
    """Decodes a given jpeg, crops a specified region from it, and returns it after resizing."""
    # this implementation will pad the input image towards the bottom-right if necessary
    im_height, im_width = ssl4rs.utils.imgproc.get_image_shape_from_file(
        file_path_or_data=io.BytesIO(input_array),
        with_turbojpeg=False,
    )
    orig_im_height, orig_im_width = im_height, im_width
    im_height = max(im_height, min_input_size[0] if min_input_size else im_height)
    im_width = max(im_width, min_input_size[1] if min_input_size else im_width)
    crop_params = crop_param_getter(
        input_array=torch.empty(  # for size lookups, create a no-alloc array
            (0, 3, im_height, im_width),
            dtype=torch.uint8,
        ),
        gsd=input_gsd,
        size=output_size,
    )
    (crop_top_idx, crop_left_idx, crop_height, crop_width, output_gsd) = crop_params
    assert crop_top_idx >= 0 and crop_left_idx >= 0, "crop top-left should always be in-bounds"
    pad_rows = pad_cols = 0
    need_padding = crop_top_idx + crop_height > orig_im_height or crop_left_idx + crop_width > orig_im_width
    if need_padding:
        assert crop_top_idx + crop_height <= im_height
        assert crop_left_idx + crop_width <= im_width
        pad_rows = (crop_top_idx + crop_height) - orig_im_height
        pad_cols = (crop_left_idx + crop_width) - orig_im_width
    crop = ssl4rs.utils.imgproc.decode_with_turbo_jpeg(
        image=input_array,
        to_bgr_format=False,
        use_fast_upsample=use_fast_upsample,
        use_fast_dct=use_fast_dct,
        crop_region=(crop_top_idx, crop_left_idx, crop_height - pad_rows, crop_width - pad_cols),
    )
    assert crop.ndim == 3 and crop.shape[2] == 3
    crop = ssl4rs.data.transforms.pad_if_needed(
        input_tensor=torchvision.transforms.functional.to_tensor(crop),
        min_height=crop_height,
        min_width=crop_width,
    )[:, 0:crop_height, 0:crop_width]
    resized_crop = torchvision.transforms.functional.resize(
        img=crop,
        size=output_size,  # noqa
        interpolation=interpolation,
        antialias=antialias,
    )
    return resized_crop, crop_params


class JPEGDecoderWithInstanceCenterCrop(GeoCenterCrop):
    """Implements a crop function that will always be centered on top of fMoW instance bboxes.

    Also expected to get still-compressed JPEG images as input.

    See the fMoW data module / data parser classes for more info on that dataset, and the
    `GroundSamplingDistanceAwareCenterFixedCrop` on the transform.
    """

    def __init__(
        self,
        size: typing.Tuple[int, int],
        output_gsd: typing.Optional[float] = None,  # None = auto-resize from bbox crop
        use_fast_upsample: bool = False,
        use_fast_dct: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        add_params_to_batch_dict: bool = True,
    ):
        """Validates cropping settings.

        Args:
            size: expected output size of the crop, for each edge, after resizing. This is expected
                to be height first, width second, and in pixel values.
            output_gsd: expected output GSD value for the generated crops. The total area in the
                input images that corresponds to the output crop will vary based on this setting; a
                large GSD is obtained by upsampling a small area in the input image, and a small
                GSD is obtained by downsampling a large area in the same image. If this value
                is `None`, the entire bbox of the instance will be resized to the crop size.
            use_fast_upsample: allows faster decoding by skipping chrominance sample smoothing; see
                https://jpeg-turbo.dpldocs.info/libjpeg.turbojpeg.TJFLAG_FASTUPSAMPLE.html
            use_fast_dct: allows the use of the fastest DCT/IDCT algorithm available; see
                https://jpeg-turbo.dpldocs.info/libjpeg.turbojpeg.TJFLAG_FASTDCT.html
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
            target_key="image",
            gsd_key="gsd",
            allow_auto_padding=True,
            interpolation=interpolation,
            antialias=antialias,
            add_params_to_batch_dict=add_params_to_batch_dict,
        )
        self.use_fast_upsample = use_fast_upsample
        self.use_fast_dct = use_fast_dct

    @staticmethod
    def _get_largest_rect_in_bbox(
        bbox: typing.Tuple[int, int, int, int],  # bbox_left, bbox_top, bbox_width, bbox_height
        expected_aspect_ratio: float,
    ) -> typing.Tuple[int, int, int, int]:  # out_left, out_top, out_width, out_height
        """Returns the params of the largest aspect-ratio-respecting ROI in the bbox."""
        bbox_left, bbox_top, bbox_width, bbox_height = bbox
        bbox_aspect_ratio = bbox_width / bbox_height
        if bbox_aspect_ratio > expected_aspect_ratio:
            out_height = bbox_height
            out_width = int(round(out_height * expected_aspect_ratio))
        else:
            out_width = bbox_width
            out_height = int(round(out_width / expected_aspect_ratio))
        out_top = int(round(bbox_top + (bbox_height - out_height) / 2))
        out_left = int(round(bbox_left + (bbox_width - out_width) / 2))
        return out_left, out_top, out_width, out_height

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
            output_gsd: expected output GSD value post-resizing. If `None`, the full instance
                crop is expected to be used for the crop.
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
        assert batch_dict is not None, "missing batch dict (must be called with full fMoW batch data)"
        assert "bbox" in batch_dict, "missing bbox in batch dict?"
        bbox = batch_dict["bbox"]
        bbox_left, bbox_top, bbox_width, bbox_height = bbox
        if output_gsd is None:
            # in this case, get a fixed-aspect-ratio rect from the bbox region
            crop_params = JPEGDecoderWithInstanceCenterCrop._get_largest_rect_in_bbox(
                bbox=bbox,
                expected_aspect_ratio=size[1] / size[0],
            )
            crop_left, crop_top, crop_width, crop_height = crop_params
            scale = ((crop_width / size[1]) + (crop_height / size[0])) / 2
            output_gsd = gsd * scale
        else:
            # otherwise, just get a fixed-size crop centered on the bbox center itself
            assert allow_auto_padding
            _, _, crop_height, crop_width, output_gsd = GeoCenterCrop.get_params(
                input_array=input_array,
                gsd=gsd,
                size=size,
                output_gsd=output_gsd,
                allow_auto_padding=allow_auto_padding,
            )
            # now, relocate the crop to the bounding box location...
            bbox_center = (bbox_top + bbox_height // 2, bbox_left + bbox_width // 2)
            crop_top = bbox_center[0] - crop_height // 2
            crop_left = bbox_center[1] - crop_width // 2
        return crop_top, crop_left, crop_height, crop_width, output_gsd

    def forward(
        self,
        data: typing.Union[torch.Tensor, "BatchDictType"],
        gsd: typing.Optional[float] = None,
    ) -> typing.Union[typing.Tuple[torch.Tensor, float], "BatchDictType"]:
        """Returns a crop inside the not-yet-loaded jpeg data provided via a batch dict.

        Args:
            data: the loaded batch dictionary that contains a JPEG image to be decoded and cropped.
                The batch dictionary will be UPDATED IN PLACE since we do not deep copy this ref.
            gsd: the current ground sample distance associated with the array to be cropped. If it
                is not directly specified via this argument, the `gsd_key` attribute will be used
                to fetch it from inside the batch dictionary.

        Returns:
            The cropped tensor and its GSD value, if only a tensor was provided, or the reference
            to the same (but updated) batch dictionary that contains the crop + GSD value.
        """
        if isinstance(data, dict):
            assert self.target_key is not None, "missing 'target_key' attribute for batch dicts!"
            input_array = data[self.target_key]
        else:
            input_array = data
        assert isinstance(input_array, bytes), f"unexpected encoded jpeg type: {type(input_array)}"
        if gsd is None:
            assert isinstance(data, dict), "must provide gsd as arg or provide it through batch dict!"
            assert self.gsd_key is not None, "missing 'gsd_key' attribute for batch dicts!"
            gsd = data[self.gsd_key]
        assert isinstance(gsd, float), f"invalid input gsd type: {type(gsd)}"
        resized_crop, crop_params = _decode_crop_and_resize(
            crop_param_getter=functools.partial(
                self.get_params,
                output_gsd=self.output_gsd,
                allow_auto_padding=self.allow_auto_padding,
                batch_dict=data if isinstance(data, dict) else None,
            ),
            input_array=input_array,
            input_gsd=gsd,
            output_size=self.size,
            min_input_size=None,
            use_fast_upsample=self.use_fast_upsample,
            use_fast_dct=self.use_fast_dct,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        output_gsd = crop_params[-1]  # (crop_top_idx, crop_left_idx, crop_height, crop_width, output_gsd)
        if isinstance(data, dict):
            data[self.target_key] = resized_crop
            data[self.gsd_key] = output_gsd
            if self.add_params_to_batch_dict:
                data[self.target_key + "/crop_params"] = crop_params
            return data
        else:
            return resized_crop, output_gsd


class JPEGDecoderWithRandomResizedCrop(
    ssl4rs.data.transforms.geo.crop.GroundSamplingDistanceAwareRandomResizedCrop,
):
    """Derived version of the GSD-aware random resized crop that decodes JPEGs at the same time."""

    def __init__(
        self,
        min_input_size: typing.Tuple[int, int],
        output_size: typing.Tuple[int, int],
        gsd_ratios: typing.Tuple[float, float],
        use_fast_upsample: bool = False,
        use_fast_dct: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        add_params_to_batch_dict: bool = True,
    ):
        """Validates cropping settings.

        Args:
            min_input_size: minimum size of the input images used for cropping; if the input shape
                is smaller than this, it will be padded to match it. This is expected to be height
                first, width second, and in pixel values.
            output_size: expected output size of the crop, for each edge, after resizing. This is
                expected to be height first, width second, and in pixel values.
            gsd_ratios: expected min/max ratios to use when selecting a new ground sampling
                distance value. For example, with a GSD of 4m and a (min,max) ratio of (0.5, 2),
                a new GSD value will be uniformly sampled from the [2m, 8m] interval, and used
                to determine the overall shape of the region to crop and resize. A minimum ratio
                that is under 1 will lead to upsampling instead of downsampling, which may create
                blurry images. Use with caution!
            use_fast_upsample: allows faster decoding by skipping chrominance sample smoothing; see
                https://jpeg-turbo.dpldocs.info/libjpeg.turbojpeg.TJFLAG_FASTUPSAMPLE.html
            use_fast_dct: allows the use of the fastest DCT/IDCT algorithm available; see
                https://jpeg-turbo.dpldocs.info/libjpeg.turbojpeg.TJFLAG_FASTDCT.html
            interpolation: Desired interpolation enum defined by
                `torchvision.transforms.InterpolationMode`. Default is bilinear.
            antialias: defines whether to apply antialiasing. It only affects tensors when the
                bilinear or bicubic interpolation modes are selected, and is ignored otherwise.
            add_params_to_batch_dict: toggles whether crop parameters should be also added to the
                batch dictionary when one is provided. These parameters will be stored
        """
        super().__init__(
            size=output_size,
            gsd_ratios=gsd_ratios,
            target_key="image",
            gsd_key="gsd",
            interpolation=interpolation,
            antialias=antialias,
            add_params_to_batch_dict=add_params_to_batch_dict,
        )
        self.min_input_size = min_input_size
        self.use_fast_upsample = use_fast_upsample
        self.use_fast_dct = use_fast_dct

    def forward(
        self,
        data: typing.Union[torch.Tensor, "BatchDictType"],
        gsd: typing.Optional[float] = None,
    ) -> typing.Union[typing.Tuple[torch.Tensor, float], "BatchDictType"]:
        """Returns a crop inside the not-yet-loaded jpeg data provided via a batch dict.

        Args:
            data: the loaded batch dictionary that contains a JPEG image to be decoded and cropped.
                The batch dictionary will be UPDATED IN PLACE since we do not deep copy this ref.
            gsd: the current ground sample distance associated with the array to be cropped. If it
                is not directly specified via this argument, the `gsd_key` attribute will be used
                to fetch it from inside the batch dictionary.

        Returns:
            The cropped tensor and its GSD value, if only a tensor was provided, or the reference
            to the same (but updated) batch dictionary that contains the crop + GSD value.
        """
        if isinstance(data, dict):
            assert self.target_key is not None, "missing 'target_key' attribute for batch dicts!"
            input_array = data[self.target_key]
        else:
            input_array = data
        assert isinstance(input_array, bytes), f"unexpected encoded jpeg type: {type(input_array)}"
        if gsd is None:
            assert isinstance(data, dict), "must provide gsd as arg or provide it through batch dict!"
            assert self.gsd_key is not None, "missing 'gsd_key' attribute for batch dicts!"
            gsd = data[self.gsd_key]
        assert isinstance(gsd, float), f"invalid input gsd type: {type(gsd)}"
        resized_crop, crop_params = _decode_crop_and_resize(
            crop_param_getter=functools.partial(
                self.get_params,
                gsd_ratios=self.gsd_ratios,
                batch_dict=data if isinstance(data, dict) else None,
            ),
            input_array=input_array,
            input_gsd=gsd,
            output_size=self.size,
            min_input_size=self.min_input_size,
            use_fast_upsample=self.use_fast_upsample,
            use_fast_dct=self.use_fast_dct,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        output_gsd = crop_params[-1]  # (crop_top_idx, crop_left_idx, crop_height, crop_width, output_gsd)
        if isinstance(data, dict):
            data[self.target_key] = resized_crop
            data[self.gsd_key] = output_gsd
            if self.add_params_to_batch_dict:
                data[self.target_key + "/crop_params"] = crop_params
            return data
        else:
            return resized_crop, output_gsd


# @@@@ TODO: add another random resized crop with 1/2 and 1/4 downsampling via turbojpeg?
