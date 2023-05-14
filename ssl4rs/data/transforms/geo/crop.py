import math
import typing

import torch
import torchvision.transforms
import torchvision.transforms.functional
from torchvision.transforms import InterpolationMode

import ssl4rs.utils.imgproc

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType


class GroundSamplingDistanceAwareRandomResizedCrop(torch.nn.Module):
    """Implements a random resized crop function that is aware of the ground sample distance.

    This "random resized crop" function will output crops with a constant shape. This shape is
    specified inside the constructor (`size`), but it will contain data from a region in the
    original array that might be larger or smaller than `size`, and that will have been resized.
    The target crop area inside the original array is the "random" part of this function: it will
    be sampled uniformly using the min/max GSD ratios (`gsd_ratios`) specified in the constructor,
    and applied to the real GSD value found at runtime. This real GSD value must be provided in the
    batch dictionary or as a call argument at runtime.

    The input array is assumed to have a [..., H, W] shape, where the leading dimensions will be
    left untouched, and where the last two dimensions are expected to be the array's height and
    width. The crop area will be determined based on these two dimensions and on the GSD, and the
    result will be resized to the target shape using torchvision's resize function.

    Note that like most torchvision transforms, this class inherits from `torch.nn.Module` in order
    to be compatible with torchscript (and to be compatible with accelerated transform pipelines).

    For more info on the arguments, see the `torchvision.transforms.RandomResizedCrop` class.
    """

    def __init__(
        self,
        size: typing.Tuple[int, int],
        gsd_ratios: typing.Tuple[float, float],
        target_key: typing.Optional[typing.AnyStr] = None,
        gsd_key: typing.Optional[typing.AnyStr] = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        add_params_to_batch_dict: bool = True,
    ):
        """Validates cropping settings.

        Args:
            size: expected output size of the crop, for each edge, after resizing. This is expected
                to be height first, width second, and in pixel values.
            gsd_ratios: expected min/max ratios to use when selecting a new ground sampling
                distance value. For example, with a GSD of 4m and a (min,max) ratio of (0.5, 2),
                a new GSD value will be uniformly sampled from the [2m, 8m] interval, and used
                to determine the overall shape of the region to crop and resize. A minimum ratio
                that is under 1 will lead to upsampling instead of downsampling, which may create
                blurry images. Use with caution!
            target_key: the key for the target tensor to be cropped, in case we process batch
                dictionaries. Note that we will replace the tensor by its cropped version in-place
                within the existing dictionary.
            gsd_key: the key for the ground sampling distance value to be fetched, in case we
                process batch dictionaries.
            interpolation: Desired interpolation enum defined by
                `torchvision.transforms.InterpolationMode`. Default is bilinear.
            antialias: defines whether to apply antialiasing. It only affects tensors when the
                bilinear or bicubic interpolation modes are selected, and is ignored otherwise.
            add_params_to_batch_dict: toggles whether crop parameters should be also added to the
                batch dictionary when one is provided. These parameters will be stored
        """
        super().__init__()
        assert len(size) == 2 and all(
            [isinstance(i, int) and i > 0 for i in size]
        ), f"invalid size tuple, should be (height, width) in pixels, got: {size}"
        assert len(gsd_ratios) == 2 and all(
            [isinstance(v, (int, float)) and v > 0 for v in gsd_ratios]
        ), f"invalid gsd ratios tuple, should be (min, max) floats, got: {gsd_ratios}"
        if isinstance(interpolation, int):
            interpolation = torchvision.transforms.functional._interpolation_modes_from_int(interpolation)
        assert interpolation in InterpolationMode, f"invalid interp mode: {interpolation}"
        self.size = size
        self.gsd_ratios = (float(min(gsd_ratios)), float(max(gsd_ratios)))
        self.target_key = target_key
        self.gsd_key = gsd_key
        self.interpolation = interpolation
        self.antialias = antialias
        self.add_params_to_batch_dict = add_params_to_batch_dict

    def forward(
        self,
        data: typing.Union[torch.Tensor, "BatchDictType"],
        gsd: typing.Optional[float] = None,
    ) -> typing.Union[typing.Tuple[torch.Tensor, float], "BatchDictType"]:
        """Crops the specified data array in the provided batch dictionary.

        Args:
            data: the loaded batch dictionary that contains an array to be cropped, or the tensor
                to be processed directly. If a batch dictionary is used, it will be UPDATED IN
                PLACE since we do not deep copy this reference.
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
        if isinstance(data, torch.Tensor):
            input_array = data
        assert isinstance(input_array, torch.Tensor), f"invalid input array type: {type(input_array)}"
        assert input_array.ndim >= 2, f"invalid input array ndim: {input_array.ndim}"
        if gsd is None:
            assert isinstance(data, dict), "must provide gsd as arg or provide it through batch dict!"
            assert self.gsd_key is not None, "missing 'gsd_key' attribute for batch dicts!"
            gsd = data[self.gsd_key]
        assert isinstance(gsd, float), f"invalid input gsd type: {type(gsd)}"
        crop_params = self.get_params(
            input_array=input_array,
            gsd=gsd,
            size=self.size,
            gsd_ratios=self.gsd_ratios,
            batch_dict=data if isinstance(data, dict) else None,
        )
        (crop_top_idx, crop_left_idx, crop_height, crop_width, output_gsd) = crop_params
        resized_crop = self.get_resized_crop(
            input_array=input_array,
            crop_top_idx=crop_top_idx,
            crop_left_idx=crop_left_idx,
            crop_height=crop_height,
            crop_width=crop_width,
            output_height=self.size[0],
            output_width=self.size[1],
            interpolation=self.interpolation,
            antialias=self.antialias,
            batch_dict=data if isinstance(data, dict) else None,
        )
        if isinstance(data, dict):
            data[self.target_key] = resized_crop
            data[self.gsd_key] = output_gsd
            if self.add_params_to_batch_dict:
                data[self.target_key + "/crop_params"] = crop_params
            return data
        else:
            return resized_crop, output_gsd

    @staticmethod
    def get_params(
        input_array: torch.Tensor,
        gsd: float,
        size: typing.Tuple[int, int],
        gsd_ratios: typing.Tuple[float, float],
        batch_dict: typing.Optional["BatchDictType"] = None,
    ) -> typing.Tuple[int, int, int, int, float]:
        """Returns the parameters for a crop-and-resize operation on the input array.

        Args:
            input_array: the input array from which a crop should be generated.
            gsd: the ground sample distance associated with the above array.
            size: expected output size of the crop, for each edge, after resizing. This is expected
                to be height first, width second, and in pixel values.
            gsd_ratios: expected min/max ratios to use when selecting a new ground sampling
                distance value. See constructor for more info.
            batch_dict: input batch dictionary provided to the transform op; might be used in
                derived classes to process other batch attributes simultaneously.

        Returns:
            A tuple of parameters (top-idx, left-idx, height, width) to be passed to a crop
            function to return a crop, and the ground sampling distance (GSD) of the resulting
            crop (after its future resizing).
        """
        assert input_array.ndim >= 2, "invalid input tensor ndim count"
        input_height, input_width = input_array.shape[-2:]
        assert input_height * input_width > 0, "invalid input tensor height/width dims"
        # the min/max request gsd values are scaled based on the ratios provided to the constructor
        min_requested_gsd, max_requested_gsd = gsd * gsd_ratios[0], gsd * gsd_ratios[1]
        # in order to check if we can match those requests, we need to check the max crop's GSD
        scale_max_h, scale_max_w = input_height / size[0], input_width / size[1]
        scale_max = min(scale_max_h, scale_max_w)
        max_possible_gsd = scale_max * gsd
        if max_possible_gsd < min_requested_gsd:
            raise ValueError(
                f"cannot create a random crop with predefined gsd ratio interval: {gsd_ratios}\n"
                f"\tinput array = ({input_height} x {input_width}) @ {gsd} meters\n"
                f"\tresizing a ({int(round(size[0] * scale_max))} x {int(round(size[1] * scale_max))})"
                f" crop to ({size[0]} x {size[1]})"
                f" would provide a GSD of {max_possible_gsd} meters\n"
                "\t...that would be less than the minimum requested GSD: "
                f"{gsd_ratios[0]} (min ratio) x {gsd} (input gsd) = {min_requested_gsd} meters"
            )
        max_gsd = min(max_requested_gsd, max_possible_gsd)
        # we sample an output gsd uniformly in the interval of requested-to-possible gsd values
        expected_output_gsd = torch.empty(1).uniform_(min_requested_gsd, max_gsd).item()
        # we compute the scale factor we'll have to use for the interpolation of the crop
        expected_scale = expected_output_gsd / gsd
        # ...and the size of the crop we'll be carving out of the input array
        crop_aspect_ratio = size[0] / size[1]
        crop_height = int(round(size[0] * expected_scale))
        crop_width = int(round(crop_height / crop_aspect_ratio))
        assert crop_height <= input_height and crop_width <= input_width
        # now that we have the target crop size (pre-interpolation), we need to decide where to place it
        top_left_coords = (
            torch.randint(0, input_height - crop_height + 1, size=(1,)).item(),
            torch.randint(0, input_width - crop_width + 1, size=(1,)).item(),
        )
        # finally, for the given target size, compute the (approx, wrt rounding errors) output gsd
        final_scale = ((crop_height / size[0]) + (crop_width / size[1])) / 2
        output_gsd = final_scale * gsd  # with non-square crops, this might be imperfect
        return (
            top_left_coords[0],
            top_left_coords[1],
            crop_height,
            crop_width,
            output_gsd,
        )

    @staticmethod
    def get_resized_crop(
        input_array: torch.Tensor,
        crop_top_idx: int,
        crop_left_idx: int,
        crop_height: int,
        crop_width: int,
        output_height: int,
        output_width: int,
        interpolation: InterpolationMode,
        antialias: bool,
        batch_dict: typing.Optional["BatchDictType"] = None,
    ) -> torch.Tensor:
        """Crops the given input array with the specified parameters.

        This function simply calls torchvision's `resized_crop` function under the hood, but may
        be overridden in derived classes.

        Args:
            input_array: the input array from which a crop should be generated.
            crop_top_idx: the index of the row that matches the crop's top left corner.
            crop_left_idx: the index of the column that matches the crop's top left corner.
            crop_height: the total height of the crop in the input array (in pixels).
            crop_width: the total width of the crop in the input array (in pixels).
            output_height: the expected height of the output (post-resize) array (in pixels).
            output_width: the expected width of the output (post-resize) array (in pixels).
            interpolation: Desired interpolation enum defined by
                `torchvision.transforms.InterpolationMode`. Default is bilinear.
            antialias: defines whether to apply antialiasing. It only affects tensors when the
                bilinear or bicubic interpolation modes are selected, and is ignored otherwise.
            batch_dict: input batch dictionary provided to the transform op; might be used in
                derived classes to process other batch attributes simultaneously.

        Returns:
            The tensor that corresponds to the specified cropped and resized (with a predetermined
            interpolation approach) area of the input array.
        """
        resized_crop = torchvision.transforms.functional.resized_crop(
            img=input_array,
            top=crop_top_idx,
            left=crop_left_idx,
            height=crop_height,
            width=crop_width,
            size=(output_height, output_width),  # noqa
            interpolation=interpolation,
            antialias=antialias,
        )
        return resized_crop

    def __repr__(self) -> str:
        out_str = self.__class__.__name__ + "("
        out_str += f"size={self.size}"
        out_str += f", gsd_ratios={self.gsd_ratios}"
        out_str += f", target_key={self.target_key}"
        out_str += f", gsd_key={self.gsd_key}"
        out_str += f", interpolation={self.interpolation}"
        out_str += f", antialias={self.antialias}"
        out_str += ")"
        return out_str


class GroundSamplingDistanceAwareCenterFixedCrop(torch.nn.Module):
    """Implements a center crop function that is aware of the ground sample distance.

    This "center fixed crop" function will output crops with a constant shape. This shape is
    specified inside the constructor (`size`), but it will contain data from a region in the
    original array that might be larger or smaller than `size`, and that will have been resized.
    The target crop area inside the original array is determined using the `output_gsd` value
    provided to the constructor. If the original GSD is larger or smaller than the target GSD, the
    crop area will be adjusted in consequence in order to always provide output crops with the
    requested shape and GSD values.

    The input array is assumed to have a [..., H, W] shape, where the leading dimensions will be
    left untouched, and where the last two dimensions are expected to be the array's height and
    width. The crop area will be determined based on these two dimensions and on the GSD, and the
    result will be resized to the target shape using torchvision's resize function (if needed).

    Note that like most torchvision transforms, this class inherits from `torch.nn.Module` in order
    to be compatible with torchscript (and to be compatible with accelerated transform pipelines).
    """

    def __init__(
        self,
        size: typing.Tuple[int, int],
        output_gsd: typing.Optional[float] = None,  # None = do not resize
        target_key: typing.Optional[typing.AnyStr] = None,
        gsd_key: typing.Optional[typing.AnyStr] = None,
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
            target_key: the key for the target tensor to be cropped, in case we process batch
                dictionaries. Note that we will replace the tensor by its cropped version in-place
                within the existing dictionary.
            gsd_key: the key for the ground sampling distance value to be fetched, in case we
                process batch dictionaries.
            allow_auto_padding: defines to auto-pad input arrays when they are too small for the
                specified crop size/GSD values.
            interpolation: Desired interpolation enum defined by
                `torchvision.transforms.InterpolationMode`. Default is bilinear.
            antialias: defines whether to apply antialiasing. It only affects tensors when the
                bilinear or bicubic interpolation modes are selected, and is ignored otherwise.
            add_params_to_batch_dict: toggles whether crop parameters should be also added to the
                batch dictionary when one is provided. These parameters will be stored
        """
        super().__init__()
        assert len(size) == 2 and all(
            [isinstance(i, int) and i > 0 for i in size]
        ), f"invalid size tuple, should be (height, width) in pixels, got: {size}"
        assert output_gsd is None or output_gsd > 0, f"invalid output gsd: {output_gsd}"
        if isinstance(interpolation, int):
            interpolation = torchvision.transforms.functional._interpolation_modes_from_int(interpolation)
        assert interpolation in InterpolationMode, f"invalid interp mode: {interpolation}"
        self.size = size
        self.output_gsd = float(output_gsd) if output_gsd is not None else None
        self.target_key = target_key
        self.gsd_key = gsd_key
        self.allow_auto_padding = allow_auto_padding
        self.interpolation = interpolation
        self.antialias = antialias
        self.add_params_to_batch_dict = add_params_to_batch_dict

    def forward(
        self,
        data: typing.Union[torch.Tensor, "BatchDictType"],
        gsd: typing.Optional[float] = None,
    ) -> typing.Union[typing.Tuple[torch.Tensor, float], "BatchDictType"]:
        """Crops the specified data array in the provided batch dictionary.

        Note: if the crop size/gsd combos require an area larger than the input array, padding will
        be added to that array automatically, but only if `allow_auto_padding=True`.

        Args:
            data: the loaded batch dictionary that contains an array to be cropped, or the tensor
                to be processed directly. If a batch dictionary is used, it will be UPDATED IN
                PLACE since we do not deep copy this reference.
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
        if isinstance(data, torch.Tensor):
            input_array = data
        assert isinstance(input_array, torch.Tensor), f"invalid input array type: {type(input_array)}"
        assert input_array.ndim >= 2, f"invalid input array ndim: {input_array.ndim}"
        if gsd is None:
            assert isinstance(data, dict), "must provide gsd as arg or provide it through batch dict!"
            assert self.gsd_key is not None, "missing 'gsd_key' attribute for batch dicts!"
            gsd = data[self.gsd_key]
        assert isinstance(gsd, float), f"invalid input gsd type: {type(gsd)}"
        crop_params = self.get_params(
            input_array=input_array,
            gsd=gsd,
            size=self.size,
            output_gsd=self.output_gsd,
            allow_auto_padding=self.allow_auto_padding,
            batch_dict=data if isinstance(data, dict) else None,
        )
        (crop_top_idx, crop_left_idx, crop_height, crop_width, output_gsd) = crop_params
        center_crop = self.get_center_crop(
            input_array=input_array,
            crop_top_idx=crop_top_idx,
            crop_left_idx=crop_left_idx,
            crop_height=crop_height,
            crop_width=crop_width,
            output_height=self.size[0],
            output_width=self.size[1],
            allow_auto_padding=self.allow_auto_padding,
            interpolation=self.interpolation,
            antialias=self.antialias,
            batch_dict=data if isinstance(data, dict) else None,
        )
        if isinstance(data, dict):
            data[self.target_key] = center_crop
            data[self.gsd_key] = output_gsd
            if self.add_params_to_batch_dict:
                data[self.target_key + "/crop_params"] = crop_params
            return data
        else:
            return center_crop, output_gsd

    @staticmethod
    def get_params(
        input_array: torch.Tensor,
        gsd: float,
        size: typing.Tuple[int, int],
        output_gsd: typing.Optional[float],
        allow_auto_padding: bool,
        batch_dict: typing.Optional["BatchDictType"] = None,
    ) -> typing.Tuple[int, int, int, int, float]:
        """Returns the parameters for a center crop operation on the input array.

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
        assert input_array.ndim >= 2, "invalid input tensor ndim count"
        input_height, input_width = input_array.shape[-2:]
        assert input_height * input_width > 0, "invalid input tensor height/width dims"

        if output_gsd is None or math.isclose(gsd, output_gsd):  # (no resizing)
            if not allow_auto_padding and (size[0] > input_height or size[1] > input_width):
                raise ValueError(
                    "crop size too large for input array!\n"
                    f"\tcrop size = {size}, input array = {(input_height, input_width)}"
                )
            crop_height, crop_width = size[0], size[1]
            output_gsd = gsd
        else:
            # in order to check if we can match the requested GSD, we need to check the max crop's GSD
            scale_max_h, scale_max_w = input_height / size[0], input_width / size[1]
            scale_max = min(scale_max_h, scale_max_w)
            max_possible_gsd = scale_max * gsd
            if max_possible_gsd < output_gsd and not allow_auto_padding:
                raise ValueError(
                    f"cannot create a random crop with output gsd: {output_gsd}\n"
                    f"\tinput array = ({input_height} x {input_width}) @ {gsd} meters\n"
                    f"\tresizing a ({int(round(size[0] * scale_max))} x {int(round(size[1] * scale_max))})"
                    f" crop to ({size[0]} x {size[1]})"
                    f" would provide a GSD of {max_possible_gsd} meters\n"
                    f"\t...that would be less than the requested GSD: {output_gsd} meters"
                )
            # compute the scale factor & size used for the interpolation of the crop
            scale = output_gsd / gsd
            crop_aspect_ratio = size[0] / size[1]
            crop_height = int(round(size[0] * scale))
            crop_width = int(round(crop_height / crop_aspect_ratio))

        crop_top = int(round((input_height - crop_height) / 2.0))
        crop_left = int(round((input_width - crop_width) / 2.0))
        # note: GSD might be imperfect if we do scaling w/ small crops
        return crop_top, crop_left, crop_height, crop_width, output_gsd

    @staticmethod
    def get_center_crop(
        input_array: torch.Tensor,
        crop_top_idx: int,
        crop_left_idx: int,
        crop_height: int,
        crop_width: int,
        output_height: int,
        output_width: int,
        allow_auto_padding: bool,
        interpolation: InterpolationMode,
        antialias: bool,
        batch_dict: typing.Optional["BatchDictType"] = None,
    ) -> torch.Tensor:
        """Crops the given input array with the specified parameters.

        This function simply calls torchvision's `crop` and `resize` functions under the hood, but
        may be overridden in derived classes.

        Args:
            input_array: the input array from which a crop should be generated.
            crop_top_idx: the index of the row that matches the crop's top left corner.
            crop_left_idx: the index of the column that matches the crop's top left corner.
            crop_height: the total height of the crop in the input array (in pixels).
            crop_width: the total width of the crop in the input array (in pixels).
            output_height: the expected height of the output (post-resize) array (in pixels).
            output_width: the expected width of the output (post-resize) array (in pixels).
            allow_auto_padding: defines to auto-pad input arrays when they are too small for the
                specified crop size/GSD values.
            interpolation: Desired interpolation enum defined by
                `torchvision.transforms.InterpolationMode`. Default is bilinear.
            antialias: defines whether to apply antialiasing. It only affects tensors when the
                bilinear or bicubic interpolation modes are selected, and is ignored otherwise.
            batch_dict: input batch dictionary provided to the transform op; might be used in
                derived classes to process other batch attributes simultaneously.

        Returns:
            The tensor that corresponds to the specified cropped and resized (with a predetermined
            interpolation approach) area of the input array.
        """
        crop = None
        if allow_auto_padding:
            # note: this will likely break torchscript compilations!
            crop_region = ssl4rs.utils.PatchCoord(
                top_left=(crop_top_idx, crop_left_idx),
                shape=(crop_height, crop_width),
            )
            image_region = ssl4rs.utils.PatchCoord(
                top_left=(0, 0),
                shape=(input_array.shape[-2], input_array.shape[-1]),
            )
            if crop_region not in image_region:
                crop = ssl4rs.utils.imgproc.flex_crop(
                    image=input_array,
                    patch=crop_region,
                )
        if crop is None:
            crop = torchvision.transforms.functional.crop(
                img=input_array,
                top=crop_top_idx,
                left=crop_left_idx,
                height=crop_height,
                width=crop_width,
            )
        assert crop.shape[-2] == crop_height and crop.shape[-1] == crop_width
        if crop_height != output_height or crop_width != output_width:
            crop = torchvision.transforms.functional.resize(
                img=crop,
                size=(output_height, output_width),  # noqa
                interpolation=interpolation,
                antialias=antialias,
            )
        return crop

    def __repr__(self) -> str:
        out_str = self.__class__.__name__ + "("
        out_str += f"size={self.size}"
        out_str += f", output_gsd={self.output_gsd}"
        out_str += f", target_key={self.target_key}"
        out_str += f", gsd_key={self.gsd_key}"
        out_str += f", allow_auto_padding={self.allow_auto_padding}"
        out_str += f", interpolation={self.interpolation}"
        out_str += f", antialias={self.antialias}"
        out_str += ")"
        return out_str


GSDAwareRandomResizedCrop = GroundSamplingDistanceAwareRandomResizedCrop
GSDAwareCenterFixedCrop = GroundSamplingDistanceAwareCenterFixedCrop
