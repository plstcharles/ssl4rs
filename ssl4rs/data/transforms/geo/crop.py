import typing

import torch
import torchvision.transforms
import torchvision.transforms.functional
from torchvision.transforms import InterpolationMode

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType


class GroundSamplingDistanceAwareRandomResizedCrop(torch.nn.Module):
    """Implements a crop function that is aware of a geospatial array's ground sample distance.

    This function works a bit like a random cropper, but it will consider a min/max GSD interval
    instead of a min/max scale factor to identify the size of the input region to crop and resize.
    The GSD info must be provided in the batch dictionary or as a call argument at runtime.

    The input array is assumed to have a [..., H, W] shape, where the leading dimensions will be
    left untouched, and where the last two dimensions are expected to be the array height / width.
    The crop area will be determined based on these two dimensions and on the GSD, and the result
    will be resized to the target shape.

    Note that like most torchvision transforms, this class inherits from `torch.nn.Module` in order
    to be compatible with torchscript (and to be compatible with accelerated transform pipelines).

    For more info on the arguments, see the `torchvision.transforms.RandomResizedCrop` class.

    TODO @@@@: validate that this is compilable w/ torchscript?
    """

    def __init__(
        self,
        size: typing.Tuple[int, int],
        gsd_ratios: typing.Tuple[float, float],
        target_key: typing.Optional[typing.AnyStr] = None,
        gsd_key: typing.Optional[typing.AnyStr] = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ):
        """Validates cropping settings.

        Args:
            size: expected output size of the crop, for each edge, after resizing. This is expected
                to be height first, width second, and pixel values.
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
        assert isinstance(input_array, torch.Tensor), f"invalid input data type: {type(data)}"
        assert input_array.ndim >= 2, f"invalid input array ndim: {input_array.ndim}"
        if gsd is None:
            assert isinstance(data, dict), "must provide gsd as arg or provide it through batch dict!"
            assert self.gsd_key is not None, "missing 'gsd_key' attribute for batch dicts!"
            gsd = data[self.gsd_key]
        assert isinstance(gsd, float), f"invalid input gsd type: {type(gsd)}"
        (target_top_idx, target_left_idx, target_height, target_width, output_gsd) = self.get_params(
            input_array=input_array,
            gsd=gsd,
            size=self.size,
            gsd_ratios=self.gsd_ratios,
        )
        resized_crop = self.get_resized_crop(
            input_array=input_array,
            target_top_idx=target_top_idx,
            target_left_idx=target_left_idx,
            target_height=target_height,
            target_width=target_width,
            size=self.size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        if isinstance(data, dict):
            data[self.target_key] = resized_crop
            data[self.gsd_key] = output_gsd
            return data
        else:
            return resized_crop, output_gsd

    @staticmethod
    def get_params(
        input_array: torch.Tensor,
        gsd: float,
        size: typing.Tuple[int, int],
        gsd_ratios: typing.Tuple[float, float],
    ) -> typing.Tuple[int, int, int, int, float]:
        """Returns the parameters for a crop-and-resize operation on the input array.

        Args:
            input_array: the input array from which a crop should be generated.
            gsd: the ground sample distance associated with the above array.
            size: expected output size of the crop, for each edge, after resizing. This is expected
                to be height first, width second, and pixel values.
            gsd_ratios: expected min/max ratios to use when selecting a new ground sampling
                distance value. See constructor for more info.

        Returns:
            A tuple of parameters (top-idx, left-idx, height, width) to be passed to a crop
            function to return a crop, and the ground sampling distance (GSD) of the resulting
            crop.
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
        target_size = (int(round(size[0] * expected_scale)), int(round(size[1] * expected_scale)))
        assert target_size[0] <= input_height and target_size[1] <= input_width
        # now that we have the target crop size (pre-interpolation), we need to decide where to place it
        target_top_left_coords = (
            torch.randint(0, input_height - target_size[0] + 1, size=(1,)).item(),
            torch.randint(0, input_width - target_size[1] + 1, size=(1,)).item(),
        )
        # finally, for the given target size, compute the real output gsd
        final_scale = target_size[0] / size[0]
        output_gsd = final_scale * gsd
        return (target_top_left_coords[0], target_top_left_coords[1], target_size[0], target_size[1], output_gsd)

    @staticmethod
    def get_resized_crop(
        input_array: torch.Tensor,
        target_top_idx: int,
        target_left_idx: int,
        target_height: int,
        target_width: int,
        size: typing.Tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> torch.Tensor:
        """Crops the given input array with the specified parameters.

        This function simply calls torchvision's `resized_crop` function under the hood, but may
        be overridden in derived classes.

        Args:
            input_array: the input array from which a crop should be generated.
            target_top_idx: the index of the row that matches the crop's top left corner.
            target_left_idx: the index of the column that matches the crop's top left corner.
            target_height: the total height of the crop in the input array (in pixels).
            target_width: the total width of the crop in the input array (in pixels).
            size: expected output size of the crop, for each edge, after resizing. This is expected
                to be height first, width second, and pixel values.
            interpolation: Desired interpolation enum defined by
                `torchvision.transforms.InterpolationMode`. Default is bilinear.
            antialias: defines whether to apply antialiasing. It only affects tensors when the
                bilinear or bicubic interpolation modes are selected, and is ignored otherwise.

        Returns:
            The tensor that corresponds to the specified cropped and resized (with a predetermined
            interpolation approach) area of the input array.
        """
        resized_crop = torchvision.transforms.functional.resized_crop(
            img=input_array,
            top=target_top_idx,
            left=target_left_idx,
            height=target_height,
            width=target_width,
            size=size,  # noqa
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
