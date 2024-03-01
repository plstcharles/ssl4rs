"""Implements a deeplake data repackager for the UC Merced Land Use dataset.

See the following URL(s) for more info on this dataset:
http://weegee.vision.ucmerced.edu/datasets/landuse.html
http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
"""

import pathlib
import typing

import cv2 as cv
import deeplake
import numpy as np

import ssl4rs.data.metadata.ucmerced
import ssl4rs.data.repackagers.utils
import ssl4rs.utils.config
import ssl4rs.utils.filesystem
import ssl4rs.utils.imgproc
import ssl4rs.utils.logging


class DeepLakeRepackager(ssl4rs.data.repackagers.utils.DeepLakeRepackager):
    """Repackages the UC Merced Land Use dataset into a deeplake-compatible format.

    This dataset contains 256x256 images with a GSD of ~0.3m that can be used for classification.
    There are 21 classes and 21x100=2100 images in this dataset.

    For more info, see the dataset pages here:
    http://weegee.vision.ucmerced.edu/datasets/landuse.html
    https://www.tensorflow.org/datasets/catalog/uc_merced
    """

    metadata = ssl4rs.data.metadata.ucmerced

    @property  # we need to provide this for the base class!
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor declarations used during repackaging."""
        return dict(
            image=dict(htype="image", dtype=np.uint8, sample_compression="jpg"),
            label=dict(htype="class_label", dtype=np.int16, class_names=self.metadata.class_names),
        )

    @property  # we need to provide this for the base class!
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information that will be exported in the deeplake object."""
        return dict(
            name=self.dataset_name,
            class_names=self.metadata.class_names,
            class_distrib=self.metadata.class_distrib,
            image_shape=list(self.metadata.image_shape),  # tuples will be changed to lists by deeplake...
            ground_sampling_distance=self.metadata.ground_sampling_distance,
        )

    @property  # we need to provide this for the base class!
    def dataset_name(self) -> str:
        """Returns the dataset name used to identify this particular dataset."""
        return "UCMerced"

    def __len__(self) -> int:
        """Returns the total number of images defined in this dataset."""
        return self.metadata.image_count

    def __init__(
        self,
        dataset_root_path: typing.Union[typing.AnyStr, pathlib.Path],
    ):
        """Parses the dataset structure and makes sure all the data is present.

        Args:
            dataset_root_path: path to the directory containing all the UCMerced data.
        """
        self.data_root_path = pathlib.Path(dataset_root_path)
        assert self.data_root_path.exists(), f"invalid dataset path: {self.data_root_path}"
        for class_idx, class_name in enumerate(self.metadata.class_names):
            class_name_slug = ssl4rs.utils.filesystem.slugify(class_name)
            class_dir_path = self.data_root_path / class_name_slug
            assert class_dir_path.is_dir(), f"invalid class directory path: {class_dir_path}"
            img_paths = list(sorted(class_dir_path.glob(f"{class_name_slug}*.tif")))
            assert len(img_paths) != 0, f"could not find any images in class dir: {class_dir_path}"
            img_count, exp_img_count = len(img_paths), self.metadata.class_distrib[class_name]
            assert (
                img_count == exp_img_count
            ), f"bad image count for {class_name} (found {img_count} instead of {exp_img_count})"
            # we'll open a single image per class here to make sure the resolution is as expected...
            height, width = ssl4rs.utils.imgproc.get_image_shape_from_file(img_paths[0])
            assert (
                height == self.metadata.image_shape[0] and width == self.metadata.image_shape[1]
            ), f"unexpected image shape (got {height}x{width}, expected 256x256)"
        # finally, prepare the global-to-classwise index range mapper for the getitem function
        self.image_idxs_ranges = [
            range(
                sum(self.metadata.class_distrib[self.metadata.class_names[cidx]] for cidx in range(0, class_idx)),
                sum(self.metadata.class_distrib[self.metadata.class_names[cidx]] for cidx in range(0, class_idx + 1)),
            )
            for class_idx, class_name in enumerate(self.metadata.class_names)
        ]
        # once we get here, we're ready to repackage the dataset!

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Fetches and returns a data sample for this dataset.

        In this case, we return an image and its associated class label into a dictionary. Note
        that this code will likely be slower than the deeplake fetching implementation, thus why
        this is a "repackager" object, and not a dataset parser (although it could be used as
        one...).
        """
        assert 0 <= item < len(self), f"invalid data sample index being queried: {item}"
        class_idx, sample_idx = next(
            (class_idx, item - class_range.start)
            for class_idx, class_range in enumerate(self.image_idxs_ranges)
            if item in class_range
        )
        class_name_slug = ssl4rs.utils.filesystem.slugify(self.metadata.class_names[class_idx])
        image_name = f"{class_name_slug}{sample_idx:02d}.tif"
        image_path = self.data_root_path / class_name_slug / image_name
        assert image_path.exists(), f"unexpected invalid image path in getitem: {image_path}"
        image = deeplake.read(str(image_path)).array
        if image.shape != self.metadata.image_shape:
            image = cv.resize(
                image,
                dsize=(self.metadata.image_shape[1], self.metadata.image_shape[0]),
                interpolation=cv.INTER_CUBIC,
            )
        return dict(  # note: the tensor names here must match with the ones in `tensor_info`!
            image=image,
            label=class_idx,
        )


def _repackage_ucmerced(dataset_root_path: pathlib.Path):
    assert dataset_root_path.is_dir(), f"missing dataset root directory: {dataset_root_path}"
    repackager = DeepLakeRepackager(dataset_root_path)
    output_path = dataset_root_path / ".deeplake"
    repackager.export(output_path)
    assert pathlib.Path(output_path).exists()


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config()
    _repackage_ucmerced(pathlib.Path(config_.utils.data_root_dir) / "UCMerced_LandUse")
