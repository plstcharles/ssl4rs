"""Implements a deeplake data repackager for the AID dataset.

See the following URL for more info on this dataset: https://captain-whu.github.io/AID/
"""

import pathlib
import typing

import deeplake
import numpy as np

import ssl4rs.data.metadata.aid
import ssl4rs.data.repackagers.utils
import ssl4rs.utils.imgproc


class DeepLakeRepackager(ssl4rs.data.repackagers.utils.DeepLakeRepackager):
    """Repackages the Aerial Image Dataset (AID) into a deeplake-compatible format.

    This dataset contains large-scale aerial images that can be used for classification. There are
    10,000 images (600x600, RGB) in this dataset, and these are given one of 30 class labels.

    See the following URL for more info on this dataset: https://captain-whu.github.io/AID/

    Note that this dataset does NOT have a fixed Ground Sampling Distance (GSD); images contained
    herein are mixed across different sources with GSDs between 0.5m and 8m.
    """

    metadata = ssl4rs.data.metadata.aid

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
        )

    @property  # we need to provide this for the base class!
    def dataset_name(self) -> str:
        """Returns the dataset name used to identify this particular dataset."""
        return "AID"

    def __len__(self) -> int:
        """Returns the total number of images defined in this dataset."""
        return self.metadata.image_count

    def __init__(
        self,
        dataset_root_path: typing.Union[typing.AnyStr, pathlib.Path],
    ):
        """Parses the dataset structure and makes sure all the data is present.

        Args:
            dataset_root_path: path to the directory containing all the AID data.
        """
        super().__init__()
        self.data_root_path = pathlib.Path(dataset_root_path)
        assert self.data_root_path.exists(), f"invalid dataset path: {self.data_root_path}"
        assert sum(self.metadata.class_distrib.values()) == len(self)
        for class_idx, class_name in enumerate(self.metadata.class_names):
            class_dir_path = self.data_root_path / class_name
            assert class_dir_path.is_dir(), f"invalid class directory path: {class_dir_path}"
            img_paths = sorted(class_dir_path.glob(f"{class_name.lower()}_*.jpg"))
            assert len(img_paths) != 0, f"could not find any images in class dir: {class_dir_path}"
            img_count, exp_img_count = len(img_paths), self.metadata.class_distrib[class_name]
            assert (
                img_count == exp_img_count
            ), f"bad image count for {class_name} (found {img_count} instead of {exp_img_count})"
            img_idxs = [int(str(p.parts[-1]).rsplit("_")[-1].split(".jpg")[0]) for p in img_paths]
            assert np.array_equal(
                np.unique(img_idxs), np.unique(range(1, img_count + 1))
            ), f"unexpected duplicate image names / split results for {class_name}"
            # we'll open a single image per class here to make sure the resolution is as expected...
            picked_img_path = np.random.choice(img_paths)
            height, width = ssl4rs.utils.imgproc.get_image_shape_from_file(picked_img_path)
            assert (
                width == self.metadata.image_shape[1] and height == self.metadata.image_shape[0]
            ), f"unexpected image shape (got {width}x{height}, expected 600x600)"
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
        class_name = self.metadata.class_names[class_idx]
        image_name = f"{class_name.lower()}_{sample_idx + 1}.jpg"
        image_path = self.data_root_path / class_name / image_name
        assert image_path.exists(), f"unexpected invalid image path in getitem: {image_path}"
        return dict(  # note: the tensor names here must match with the ones in `tensor_info`!
            image=deeplake.read(str(image_path)),  # this will defer loading the full image data if needed
            label=class_idx,
        )


def _repackage_aid(dataset_root_path: pathlib.Path):
    assert dataset_root_path.is_dir(), f"missing dataset root directory: {dataset_root_path}"
    repackager = DeepLakeRepackager(dataset_root_path)
    output_path = dataset_root_path / ".deeplake"
    repackager.export(output_path)
    assert pathlib.Path(output_path).exists()


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config()
    _repackage_aid(pathlib.Path(config_.utils.data_root_dir) / "aid")
