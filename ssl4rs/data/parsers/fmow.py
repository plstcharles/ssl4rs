"""Implements a data parser for the Functional Map of the World (fMoW) dataset.

See the following URLs for more info on this dataset:
https://arxiv.org/abs/1711.07846
https://github.com/fMoW/dataset
https://spacenet.ai/iarpa-functional-map-of-the-world-fmow/
"""

import pathlib
import typing

import cv2 as cv
import deeplake
import numpy as np

import ssl4rs.data.metadata.fmow
import ssl4rs.data.parsers.utils
import ssl4rs.utils.imgproc
import ssl4rs.utils.logging

logger = ssl4rs.utils.logging.get_logger(__name__)


class DeepLakeParser(ssl4rs.data.parsers.utils.DeepLakeParser):
    """Provides data parsing functions for the ENTIRE fMoW dataset (i.e. the 'all' subset).

    Note that fMoW requires a bit of special handling on top of the base deeplake parser. In short,
    we need to handle different datasets (for instance data and image data) that are different
    lengths, and we need to do this manually for each subset.

    This class implements getter functions for each of the expected subsets of the fMoW dataset,
    specifically: `get_train_subset`, `get_val_subset`, `get_test_subset`, and `get_seq_subset`.
    Those subsets will be provided in a DeepLakeParser-compatible interface to simplify and
    optimize data loading.
    """

    metadata = ssl4rs.data.metadata.fmow

    supported_decompression_strategies = [
        "defer",  # defer image decompression to later (will have to be done outside the class)
        "deeplake",  # use the deeplake (built-in) decompression implementation (for any format)
        "opencv",  # use the opencv (imdecode) decompression implementation (for jpegs/pngs, to BGR)
        "libjpeg-turbo",  # use the libjpeg-turbo decompression implementation (for jpegs only!)
    ]

    def __init__(
        self,
        dataset_path_or_object: typing.Union[typing.AnyStr, pathlib.Path, deeplake.Dataset],
        decompression_strategy: str = "deeplake",
        batch_transforms: "ssl4rs.data.BatchTransformType" = None,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
        **extra_deeplake_kwargs,
    ):
        """Parses a fMoW deeplake archive or wraps an already-opened object.

        Note that due to the design of this class (and in contrast to the exporter class), all
        datasets should only ever be opened in read-only mode here.
        """
        self.save_hyperparameters(
            ignore=["dataset_path_or_object", "extra_deeplake_kwargs"],
            logger=False,
        )
        super().__init__(
            dataset_path_or_object=dataset_path_or_object,
            batch_transforms=batch_transforms,
            batch_id_prefix=batch_id_prefix,
            save_hyperparams=False,
            **extra_deeplake_kwargs,
        )
        assert (
            decompression_strategy in self.supported_decompression_strategies
        ), f"invalid decompression strategy: {decompression_strategy}"
        self.decompression_strategy = decompression_strategy

    def __len__(self) -> int:
        """Returns the total size (in terms of instance count) of the dataset."""
        return len(self.dataset.instances)

    def _get_raw_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch loaded from the deeplake dataset at a specified index.

        In contrast with the `__getitem__` function, this internal call will not apply transforms.

        This is a custom reimplementation of the base class version that processes sequences of data
        so that they can be batched properly.
        """
        instance_data = self.dataset.instances[index]
        image_idxs = instance_data.image_idxs.list()
        image_data = [self.dataset.images[img_idx] for img_idx in image_idxs]
        batch = _get_batch_from_sample_data(instance_data, image_data, self.decompression_strategy)
        return batch

    def _get_subset(
        self,
        subset: str,
    ) -> "_DeepLakeSubsetParser":
        """Returns a subset data parser for a specific intersection of dataset instances."""
        assert subset in self.metadata.subset_types, f"unsupported subset: {subset}"
        logger.info(f"Preparing parser for {self.dataset_name}-{subset}")
        target_subset_idx = self.metadata.subset_types.index(subset)
        instances_parser = self.dataset.instances.filter(
            lambda batch: batch["subset"].numpy() == target_subset_idx,
        )
        num_instances = len(instances_parser)
        assert num_instances > 0
        target_image_idxs = instances_parser.image_idxs.list()
        assert len(target_image_idxs) == num_instances
        target_image_idxs = np.concatenate(target_image_idxs)
        assert len(target_image_idxs) == len(np.unique(target_image_idxs))
        image_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(target_image_idxs)}
        images_parser = self.dataset.images[target_image_idxs.tolist()]
        instance_tensor_keys = set(instances_parser.tensors.keys())
        image_tensor_keys = set(images_parser.tensors.keys())
        assert not instance_tensor_keys.intersection(image_tensor_keys)
        return _DeepLakeSubsetParser(
            parent_dataset=self.dataset,
            instance_subset=instances_parser,
            image_subset=images_parser,
            image_idx_map=image_idx_map,
            subset=subset,
            **self.hparams,
        )

    def get_train_subset(self):
        """Returns a `DeepLakeParser`-compatible parser for the fMoW training set."""
        return self._get_subset("train")

    def get_val_subset(self):
        """Returns a `DeepLakeParser`-compatible parser for the fMoW validation set."""
        return self._get_subset("val")

    def get_test_subset(self):
        """Returns a `DeepLakeParser`-compatible parser for the fMoW testing set."""
        return self._get_subset("test")

    def get_seq_subset(self):
        """Returns a `DeepLakeParser`-compatible parser for the fMoW seq set."""
        return self._get_subset("seq")


class _DeepLakeSubsetParser(ssl4rs.data.parsers.utils.DeepLakeParser):
    """Provides data parsing functions for a fMoW data subset.

    NOTE: AN INSTANCE OF THIS CLASS SHOULD NEVER BE CREATED DIRECTLY BY A USER. THE INTERFACE
    SHOULD BE THE SAME AS THE `DeepLakeParser` CLASS ABOVE.
    """

    metadata = ssl4rs.data.metadata.fmow
    supported_decompression_strategies = DeepLakeParser.supported_decompression_strategies

    def __init__(
        self,
        parent_dataset: deeplake.Dataset,
        instance_subset: deeplake.Dataset,
        image_subset: deeplake.Dataset,
        image_idx_map: typing.Dict[int, int],
        decompression_strategy: str,
        subset: str,
        batch_transforms: "ssl4rs.data.BatchTransformType" = None,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
    ):
        """Parses a fMoW deeplake subset."""
        self.save_hyperparameters(
            ignore=["parent_dataset", "instance_subset", "image_subset", "image_idx_map"],
            logger=False,
        )
        super().__init__(
            dataset_path_or_object=parent_dataset,
            batch_transforms=batch_transforms,
            batch_id_prefix=batch_id_prefix,
            save_hyperparams=False,
        )
        self.instance_subset = instance_subset
        self.image_subset = image_subset
        self.image_idx_map = image_idx_map
        self.subset = subset
        self.decompression_strategy = decompression_strategy

    def __len__(self) -> int:
        """Returns the total size (in terms of instance count) of the data subset."""
        return len(self.instance_subset)

    def _get_raw_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch loaded from the deeplake dataset at a specified index.

        In contrast with the `__getitem__` function, this internal call will not apply transforms.

        This is a custom reimplementation of the base class version that processes sequences of data
        so that they can be batched properly.
        """
        instance_data = self.instance_subset[index]  # noqa
        image_idxs = [self.image_idx_map[idx] for idx in instance_data.image_idxs.list()]
        image_data = self.image_subset[image_idxs]
        batch = _get_batch_from_sample_data(instance_data, image_data, self.decompression_strategy)
        return batch

    @property
    def dataset_name(self) -> typing.AnyStr:
        """Returns the dataset name used to identify this particular data subset."""
        return f"{super().dataset_name}-{self.subset}"


def _get_batch_from_sample_data(
    instance_data,
    image_data,
    decompression_strategy: str,
):
    """Converts a pair of instance + image data samples into a batch dictionary."""
    batch = {
        "instance/label": instance_data.label.numpy(),
        "instance/subset": instance_data.subset.numpy(),
        "instance/id": instance_data.id.text(),
    }
    dataset_info = instance_data.parent.info
    image_type = dataset_info["image_type"]
    image_compression = dataset_info["image_compression"]
    assert len(image_data) > 0, "all instances should have at least one image?"
    if image_type == "rgb":
        batch["images/rgb/bbox"] = [img_data.rgb.bbox.numpy() for img_data in image_data]
        batch["images/rgb/metadata"] = [img_data.rgb.metadata.dict() for img_data in image_data]
        if image_compression == "jpg":
            if decompression_strategy == "defer":
                # assume the user will decompress images later (provide the bytes array directly)
                batch["images/rgb/jpg"] = [img_data.rgb.jpg.tobytes() for img_data in image_data]
            elif decompression_strategy == "deeplake":
                # use deeplake to auto-decompress its internal jpg compression by asking for a numpy array
                batch["images/rgb/jpg"] = [img_data.rgb.jpg.numpy() for img_data in image_data]
            elif decompression_strategy == "opencv":
                batch["images/rgb/jpg"] = [  # remember: with OpenCV, the images will be in BGR order!
                    cv.imdecode(np.fromstring(img_data.rgb.jpg.tobytes(), np.uint8), cv.IMREAD_COLOR)
                    for img_data in image_data
                ]
            elif decompression_strategy == "libjpeg-turbo":
                # with turbo-jpeg, by default, we'll activate all the extra-speed stuff here
                batch["images/rgb/jpg"] = [
                    ssl4rs.utils.imgproc.decode_jpg(
                        image=img_data.rgb.jpg.tobytes(),
                        to_bgr_format=False,
                        use_fast_upsample=True,
                        use_fast_dct=True,
                    )
                    for img_data in image_data
                ]
            else:
                raise ValueError(f"bad decompression strategy: {decompression_strategy}")
        else:
            raise NotImplementedError  # todo: implement me for png/lz4 data
    else:
        raise NotImplementedError  # todo: implement me for multispectral data
    return batch
