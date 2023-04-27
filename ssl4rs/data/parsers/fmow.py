"""Implements a data parser for the Functional Map of the World (fMoW) dataset.

See the following URLs for more info on this dataset:
https://arxiv.org/abs/1711.07846
https://github.com/fMoW/dataset
https://spacenet.ai/iarpa-functional-map-of-the-world-fmow/
"""

import typing

import deeplake

import ssl4rs.data.parsers.utils
import ssl4rs.data.repackagers.fmow


class DeepLakeParser(ssl4rs.data.parsers.utils.DeepLakeParser):
    """fMoW requires a bit of special handling on top of the base deeplake parser."""

    metadata = ssl4rs.data.metadata.fmow

    def __init__(
        self,
        dataset_path_or_object: typing.Union[typing.AnyStr, deeplake.Dataset],
        batch_transforms: "ssl4rs.data.BatchTransformType" = None,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
        **extra_deeplake_kwargs,
    ):
        """Parses a fMoW deeplake archive or wraps an already-opened object.

        Note that due to the design of this class (and in contrast to the exporter class), all
        datasets should only ever be opened in read-only mode here.
        """
        super().__init__(
            dataset_path_or_object=dataset_path_or_object,
            batch_transforms=batch_transforms,
            batch_id_prefix=batch_id_prefix,
            **extra_deeplake_kwargs,
        )
        # TODO @@@@@: add ignore metadata, ignore imgidxs, ... (to avoid batching those?)
        # TODO 2 @@@@@@@: add flags to change jpeg loading strategy (deeplake vs manual decoding)

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
        batch = {
            "images/rgb/jpg": [img_data.rgb.jpg for img_data in image_data],
            "images/rgb/bbox": [img_data.rgb.bbox.numpy() for img_data in image_data],
            "images/rgb/metadata": [img_data.rgb.metadata.dict() for img_data in image_data],
            "instance/image_idxs": image_idxs,
            "instance/label": instance_data.label.numpy(),
            "instance/subset": instance_data.subset.numpy(),
            "instance/id": instance_data.id.text(),
        }
        return batch
        # @@@@@@@@@@ get images with .tobytes() and decompress manually w/ libjpeg-turbo? (maybe in wrapper?)
