"""Implements a data parser for the Functional Map of the World (FMoW) dataset.

See the following URLs for more info on this dataset:
https://arxiv.org/abs/1711.07846
https://github.com/fMoW/dataset
https://spacenet.ai/iarpa-functional-map-of-the-world-fmow/
"""

import typing

import ssl4rs.data.parsers.utils
import ssl4rs.data.repackagers.fmow


class DeepLakeParser(ssl4rs.data.parsers.utils.DeepLakeParser):
    """FMoW requires a bit of special handling on top of the base deeplake parser."""

    metadata = ssl4rs.data.metadata.fmow

    def _get_raw_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch loaded from the deeplake dataset at a specified index.

        In contrast with the `__getitem__` function, this internal call will not apply transforms.

        This is a custom reimplementation of the base class version that processes sequences of data
        so that they can be batched properly.
        """
        data = self.dataset[index]  # noqa
        assert all([name in data for name in self.tensor_names])
        batch = dict(
            bboxes=data["bboxes"].numpy(),  # shape = (N, 4) where N is the view count
            instance=data["instance"].text(),  # str with instance name
            label_idx=int(data["label"].numpy()),  # 0-based class index
            metadata=data["metadata"].dict(),  # list of N (= view count) metadata dicts
            subset_idx=int(data["subset"].numpy()),  # 0-based subset index
            views=data["views"],  # return the deeplake dataset/tensor view as-is for more preproc
        )
        assert len(batch["bboxes"]) == len(batch["metadata"])
        assert len(batch["bboxes"]) == batch["views"].shape[0]
        batch["view_count"] = batch["views"].shape[0]
        batch["label"] = self.metadata.class_names[batch["label_idx"]]
        batch["subset"] = self.metadata.subset_types[batch["subset_idx"]]
        return batch
