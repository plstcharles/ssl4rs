"""Implements a deeplake data repackager for the Functional Map of the World (FMoW) dataset.

See the following URLs for more info on this dataset:
https://arxiv.org/abs/1711.07846
https://github.com/fMoW/dataset
https://spacenet.ai/iarpa-functional-map-of-the-world-fmow/
"""
import bz2
import io
import json
import pathlib
import pickle
import re
import tarfile
import typing

import deeplake
import numpy as np
import PIL.Image
import tqdm

import ssl4rs.data.metadata.fmow
import ssl4rs.data.repackagers.utils
import ssl4rs.utils.imgproc

expected_max_fmow_pixels = np.prod(ssl4rs.data.metadata.fmow.max_image_shape[0:2])
PIL.Image.MAX_IMAGE_PIXELS = max(expected_max_fmow_pixels, PIL.Image.MAX_IMAGE_PIXELS)
logger = ssl4rs.utils.logging.get_logger(__name__)


class DeepLakeRepackager(ssl4rs.data.repackagers.utils.DeepLakeRepackager):
    """Repackages the Functional Map of the World (FMoW) dataset into a deeplake format.

    Each class instance will be stored as a group with all its metadata in one place, and it will
    point (using indices) to the images that belong to it in a separate dataset. Each image will be
    stored in its original format so that they can be manually decoded if needed.

    Note: running this repackager requires 16GB+ of RAM, as many JSON configs will be preloaded to
    memory to simplify/speed up mappings. When processing the RGB dataset with all subsets and JPEG
    compression, the resulting deeplake archive will be roughly 180GB (as of FMoW v1.2.1).
    """

    metadata = ssl4rs.data.metadata.fmow

    @property  # we need to provide this for the base class!
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor info (declaration) arguments used during creation."""
        assert self.image_type == "rgb", "other dataset types not supported yet! @@@@"
        images_prefix = f"images/{self.image_type}"
        images_dataset_dict = dict(
            htype="image.rgb" if self.image_type == "rgb" else "generic",
            dtype=np.uint8 if self.image_type == "rgb" else np.int16,  # TODO: make sure int16 is OK?
            sample_compression=self.image_compression_type,
        )
        return {
            # note: the 'images' tensor datasets share the same length among themselves;
            # they contain the concatenated image data of ALL instances, and a handful of these
            # should be returned for each instance, based on the `image_idxs` values provided below
            f"{images_prefix}/{self.image_compression_type}": images_dataset_dict,
            f"{images_prefix}/bbox": dict(htype="bbox", dtype=np.int32, coords=dict(type="pixel", mode="LTWH")),
            f"{images_prefix}/metadata": dict(htype="json", sample_compression=None),
            # the 'instances' tensor datasets share the same length, and each sample should
            # correspond to a single instance in the original FMoW dataset (i.e. one real "object")
            "instances/image_idxs": dict(htype="list", sample_compression=None),  # list of image indices
            "instances/label": dict(htype="class_label", dtype=np.int16, class_names=self.metadata.class_names),
            "instances/subset": dict(htype="class_label", dtype=np.int8, class_names=self.metadata.subset_types),
            "instances/id": dict(htype="text", dtype=str, sample_compression=None),
        }

    @property  # we need to provide this for the base class!
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information that will be exported in the deeplake object."""
        return dict(
            name=self.dataset_name,
            class_names=self.metadata.class_names,
            image_type=self.image_type,
            image_compression=self.image_compression_type,
            subset_types=self.subset_types,
            version=self.version,
        )

    @property  # we need to provide this for the base class!
    def dataset_name(self) -> str:
        """Returns the dataset name used to identify this particular dataset."""
        return f"FMoW-{self.image_type}"

    @staticmethod
    def _insert_with_recursive_key(
        out_dict: typing.Dict[typing.AnyStr, typing.Any],
        key: typing.AnyStr,
        data: typing.Any,
        key_delimiter: typing.AnyStr = "/",
        overwrite_if_needed: bool = True,
    ) -> None:
        """Inserts a value inside a dictionary with a recursive key, with subdicts if needed."""
        assert isinstance(key_delimiter, str) and isinstance(key, str)
        sub_keys = key.split(key_delimiter)
        for group_key in sub_keys[:-1]:
            if group_key not in out_dict:
                out_dict[group_key] = dict()
            out_dict = out_dict[group_key]
        final_key = sub_keys[-1]
        if not overwrite_if_needed:
            assert final_key not in out_dict, f"dictionary already contains a value at {key}"
        out_dict[final_key] = data

    def _load_version(self) -> typing.AnyStr:
        """Loads the content of FMoW's CHANGELOG.md and reads the latest version from it."""
        changelog_path = self.data_root_path / "CHANGELOG.md"
        assert changelog_path.is_file(), f"invalid changelog file: {changelog_path}"
        latest_version_number = None
        with open(changelog_path) as fd:
            for line in fd:
                match = re.search(r"^## \[(\d+\.\d+\.\d+)]", line)
                if match:
                    latest_version_number = match.group(1)
                    break
        assert latest_version_number is not None, "could not find a version number from changelog"
        version = latest_version_number
        logger.info(f"will load FMoW v{version}")
        return version

    def _load_manifest(self) -> typing.List:
        """Loads the `manifest.json.bz2` file contents to memory."""
        manifest_path = self.data_root_path / "manifest.json.bz2"
        assert manifest_path.is_file(), f"invalid manifest file: {manifest_path}"
        logger.info(f"parsing FMoW file: {manifest_path}")
        with bz2.open(manifest_path, "rt") as fd:
            manifest_data = json.load(fd)
        assert isinstance(manifest_data, list), "invalid manifest"
        return manifest_data

    def _load_groundtruth(self) -> typing.Tuple[typing.Dict, typing.Dict]:
        """Loads the `groundtruth.tar.bz2` file contents to memory."""
        groundtruth_path = self.data_root_path / "groundtruth.tar.bz2"
        assert groundtruth_path.is_file(), f"invalid gt file: {groundtruth_path}"
        groundtruth_cache_path = self.data_root_path / "groundtruth.cache.pkl"
        if groundtruth_cache_path.is_file() and self._use_cache:
            with open(groundtruth_cache_path, "rb") as fd:
                gt_mappings, gt_data = pickle.load(fd)
            return gt_mappings, gt_data
        logger.info(f"parsing FMoW file: {groundtruth_path}")
        with open(groundtruth_path, mode="rb") as fd:
            tar_data = io.BytesIO(fd.read())
        gt_mappings, gt_data = dict(), dict()
        with tarfile.open(fileobj=tar_data, mode="r:bz2") as tar_fd:
            expected_total_members = 1047693  # seems OK as of FMoW v1.2.1 (tested on 2023-03-26)
            with tqdm.tqdm(total=expected_total_members, desc="loading groundtruth archive") as pbar:
                while True:
                    curr_member = tar_fd.next()
                    if curr_member is None:
                        break  # we've seen the entire archive, return
                    if not curr_member.isfile():
                        continue  # skip over directories
                    assert curr_member.isfile()  # should not contain symlinks and other stuff...
                    assert curr_member.name.endswith(".json")  # should only contain json files...
                    curr_data = json.load(tar_fd.extractfile(curr_member))
                    if curr_member.name in ["seq_gt_mapping.json", "test_gt_mapping.json"]:
                        assert isinstance(curr_data, list) and isinstance(curr_data[0], dict)
                        # we'll remap the mappings to a dictionary with the secret ids as keys
                        curr_data = {mapping["output"].split("/")[1]: mapping for mapping in curr_data}
                        subset_id = curr_member.name.split("_")[0]
                        assert subset_id not in gt_mappings
                        gt_mappings[subset_id] = curr_data
                    else:
                        self._insert_with_recursive_key(
                            out_dict=gt_data,
                            key=curr_member.name,
                            data=curr_data,
                            overwrite_if_needed=False,
                        )
                    pbar.update(1)
        if self._use_cache:
            with open(groundtruth_cache_path, "wb") as fd:
                pickle.dump((gt_mappings, gt_data), fd)
        return gt_mappings, gt_data

    def _parse_metadata(self) -> typing.Dict:
        """Parses the metadata of all instances using the groundtruth json data."""
        metadata_cache_path = self.data_root_path / "metadata.cache.pkl"
        if metadata_cache_path.is_file() and self._use_cache:
            with open(metadata_cache_path, "rb") as fd:
                metadata = pickle.load(fd)
            return metadata
        metadata = dict()
        assert self._gt_mappings and self._gt_data, "gt data must be loaded first"
        wanted_prefixes = [subset_id + "/" for subset_id in self.subset_types]
        wanted_suffixes = [f"_{self.image_type}.jpg"]
        assert len(self._manifest_data) > 0
        for file_path in tqdm.tqdm(self._manifest_data, desc="building metadata tables"):
            assert isinstance(file_path, str)
            is_file_in_expected_subset = [file_path.startswith(prefix) for prefix in wanted_prefixes]
            is_file_in_expected_format = [file_path.endswith(suffix) for suffix in wanted_suffixes]
            if not (any(is_file_in_expected_subset) and any(is_file_in_expected_format)):
                continue
            is_file_in_secret_subset = any([file_path.startswith(s) for s in ["test/", "seq/"]])
            is_file_in_public_subset = any([file_path.startswith(s) for s in ["train/", "val/"]])
            assert is_file_in_secret_subset or is_file_in_public_subset
            file_name_tokens = file_path.split("/")
            if is_file_in_secret_subset:
                # we need to remap the file name tokens to the real class/instance ids
                assert len(file_name_tokens) == 3, "unexpected file name tokens for test/seq set"
                subset_id, secret_id, file_id = file_name_tokens
                mapping = self._gt_mappings[subset_id][secret_id]
                remapped_tokens = mapping["input"].split("/")
                assert len(remapped_tokens) == 3
                remapped_subset_id, class_id, instance_id = remapped_tokens
                assert remapped_subset_id == subset_id + "_gt"
                assert class_id in self.metadata.class_names
                instance_metadata = self._gt_data[remapped_subset_id][class_id][instance_id]
            else:
                # we can extract the class/instance ids as-is
                assert len(file_name_tokens) == 4, "unexpected file name tokens for train/val set"
                subset_id, class_id, instance_id, file_id = file_name_tokens
                assert subset_id in self.subset_types
                assert class_id in self.metadata.class_names
                instance_metadata = self._gt_data[subset_id][class_id][instance_id]
            assert isinstance(instance_metadata, dict)
            for curr_img_meta in instance_metadata.values():
                assert "bounding_boxes" in curr_img_meta
                assert isinstance(curr_img_meta["bounding_boxes"], list)
                # there should only be one groundtruth bbox per entire image
                assert sum(bbox["ID"] == -1 for bbox in curr_img_meta["bounding_boxes"]) == 1
                # ... and it should be the first in the list
                assert curr_img_meta["bounding_boxes"][0]["ID"] == -1
            assert instance_id.startswith(class_id)
            assert file_id.count("_") >= 2
            img_id = file_id.rsplit("_", maxsplit=2)[-2]
            assert int(img_id) >= 0
            img_metadata_key = f"{instance_id}_{img_id}_{self.image_type}.json"
            img_metadata = instance_metadata[img_metadata_key]
            assert isinstance(instance_metadata, dict)
            assert not file_path.endswith(".json")
            file_abs_path = self.data_root_path / file_path
            assert file_abs_path.is_file(), f"missing file: {file_path}"
            self._insert_with_recursive_key(
                out_dict=metadata,
                key=subset_id + "/" + instance_id + f"/img_{img_id}",
                data=dict(
                    file_abs_path=file_abs_path,
                    metadata=img_metadata,
                    subset_id=subset_id,
                    class_id=class_id,
                    instance_id=instance_id,
                    file_id=file_id,
                    img_id=img_id,
                ),
                overwrite_if_needed=False,
            )
        if self._use_cache:
            with open(metadata_cache_path, "wb") as fd:
                pickle.dump(metadata, fd)
        return metadata

    def _prepare_maps(self) -> None:
        """Prepares the sample index and instance ID maps used to fetch specific data."""
        assert self._total_instances == 0, "should only call once!"
        for subset_id, instance_ids in self._metadata.items():
            for instance_id, img_ids in self._metadata[subset_id].items():
                assert len(list(img_ids)) > 0
                imgs = [self._metadata[subset_id][instance_id][img_id] for img_id in img_ids]
                assert len({v["class_id"] for v in imgs}) == 1
                class_id = imgs[0]["class_id"]
                ids_tuple = (subset_id, class_id, instance_id, tuple(img_ids))
                assert ids_tuple not in self._ids_to_idx_map
                self._ids_to_idx_map[ids_tuple] = self._total_instances
                self._idx_to_ids_map[self._total_instances] = ids_tuple
                self._total_instances += 1
                self._total_images += len(img_ids)
        assert self._total_instances > 0, "something went wrong..."

    def __init__(
        self,
        dataset_root_path: typing.Union[typing.AnyStr, pathlib.Path],
        image_type: typing.AnyStr = "rgb",
        image_compression_type: typing.AnyStr = "jpg",
        subset_type: typing.Union[typing.AnyStr, typing.List] = "all",
        use_cache: bool = False,
    ):
        """Parses the dataset structure and makes sure all the data is present.

        Args:
            dataset_root_path: path to the directory containing all the FMoW data.
            image_type: image type to extract; should be either 'rgb' for the RGB-only images,
                or 'full' for the multispectral images.
            image_compression_type: image compression type to use in repackaged dataset.
            subset_type: subset type to extract; should be train/val/test/seq, or 'all' for their
                combination.
            use_cache: toggles whether to cache the groundtruth and metadata attributes.
        """
        assert image_type in self.metadata.image_types
        self.image_type = image_type
        assert image_compression_type in ["jpg"], f"unsupported compression type: {image_compression_type}"
        self.image_compression_type = image_compression_type
        if not isinstance(subset_type, list):
            assert subset_type in self.metadata.subset_types, f"unsupported FMoW subset type: {subset_type}"
            if subset_type == "all":
                self.subset_types = [s for s in self.metadata.subset_types if s != "all"]
            else:
                self.subset_types = [subset_type]
        else:
            assert all(
                [s in self.metadata.subset_types for s in subset_type]
            ), f"bad FMoW subset type(s): {subset_type}"
            self.subset_types = list(set(subset_type))
        self._use_cache = use_cache
        dataset_root_path = pathlib.Path(dataset_root_path)
        self.data_root_path = dataset_root_path / f"fmow-{self.image_type}"
        assert self.data_root_path.exists(), f"invalid dataset path: {self.data_root_path}"
        expected_data = ["train", "val", "test", "seq", "groundtruth.tar.bz2", "manifest.json.bz2"]
        assert all([(self.data_root_path / c).exists() for c in expected_data])
        self.version = self._load_version()  # loaded from changelog, string in a.b.c format
        self._manifest_data = self._load_manifest()  # contains all files provided in this dataset
        self._gt_mappings, self._gt_data = self._load_groundtruth()  # used internally to assign metadata
        self._metadata = self._parse_metadata()  # contains paths + metadata for the targeted images
        # free up some memory right away
        del self._gt_mappings
        del self._gt_data
        self._idx_to_ids_map = dict()  # sample idx (int) -> subset, class, instance, images
        self._ids_to_idx_map = dict()  # subset, class, instance, images -> sample idx (int)
        self._total_instances = 0
        self._total_images = 0
        self._prepare_maps()
        # once we get here, we're ready to repackage the dataset!

    def __len__(self) -> int:
        """Returns the total number of instances defined in this dataset.

        Note: each instance may have more than one image!
        """
        return self._total_instances

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Fetches and returns a data sample for this dataset.

        In this implementation, a 'data sample' is actually an instance of a class inside the
        dataset. It may therefore have more than one image (each image is captured at a different
        time). Each image is linked with its own metadata (which includes groundtruth info).

        Note that this code will likely be slower than the deeplake fetching implementation, thus
        why this is a "repackager" object, and not a dataset parser (although it could be used as
        one...).
        """
        assert 0 <= item < len(self), f"invalid data sample index being queried: {item}"
        (subset_id, class_id, instance_id, img_ids) = self._idx_to_ids_map[item]
        assert subset_id in self.subset_types
        assert class_id in self.metadata.class_names
        img_dicts = list(self._metadata[subset_id][instance_id].values())
        # using deeplake.read will defer loading the full image data if possible/needed
        imgs = [deeplake.read(str(img["file_abs_path"])) for img in img_dicts]
        metadata = [img["metadata"] for img in img_dicts]
        gt_bboxes = [m["bounding_boxes"][0] for m in metadata]
        assert all([bbox["ID"] == -1 for bbox in gt_bboxes])  # meaning the 'ground truth' bbox
        bboxes = [np.asarray(bbox["box"], dtype=np.int32) for bbox in gt_bboxes]
        images_prefix = f"images/{self.image_type}"
        images_dataset_name = f"{images_prefix}/{self.image_compression_type}"
        return {  # note: the tensor names here must match with the ones in `tensor_info`!
            # the first three entries should be lists with the same length (i.e. the image count)
            images_dataset_name: imgs,
            f"{images_prefix}/bbox": bboxes,
            f"{images_prefix}/metadata": metadata,
            # the 'instances/image_idxs' tensors will be populated while finalizing the dataset
            "img_count": len(imgs),
            # ...and the remaining entries should be raw data samples to be appended directly
            "instances/label": self.metadata.class_names.index(class_id),
            "instances/subset": self.metadata.subset_types.index(subset_id),
            "instances/id": instance_id,
        }

    def _export_sample_data(self, sample_index, sample_out):
        """Fetches a data sample from the getitem implementation and appends it to the output."""
        # we override the default handling since our tensor datasets dont all have the same length
        sample_data = self[sample_index]  # this is where the __getitem__ is called...
        assert "instances/image_idxs" not in sample_data
        image_tensors = {k: v for k, v in sample_data.items() if k.startswith("images/")}
        assert all([isinstance(d, list) and len(d) == sample_data["img_count"] for d in image_tensors.values()])
        # as of 2023-04-15, deeplake does not like parallel processing non-uniform-length datasets
        # (if you get a strange crash below, do the deeplake export with `num_workers=0`)
        sample_out.extend(image_tensors, skip_ok=True)
        instance_tensors = {k: v for k, v in sample_data.items() if k.startswith("instances/")}
        sample_out.append(instance_tensors, skip_ok=True)  # now, append all instance data (same length)
        return sample_out

    def _finalize_dataset_export(self, dataset: deeplake.Dataset) -> None:
        """Finalizes the exportation of the deeplake dataset, adding extra info as needed."""
        # in this case, we'll do a final pass to fill the "instances/image_idxs" tensor dataset
        global_img_counter = 0
        image_idxs = []
        for instance_idx in range(len(self)):
            instance_img_ids = self._idx_to_ids_map[instance_idx][-1]
            instance_img_count = len(instance_img_ids)
            image_idxs.append([global_img_counter + offset for offset in range(instance_img_count)])
            global_img_counter += instance_img_count
        images_dataset_name = f"images/{self.image_type}/{self.image_compression_type}"
        assert global_img_counter == len(dataset[images_dataset_name])
        dataset["instances/image_idxs"].extend(image_idxs)


def _repackage_fmow_rgb(dataset_root_path: pathlib.Path):
    assert dataset_root_path.is_dir(), f"missing dataset root directory: {dataset_root_path}"
    repackager = DeepLakeRepackager(
        dataset_root_path,
        image_type="rgb",
        image_compression_type="jpg",
        use_cache=True,
    )
    output_path = dataset_root_path / "fmow-rgb" / ".deeplake"
    # as of 2023-04-15, deeplake does not like parallel processing non-uniform-length datasets
    # (we use `num_workers=0` to fix this issue; repackaging with this takes ~5h though...)
    repackager.export(output_path, num_workers=0)
    assert pathlib.Path(output_path).exists()


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config(config_name="data_profiler.yaml")
    _repackage_fmow_rgb(pathlib.Path(config_.utils.data_root_dir) / "fmow")
