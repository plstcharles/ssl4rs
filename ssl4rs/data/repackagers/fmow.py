"""Implements a deeplake data repackager for the Functional Map of the World (fMoW) dataset.

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
    """Repackages the Functional Map of the World (fMoW) dataset into a deeplake format.

    To make sure high-speed dataloaders and visualizations are supported out-of-the-box, this
    repackager will make sure all tensor datasets have aligned image-wise samples. This means all
    samples correspond to a unique image, and the images for a single instance *should* be stored
    contiguously. The images will be stored along with their instance/label metadata.

    Note: running this repackager requires 16GB+ of RAM, as many JSON configs will be preloaded to
    memory to simplify/speed up mappings. When processing the RGB dataset with all subsets and JPEG
    compression, the resulting deeplake archive will be roughly 180GB (as of fMoW v1.2.1) without
    optimized views, and 450GB with optimized views.
    """

    metadata = ssl4rs.data.metadata.fmow

    @property  # we need to provide this for the base class!
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor declarations used during repackaging."""
        if self.image_type != "rgb":
            raise NotImplementedError("non-rgb dataset types not supported yet!")
        return {
            "image": dict(
                htype="image" if self.image_type == "rgb" else "generic",
                dtype=np.uint8 if self.image_type == "rgb" else np.int16,  # TODO: make sure int16 is OK?
                sample_compression=self.image_compression_type,
                tiling_threshold=-1,  # hidden param, normally in bytes, -1 = disables tiling
            ),
            "bbox": dict(htype="bbox", dtype=np.int32, coords=dict(type="pixel", mode="LTWH")),
            "metadata": dict(htype="json", sample_compression=None),
            "instance": dict(htype="generic", dtype=np.int32, sample_compression=None),
            "label": dict(htype="class_label", dtype=np.int16, class_names=self.metadata.class_names),
            "subset": dict(htype="class_label", dtype=np.int8, class_names=self.metadata.subset_types),
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
        return f"fMoW-{self.image_type}"

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
        """Loads the content of fMoW's CHANGELOG.md and reads the latest version from it."""
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
        logger.info(f"will load fMoW v{version}")
        return version

    def _load_manifest(self) -> typing.List:
        """Loads the `manifest.json.bz2` file contents to memory."""
        manifest_path = self.data_root_path / "manifest.json.bz2"
        assert manifest_path.is_file(), f"invalid manifest file: {manifest_path}"
        logger.info(f"parsing fMoW file: {manifest_path}")
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
        logger.info(f"parsing fMoW file: {groundtruth_path}")
        with open(groundtruth_path, mode="rb") as fd:
            tar_data = io.BytesIO(fd.read())
        gt_mappings, gt_data = dict(), dict()
        with tarfile.open(fileobj=tar_data, mode="r:bz2") as tar_fd:
            expected_total_members = 1047693  # seems OK as of fMoW v1.2.1 (tested on 2023-03-26)
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
                assert ids_tuple not in self._instance_ids_to_idx_map
                self._instance_ids_to_idx_map[ids_tuple] = self._total_instances
                self._instance_idx_to_ids_map[self._total_instances] = ids_tuple
                for img_id in img_ids:
                    self._image_idx_to_instance_idx_and_image_id_map[self._total_images] = (
                        self._total_instances,
                        img_id,
                    )
                    self._total_images += 1
                self._total_instances += 1
        assert self._total_instances > 0, "something went wrong..."

    def __init__(
        self,
        dataset_root_path: typing.Union[typing.AnyStr, pathlib.Path],
        image_type: typing.AnyStr = "rgb",
        image_compression_type: typing.AnyStr = "jpg",
        subset_type: typing.Union[typing.AnyStr, typing.List] = "all",
        use_cache: bool = False,
        optimize_views: bool = True,
    ):
        """Parses the dataset structure and makes sure all the data is present.

        Args:
            dataset_root_path: path to the directory containing all the fMoW data.
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
            assert subset_type in self.metadata.subset_types, f"unsupported fMoW subset type: {subset_type}"
            if subset_type == "all":
                self.subset_types = [s for s in self.metadata.subset_types if s != "all"]
            else:
                self.subset_types = [subset_type]
        else:
            assert all(
                [s in self.metadata.subset_types for s in subset_type]
            ), f"bad fMoW subset type(s): {subset_type}"
            self.subset_types = list(set(subset_type))
        self._use_cache = use_cache
        self._optimize_views = optimize_views
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
        self._image_idx_to_instance_idx_and_image_id_map = dict()  # im idx (int) -> inst idx (int), im id
        self._instance_idx_to_ids_map = dict()  # instance idx (int) -> subset, class, instance, images
        self._instance_ids_to_idx_map = dict()  # subset, class, instance, images -> instance idx (int)
        self._total_instances = 0
        self._total_images = 0
        self._prepare_maps()
        # once we get here, we're ready to repackage the dataset!

    def __len__(self) -> int:
        """Returns the total number of images defined in this dataset.

        Note: each instance may have more than one image!
        """
        return self._total_images

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Fetches and returns a data sample (image) for this dataset.

        In this implementation, a 'data sample' is actually an image of an instance inside the
        dataset. The same instance may have more than one image associated with it. Each image is
        captured at a different time. Each image is linked with its own metadata (which includes
        groundtruth info).

        Note that this code will likely be slower than the deeplake fetching implementation, thus
        why this is a "repackager" object, and not a dataset parser (although it could be used as
        one...).
        """
        assert 0 <= item < len(self), f"invalid data sample index being queried: {item}"
        instance_idx, image_id = self._image_idx_to_instance_idx_and_image_id_map[item]
        (subset_id, class_id, instance_id, img_ids) = self._instance_idx_to_ids_map[instance_idx]
        assert subset_id in self.subset_types
        assert class_id in self.metadata.class_names
        img_dict = self._metadata[subset_id][instance_id][image_id]
        # using deeplake.read will defer loading the full image data if possible/needed
        img = deeplake.read(str(img_dict["file_abs_path"]))
        metadata = img_dict["metadata"]
        gt_bbox = metadata["bounding_boxes"][0]
        assert gt_bbox["ID"] == -1  # meaning the 'ground truth' bbox
        gt_bbox = np.asarray(gt_bbox["box"], dtype=np.int32)  # should be in LTWH format
        return {  # note: the tensor names here must match with the ones in `tensor_info`!
            "image": img,
            "bbox": gt_bbox,
            "metadata": metadata,
            "instance": instance_idx,
            "label": self.metadata.class_names.index(class_id),
            "subset": self.metadata.subset_types.index(subset_id),
        }

    def _finalize_dataset_export(
        self,
        dataset: deeplake.Dataset,
        num_workers: int,
    ) -> None:
        """Finalizes the exportation of the deeplake dataset, adding extra info as needed."""
        # in this case, we'll create optimized views for the different subsets
        dataset.rechunk(num_workers=num_workers)
        dataset.commit("base")
        for subset_type in self.subset_types:
            logger.info(f"creating view for '{subset_type}' subset...")
            subset = dataset.filter(f"subset == {self.metadata.subset_types.index(subset_type)}")
            logger.debug(f"subset has {len(subset)} samples")
            if len(subset) > 0:
                subset.save_view(id=subset_type, optimize=self._optimize_views, num_workers=num_workers)


def _repackage_fmow_rgb(dataset_root_path: pathlib.Path):
    assert dataset_root_path.is_dir(), f"missing dataset root directory: {dataset_root_path}"
    repackager = DeepLakeRepackager(
        dataset_root_path,
        image_type="rgb",
        image_compression_type="jpg",
        use_cache=True,
    )
    output_path = dataset_root_path / "fmow-rgb" / ".deeplake"
    repackager.export(output_path, num_workers=2)  # fewer threads to make sure 32GB ram is enough
    assert pathlib.Path(output_path).exists()


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config()
    _repackage_fmow_rgb(pathlib.Path(config_.utils.data_root_dir) / "fmow")
