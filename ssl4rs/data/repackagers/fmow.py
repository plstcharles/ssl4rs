"""Implements a deep lake data repackager for the Functional Map of the World (FMoW) dataset.

See the following URLs for more info on this dataset:
https://arxiv.org/abs/1711.07846
https://github.com/fMoW/dataset
https://spacenet.ai/iarpa-functional-map-of-the-world-fmow/
"""
import bz2
import io
import json
import pathlib
import re
import tarfile
import typing

import PIL.Image
import deeplake
import numpy as np
import tqdm

import ssl4rs.data.repackagers.utils
import ssl4rs.utils.imgproc

expected_max_fmow_pixels = 20000 * 20000  # some fmow images are big!
PIL.Image.MAX_IMAGE_PIXELS = max(expected_max_fmow_pixels, PIL.Image.MAX_IMAGE_PIXELS)
logger = ssl4rs.utils.logging.get_logger(__name__)


class DeepLakeRepackager(ssl4rs.data.repackagers.utils.DeepLakeRepackager):
    """Repackages the Functional Map of the World (FMoW) dataset into a deeplake format.

    Each class instance will be stored as a group with all its views as single samples in order to
    help do cross-view sampling. Each view (image) will be stored in its original binary format
    with its associated GSD and metadata as extra attributes.

    Note: running this repackager requires 16GB+ of RAM, as many JSON configs will be preloaded to
    memory to simplify/speed up mappings.
    """

    class_names = [
        "airport",
        "airport_hangar",
        "airport_terminal",
        "amusement_park",
        "aquaculture",
        "archaeological_site",
        "barn",
        "border_checkpoint",
        "burial_site",
        "car_dealership",
        "construction_site",
        "crop_field",
        "dam",
        "debris_or_rubble",
        "educational_institution",
        "electric_substation",
        "factory_or_powerplant",
        "fire_station",
        "flooded_road",
        "fountain",
        "gas_station",
        "golf_course",
        "ground_transportation_station",
        "helipad",
        "hospital",
        "impoverished_settlement",
        "interchange",
        "lake_or_pond",
        "lighthouse",
        "military_facility",
        "multi-unit_residential",
        "nuclear_powerplant",
        "office_building",
        "oil_or_gas_facility",
        "park",
        "parking_lot_or_garage",
        "place_of_worship",
        "police_station",
        "port",
        "prison",
        "race_track",
        "railway_bridge",
        "recreational_facility",
        "road_bridge",
        "runway",
        "shipyard",
        "shopping_mall",
        "single-unit_residential",
        "smokestack",
        "solar_farm",
        "space_facility",
        "stadium",
        "storage_tank",
        "surface_mine",
        "swimming_pool",
        "toll_booth",
        "tower",
        "tunnel_opening",
        "waste_disposal",
        "water_treatment_facility",
        "wind_farm",
        "zoo",
    ]
    """List of class names used in the dataset (still using a capital 1st letter for each noun)."""

    supported_image_types = ["rgb", "full"]
    """List of supported raw image types for repackaging.

    The 'RGB' dataset corresponds to multispectral or panchromatic-based RGB images. The 'full'
    dataset corresponds to the 4-band or 8-band multispectral images.
    """

    supported_subset_types = ["train", "val", "test", "seq", "all"]
    """List of supported split subsets that can be repackaged."""

    @property  # we need to provide this for the base class!
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor info (declaration) arguments used during creation."""
        assert self.image_type == "rgb", "other dataset types not supported yet! @@@@"
        return dict(
            views=dict(htype="sequence[image.rgb]", dtype=np.uint8, sample_compression="jpg"),
            metadata=dict(htype="sequence[json]", sample_compression=None),
            bboxes=dict(htype="sequence[bbox]", dtype=np.int32, coords=dict(type="pixel", mode="LTRB")),
            label=dict(htype="class_label", dtype=np.int16, class_names=self.class_names),
            subset=dict(htype="class_label", dtype=np.int8, class_names=self._expected_subset_types),
            instance=dict(htype="text", dtype=str, sample_compression=None),
        )

    @property  # we need to provide this for the base class!
    def dataset_info(self):
        """Returns metadata information that will be exported in the deeplake object."""
        return dict(
            name=self.dataset_name,
            class_names=self.class_names,
            image_type=self.image_type,
            subset_types=self.subset_types,
            version=self.version,
        )

    @property  # we need to provide this for the base class!
    def dataset_name(self):
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
        """Inserts a value inside a dictionary with a recursive key, creating subdicts if needed."""
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
        """Loads the content of `CHANGELOG.md` and reads the latest dataset version from it."""
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

    def _load_manifest(self):
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
        return gt_mappings, gt_data

    def _parse_metadata(self) -> typing.Dict:
        """Parses the metadata of all instances using the groundtruth json data."""
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
                assert class_id in self.class_names
                instance_metadata = self._gt_data[remapped_subset_id][class_id][instance_id]
            else:
                # we can extract the class/instance ids as-is
                assert len(file_name_tokens) == 4, "unexpected file name tokens for train/val set"
                subset_id, class_id, instance_id, file_id = file_name_tokens
                assert subset_id in self.subset_types
                assert class_id in self.class_names
                instance_metadata = self._gt_data[subset_id][class_id][instance_id]
            assert isinstance(instance_metadata, dict)
            for curr_view_meta in instance_metadata.values():
                assert "bounding_boxes" in curr_view_meta
                assert isinstance(curr_view_meta["bounding_boxes"], list)
                # there should only be one groundtruth bbox per entire view/image
                assert sum([bbox["ID"] == -1 for bbox in curr_view_meta["bounding_boxes"]]) == 1
                # ... and it should be the first in the list
                assert curr_view_meta["bounding_boxes"][0]["ID"] == -1
            assert instance_id.startswith(class_id)
            assert file_id.count("_") >= 2
            view_id = file_id.rsplit("_", maxsplit=2)[-2]
            assert int(view_id) >= 0
            view_metadata_key = f"{instance_id}_{view_id}_{self.image_type}.json"
            view_metadata = instance_metadata[view_metadata_key]
            assert isinstance(instance_metadata, dict)
            assert not file_path.endswith(".json")
            file_abs_path = self.data_root_path / file_path
            assert file_abs_path.is_file(), f"missing file: {file_path}"
            self._insert_with_recursive_key(
                out_dict=metadata,
                key=subset_id + "/" + instance_id + f"/view_{view_id}",
                data=dict(
                    file_abs_path=file_abs_path,
                    metadata=view_metadata,
                    subset_id=subset_id,
                    class_id=class_id,
                    instance_id=instance_id,
                    file_id=file_id,
                    view_id=view_id,
                ),
                overwrite_if_needed=False,
            )
        return metadata

    def _prepare_maps(self) -> None:
        """Prepares the sample index and instance ID maps used to fetch specific data."""
        assert self._total_samples == 0, "should only call once!"
        for subset_id, instance_ids in self._metadata.items():
            for instance_id, view_ids in self._metadata[subset_id].items():
                assert len(list(view_ids)) > 0
                views = [self._metadata[subset_id][instance_id][view_id] for view_id in view_ids]
                assert len(set([v["class_id"] for v in views])) == 1
                class_id = views[0]["class_id"]
                ids_tuple = (subset_id, class_id, instance_id, tuple(view_ids))
                assert ids_tuple not in self._ids_to_idx_map
                self._ids_to_idx_map[ids_tuple] = self._total_samples
                self._idx_to_ids_map[self._total_samples] = ids_tuple
                self._total_samples += 1
                self._total_views += len(view_ids)
        assert self._total_samples > 0, "something went wrong..."

    def __init__(
        self,
        dataset_root_path: typing.Union[typing.AnyStr, pathlib.Path],
        image_type: typing.AnyStr = "rgb",
        subset_type: typing.Union[typing.AnyStr, typing.List] = "all",
    ):
        """Parses the dataset structure and makes sure all the data is present.

        Args:
            dataset_root_path: path to the directory containing all the FMoW data.
            image_type: image type to extract; should be either 'rgb' for the RGB-only images,
                or 'full' for the multispectral images.
            subset_type: subset type to extract; should be train/val/test/seq, or 'all' for their
                combination.
        """
        assert image_type in self.supported_image_types
        self.image_type = image_type
        self._expected_subset_types = ["train", "val", "test", "seq"]
        if not isinstance(subset_type, list):
            assert subset_type in self.supported_subset_types, f"unsupported subset type: {subset_type}"
            if subset_type == "all":
                self.subset_types = self._expected_subset_types
            else:
                self.subset_types = [subset_type]
        else:
            assert all([s in self._expected_subset_types for s in subset_type]), "bad subset type(s)"
            self.subset_types = list(set(subset_type))
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
        self._idx_to_ids_map = dict()  # sample idx (int) -> subset, class, instance, views
        self._ids_to_idx_map = dict()  # subset, class, instance, views -> sample idx (int)
        self._total_samples = 0
        self._total_views = 0
        self._prepare_maps()
        # once we get here, we're ready to repackage the dataset!

    def __len__(self):
        """Returns the total number of instances defined in this dataset.

        Note: each instance may have more than one view, meaning more than one image!
        """
        return self._total_samples

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Fetches and returns a data sample for this dataset.

        In this implementation, a 'data sample' is actually an instance of a class inside the
        dataset. It may therefore have more than one temporal view (image), and each view
        will be linked with their own metadata (which includes groundtruth info).

        Note that this code will likely be slower than the deeplake fetching implementation, thus
        why this is a "repackager" object, and not a dataset parser (although it could be used as
        one...).
        """
        assert 0 <= item < len(self), f"invalid data sample index being queried: {item}"
        (subset_id, class_id, instance_id, view_ids) = self._idx_to_ids_map[item]
        assert subset_id in self.subset_types
        assert class_id in self.class_names
        view_dicts = list(self._metadata[subset_id][instance_id].values())
        # using deeplake.read will defer loading the full image data if possible/needed
        views = [deeplake.read(str(view["file_abs_path"])) for view in view_dicts]
        metadata = [view["metadata"] for view in view_dicts]
        gt_bboxes = [m["bounding_boxes"][0] for m in metadata]
        assert all([bbox["ID"] == -1 for bbox in gt_bboxes])  # meaning the 'ground truth' bbox
        bboxes = [np.asarray(bbox["box"], dtype=np.int32) for bbox in gt_bboxes]
        return dict(  # note: the tensor names here must match with the ones in `tensor_info`!
            views=views,
            metadata=metadata,
            bboxes=bboxes,
            label=self.class_names.index(class_id),
            subset=self._expected_subset_types.index(subset_id),
            instance=instance_id,
        )


def _repackage_fmow_rgb(dataset_root_path: pathlib.Path):
    assert dataset_root_path.is_dir(), f"missing dataset root directory: {dataset_root_path}"
    repackager = DeepLakeRepackager(dataset_root_path, image_type="rgb")
    output_path = dataset_root_path / "fmow-rgb" / ".deeplake"
    repackager.export(output_path)
    assert pathlib.Path(output_path).exists()


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config(config_name="data_profiler.yaml")
    _repackage_fmow_rgb(pathlib.Path(config_.utils.data_root_dir) / "fmow")
