"""Implements a deeplake data repackager for the DISA dataset."""

import json
import pathlib
import typing

import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
import shapely
import tqdm

import ssl4rs.data.metadata.disa
import ssl4rs.data.repackagers.utils
import ssl4rs.utils.logging

logger = ssl4rs.utils.logging.get_logger(__name__)


class DeepLakeRepackager(ssl4rs.data.repackagers.utils.DeepLakeRepackager):
    """Repackages the DISA dataset into a deeplake-compatible format.

    The tensors that will be created are the following (see `tensor_info` for more details/types):
        sample_id: unique identifier used to tag each data sample (field+image set).
        image_preview: image used to give a quick preview of the input data (for demo/debug).
        image_preview_roi: binary mask of valid pixels in the above preview image.
        image_bbox: bounding box of the valid image region.
        field_mask: binary mask encoding whether each pixel is labeled as a field.
        field_geoms: original field geometries (polygons).
        image_data: stack of N 4-band images (N, 4, H, W) taken at different times.
        image_roi: stack of N binary masks (N, H, W) of valid pixels in the above stack.
        image_metadata: information about each of the N images in the above stack.
    """

    metadata = ssl4rs.data.metadata.disa

    @property  # we need to provide this for the base class!
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor info (declaration) arguments used during creation."""
        return dict(
            sample_id=dict(htype="text", dtype=str),
            image_preview=dict(htype="image.rgb", dtype=np.uint8, sample_compression="jpg"),
            image_preview_roi=dict(htype="binary_mask", dtype=bool, sample_compression="lz4"),
            image_bbox=dict(htype="point", dtype=np.float64, sample_compression=None),
            field_mask=dict(htype="binary_mask", dtype=bool, sample_compression="lz4"),
            field_geoms=dict(htype="polygon", dtype=np.float64, sample_compression=None),
            image_data=dict(htype="generic", dtype=np.uint16, sample_compression=None),
            image_roi=dict(htype="binary_mask", dtype=bool, sample_compression="lz4"),
            image_metadata=dict(htype="json", sample_compression=None),
        )

    @property  # we need to provide this for the base class!
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information that will be exported in the deeplake object."""
        return dict(
            name=self.dataset_name,
            band_count=self.metadata.band_count,
            dtype=self.metadata.raster_dtype.str,
            nodata_val=self.metadata.nodata_val,
            crs=self.metadata.crs,
            image_count_per_sample=self.image_count_per_sample,
        )

    @property  # we need to provide this for the base class!
    def dataset_name(self) -> str:
        """Returns the dataset name used to identify this particular dataset."""
        return "DISA"

    def __len__(self) -> int:
        """Returns the total number of images defined in this dataset."""
        return self.sample_count

    def __init__(
        self,
        dataset_root_path: typing.Union[typing.AnyStr, pathlib.Path],
    ):
        """Parses the dataset structure and makes sure all the data is present.

        Args:
            dataset_root_path: path to the directory containing all the DISA data.
        """
        super().__init__()
        self.data_root_path = pathlib.Path(dataset_root_path)
        assert self.data_root_path.exists(), f"invalid dataset path: {self.data_root_path}"
        annotations_file_path = self.data_root_path / "annotations.json"
        assert annotations_file_path.is_file(), f"invalid annotations file: {annotations_file_path}"
        with open(annotations_file_path) as annot_fd:
            annotations = json.load(annot_fd)
        annotations = self._deduplicate_annotations(annotations)
        annotations = self._validate_annotations(annotations)
        self.sample_count = len(annotations)
        self.annotations = annotations
        assert len(self.annotations) > 0, "no data?"
        self.image_count_per_sample = len(self.annotations[0]["image_paths"])
        # once we get here, we're ready to repackage the dataset!

    @staticmethod
    def _deduplicate_annotations(
        annotations: typing.List[typing.Dict],
    ) -> typing.List[typing.Dict]:
        """Returns the list of sample-wise annotations without duplicates."""
        # assumes samples with the same image id are always duplicates (simplest way to avoid leaks)
        output_annotations_map = {}  # image-id to annotation
        for annot in annotations:
            assert "image_id" in annot, "missing mandatory 'image_id' field in annotation dict"
            if annot["image_id"] not in output_annotations_map:
                output_annotations_map[annot["image_id"]] = annot
            else:
                assert output_annotations_map[annot["image_id"]] == annot
        return list(output_annotations_map.values())

    def _validate_annotations(
        self,
        annotations: typing.List[typing.Dict],
    ) -> typing.List[typing.Dict]:
        """Returns the list of sample-wie annotations, with all fields checked for validity."""
        valid_annotations = []
        expected_image_count = None
        for annot in tqdm.tqdm(annotations, desc="Converting annotation geometries + paths"):
            annot["sample_id"] = annot["image_id"]
            del annot["image_id"]
            try:
                assert "bbox" in annot, "missing 'bbox' field in annotation dict"
                annot["bbox"] = shapely.from_wkt(annot["bbox"])
                assert isinstance(annot["bbox"], shapely.Polygon), "unrecognized bbox type"
                assert "field_masks" in annot, "missing 'field_masks' field in annotation dict"
                annot["field_masks"] = shapely.from_wkt(annot["field_masks"])
                assert isinstance(annot["field_masks"], shapely.MultiPolygon), "unrecognized field masks type"
                annot["field_geoms"] = [p for p in annot["field_masks"].geoms]
                del annot["field_masks"]
                assert "image_path_n" in annot, "missing 'image_path_n' field in annotation dict"
                assert isinstance(annot["image_path_n"], list)
                image_paths = []
                for p in annot["image_path_n"]:
                    full_image_path = self.data_root_path / p
                    assert full_image_path.is_file(), f"missing image: {full_image_path}"
                    image_paths.append(full_image_path)
                if expected_image_count is None:
                    expected_image_count = len(image_paths)
                else:
                    assert len(image_paths) == expected_image_count, "unexpected image count"
                annot["image_paths"] = image_paths
                del annot["image_path_n"]
                valid_annotations.append(annot)
            except AssertionError as e:
                logger.warning(f"skipping annotation with id={annot['sample_id']}, reason: {e}")
        return valid_annotations

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Fetches and returns a data sample for this dataset."""
        assert 0 <= item < len(self), f"invalid data sample index being queried: {item}"
        sample_data = self.annotations[item]
        sample_id = sample_data["sample_id"]
        image_metadata = [{"name": p.name} for p in sample_data["image_paths"]]
        image_shape, image_preview, image_preview_roi, image_transform = None, None, None, None
        image_data, image_roi = [], []
        for image_idx, image_path in enumerate(sample_data["image_paths"]):
            with rasterio.open(image_path) as raster:
                assert raster.nodata == 0.0
                assert raster.crs == rasterio.crs.CRS.from_string(self.metadata.crs)  # noqa
                image = raster.read()
                reprojected = False
                if image_shape is None:
                    image_shape = image.shape
                    image_preview = self._generate_preview_image(image)
                    image_preview_roi = np.any(image != raster.nodata, axis=0)
                    image_transform = raster.transform
                else:
                    if image_shape != image.shape or image_transform != raster.transform:
                        logger.warning(
                            f"found unexpected/mismatched image resolution for sample {sample_id}"
                            f"\n\texpected shape: {image_shape}, found shape: {image.shape}"
                            f"\n\texpected t: {tuple(image_transform)}, found t: {tuple(raster.transform)}"
                            f"\n\t...will reproject this image: {image_path}"
                        )
                        target_image = np.empty(shape=image_shape, dtype=np.float32)
                        rasterio.warp.reproject(
                            image,
                            target_image,
                            src_transform=raster.transform,
                            src_crs=raster.crs,
                            src_nodata=raster.nodata,
                            dst_transform=image_transform,
                            dst_crs=raster.crs,
                            dst_nodata=raster.nodata,
                            resampling=rasterio.warp.Resampling.bilinear,
                        )
                        image = np.round(target_image).astype(np.uint16)
                        reprojected = True
                    assert image_shape == image.shape, "unexpected image shape mismatch"
                    assert image.shape[0] == self.metadata.band_count, "unexpected band count"
                assert image.dtype == np.uint16
                image_metadata[image_idx] = {**raster.meta, **image_metadata[image_idx]}
                image_metadata[image_idx]["crs"] = image_metadata[image_idx]["crs"].to_dict()
                image_metadata[image_idx]["transform"] = tuple(image_metadata[image_idx]["transform"])
                # note: to reconvert the above two attribs, use:
                # crs = rasterio.crs.CRS.from_dict(crs_dict)
                # transform = affine.Affine(*affine_tuple)
                image_metadata[image_idx]["reprojected"] = reprojected
                image_data.append(image)
                image_roi.append(image != raster.nodata)
        image_data = np.stack(image_data, axis=0)
        image_roi = np.stack(image_roi, axis=0)
        image_bbox = np.asarray(list(sample_data["bbox"].exterior.coords))
        assert len(image_bbox) == 5 and np.allclose(image_bbox[0], image_bbox[-1])
        field_geoms = [list(p.exterior.coords) for p in sample_data["field_geoms"]]
        field_mask_shapes = [(poly, 1) for poly in sample_data["field_geoms"]]
        field_mask = rasterio.features.rasterize(
            shapes=field_mask_shapes,
            out_shape=image_shape[1:],
            transform=image_transform,
        ).astype(bool)
        return dict(
            sample_id=f"sample_{sample_id}",
            image_preview=image_preview,
            image_preview_roi=image_preview_roi,
            image_bbox=image_bbox[:4],
            field_mask=field_mask,
            field_geoms=field_geoms,
            image_data=image_data,
            image_roi=image_roi,
            image_metadata=image_metadata,
        )

    @classmethod
    def _generate_preview_image(cls, image_raw: np.ndarray) -> np.ndarray:
        """Generates a preview RGB image of the given 4-band raster image."""
        # note: not using global stats for this!
        assert image_raw.ndim == 3
        assert image_raw.shape[0] == cls.metadata.band_count
        assert cls.metadata.band_count >= 3
        raster_data = image_raw[:3]
        raster_roi = np.asarray(raster_data != 0.0)
        raster_norm_data = np.zeros_like(raster_data, dtype=np.float32)
        # normalize in a bandwise fashion while ignoring nodata values
        for bidx in range(3):
            band_mean = raster_data[bidx][raster_roi[bidx]].mean()
            band_std = max(raster_data[bidx][raster_roi[bidx]].std(), 1)
            raster_norm_data[bidx][raster_roi[bidx]] = (raster_data[bidx][raster_roi[bidx]] - band_mean) / band_std
        # clamp to two std (95% of all data)
        raster_norm_data[raster_norm_data > 2] = 2
        raster_norm_data[raster_norm_data < -2] = -2
        # scale to 8-bit range
        preview_image = (((raster_norm_data + 2) / 4) * 255).astype(np.uint8)
        preview_image = np.transpose(preview_image, (1, 2, 0))
        return preview_image


def _repackage_disa(dataset_root_path: pathlib.Path):
    assert dataset_root_path.is_dir(), f"missing dataset root directory: {dataset_root_path}"
    repackager = DeepLakeRepackager(dataset_root_path)
    output_path = dataset_root_path / ".deeplake"
    repackager.export(output_path, overwrite=True, num_workers=0)
    assert pathlib.Path(output_path).exists()


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config(config_name="profiler.yaml")
    _repackage_disa(pathlib.Path(config_.utils.data_root_dir) / "disa")
