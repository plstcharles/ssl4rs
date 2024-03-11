"""Implements a deeplake data repackager for the Mila-AI4H-DISA-India dataset."""
import copy
import dataclasses
import datetime
import functools
import itertools
import json
import pathlib
import typing

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.plot
import rasterio.warp
import shapely
import shapely.ops
import tqdm

import ssl4rs.data.metadata.disa
import ssl4rs.data.parsers.utils.geopandas_utils as gpd_utils
import ssl4rs.data.repackagers.utils
import ssl4rs.utils.logging

logger = ssl4rs.utils.logging.get_logger(__name__)


@dataclasses.dataclass
class PolygonData:
    """Holds data related to a single polygon parsed from Sherrie Wang's dataset."""

    shp_idx: int
    """Index of the corresponding polygon in the original annotations shapefile."""

    location_id: str
    """Id of this polygon's parent location, derived from the idx in the original shapefile."""

    geometry: shapely.geometry.Polygon
    """Polygon in its original latlon coordinates format."""

    _reproj_geometry: shapely.geometry.Polygon
    """Polygon in a reprojected metric coordinates format (for internal use only)."""


@dataclasses.dataclass
class LocationData:
    """Holds identifiers and data related to a single location in Sherrie Wang's dataset."""

    identifier: str
    """Identifier for this location, derived from the index in the original shapefile."""

    polygons: typing.List[PolygonData]
    """Annotated polygons at this location."""

    centroid: shapely.geometry.point.Point
    """Centroid coordinates computed based on all polygons at this location."""

    max_polygon_distance: float
    """Maximum distance between all polygons at this location."""

    max_polygon_radius: float
    """Maximum bounding radius of all polygons at this location."""

    scatter_ratio: float
    """Ratio of max polygon distance to max polygon diameter for this location.

    May be used to filter this location so that it is NOT exported for later use.
    """

    mask_valid_px_ratio: float
    """Approximate ratio of annotated (field) pixels for non-annotated (background) pixels.

    The approximation is done based on the expected image shape and resolution.

    May be used to filter this location so that it is NOT exported for later use.
    """

    subset: typing.Optional[str]
    """Subset for this particular location in Sherrie Wang's original data split.

    May be `None` if no match was found for this location and Sherrie Wang's split CSV.
    """


@dataclasses.dataclass
class OrderInfo:
    """Holds information about a fulfilled order and its resulting raster data."""

    identifier: str
    """Identifier for the order that was fulfilled, resulting in this raster data."""

    full_id: str
    """The 'full' id is the prefix data dirs + order id + product type + planet item id."""

    planet_item_id: str
    """The prefix used to identify the planet product items in this order.'."""

    order_metadata: dict
    """The metadata tied to this order (product acquisition parameters and attributes)."""

    order_timestamp: datetime.datetime
    """The datetime tied to this order (i.e. its acquisition timestamp)."""

    raster_xml_metadata: str
    """The XML (unparsed, str) metadata tied to the raster in this order."""

    raster_centroid: shapely.geometry.point.Point
    """The center x,y coords of the raster in this order, projected to the target CRS."""

    raster_bbox: shapely.geometry.base.BaseGeometry
    """The bounding box of the raster in this order, projected to the target CRS."""

    root_dir: pathlib.Path
    """The path to the directory where all order data can be found."""

    raster_udm2_path: pathlib.Path
    """The path to the UDM2 mask geotiff file provided in this order."""

    raster_data_path: pathlib.Path
    """The path to the raster data geotiff file provided in this order."""

    raster_data_valid_ratio: float
    """The fraction of valid pixels in the raster data for this order."""

    raster_orig_data_shape: typing.Tuple[int, int]
    """The original shape (height, width) of the pre-projection raster in this order."""

    raster_orig_transform: typing.Tuple[float, float, float, float, float, float]
    """The original affine transform of the pre-projection raster in this order."""

    raster_orig_bounds: typing.Tuple[float, float, float, float]
    """The original bounds of the pre-projection raster in this order."""

    raster_orig_crs: rasterio.crs.CRS
    """The original CRS of the pre-projection raster in this order."""


class DeepLakeRepackager(ssl4rs.data.repackagers.utils.DeepLakeRepackager):
    """Repackages the Mila-AI4H-DISA-India dataset into a deeplake-compatible format.

    This version of the repackager is meant to repackage the original (RAW) dataset, not the
    sample-ready version. This means it will parse the field shapes directly from the original
    shapefile provided by Sherrie Wang (https://zenodo.org/records/7315090) and associate them
    with the PSScene images ordered from Planet.

    Note that part of the locations may be skipped by this repackager based on whether a scatter
    ratio threshold is provided to the constructor. Also, some of the orders can be ignored if
    there are already many orders per location. See the constructor for more information.

    For more information on the tensors contained in the repackaged dataset, see the `tensor_info`
    function below.
    """

    metadata = ssl4rs.data.metadata.disa

    @property  # we need to provide this for the base class!
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor declarations used during repackaging."""
        return self.metadata.tensor_info_dicts

    @property  # we need to provide this for the base class!
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information that will be exported in the deeplake object.

        This metadata is a combination of information read from the dataset itself (e.g. the number
        of bands/channels per raster image, which should be constant) and of information provided to
        the constructor regarding how to filter locations/orders.
        """
        return dict(
            name=self.dataset_name,
            dtype=self.metadata.raster_dtype.str,
            nodata_val=self.metadata.nodata_val,
            band_count=self.raster_band_count,
            crs=self.metadata.crs,
            # orig input hyperparameters:
            max_scatter_threshold=self.max_scatter_threshold,
            max_order_count=self.max_order_count,
            max_split_location_distance=self.max_split_location_distance,
            min_valid_pixel_ratio_in_orders=self.min_valid_pixel_ratio_in_orders,
            min_valid_pixel_ratio_in_masks=self.min_valid_pixel_ratio_in_masks,
            min_time_delta_between_orders=self.min_time_delta_between_orders.total_seconds(),
            discard_subset_assignment_on_overlap=self.discard_subset_assignment_on_overlap,
        )

    @property  # we need to provide this for the base class!
    def dataset_name(self) -> str:
        """Returns the dataset name used to identify this particular dataset."""
        return "ai4h-disa"

    def __len__(self) -> int:
        """Returns the total number of valid location covered in this dataset."""
        return len(self.output_samples)

    def _parse_location_info(self) -> typing.Dict[str, LocationData]:
        """Parses and returns location data from Sherrie Wang's dataset (via its shapefile)."""
        shapefile_path = self.data_root_path / "boundary_polygons" / "india_10k_fields.shp"
        shapefile_hash = ssl4rs.utils.filesystem.get_file_hash(shapefile_path)
        assert shapefile_hash == self.metadata.shapefile_md5sum, f"md5sum mismatch: {shapefile_path}"
        geom_parser = gpd_utils.GeoPandasParser(shapefile_path, new_crs=self.metadata.crs)
        assert not geom_parser.has_duplicates()
        target_crs = rasterio.crs.CRS.from_string(self.metadata.crs)  # noqa
        target_crs_meters = rasterio.crs.CRS.from_string(self.metadata._crs_with_meter_units)  # noqa
        assert geom_parser.crs == target_crs
        assert len(geom_parser) == self.metadata.shapefile_polygon_count
        location_ids = [self._convert_location_id(s) for s in geom_parser.dataset["sample"].unique().tolist()]
        assert len(location_ids) == self.metadata.shapefile_location_count
        logger.info(f"parsing data for {len(geom_parser)} polygons across {len(location_ids)} locations")
        split_csv_path = self.data_root_path / "india_splits_grid20x20_v2.csv"
        expected_split_df_cols = ["image_id", "lat", "lon", "fold"]
        if split_csv_path.is_file():
            split_data = pd.read_csv(split_csv_path)
            assert all([col in split_data.columns for col in expected_split_df_cols])
            split_points = [shapely.geometry.Point(xy) for xy in zip(split_data.lon, split_data.lat)]
            split_data = gpd.GeoDataFrame(split_data, geometry=split_points)
            assert set(split_data["fold"].unique()) == set(self.metadata.location_subset_labels)
            logger.info(f"found split file with {len(split_data)} locations assigned to subsets")
        else:
            split_data = gpd.GeoDataFrame(columns=expected_split_df_cols, geometry=[])  # init w/ empty dataframe
            logger.warning(f"found no split file at: {split_csv_path}")
        split_data.crs = "EPSG:4326"  # set to same default as shapefile, as the CSV has no info on CRS
        split_data = split_data.to_crs(self.metadata.crs)  # update to new CRS (from default) if needed
        polygon_data_array_per_location = {lid: [] for lid in location_ids}
        gdf_converted = geom_parser.dataset.to_crs(target_crs_meters)
        for polygon_idx in range(len(geom_parser)):
            polygon_data = geom_parser[polygon_idx]
            location_id = self._convert_location_id(polygon_data["sample"])
            assert len(polygon_data["geometry"].exterior.coords) > 1
            polygon_data_array_per_location[location_id].append(
                PolygonData(
                    shp_idx=polygon_idx,
                    location_id=location_id,
                    geometry=polygon_data["geometry"],  # in latlon coords in the target CRS
                    _reproj_geometry=gdf_converted["geometry"][polygon_idx],  # in meter coords
                )
            )
        polygon_counts_per_location = [len(p) for p in polygon_data_array_per_location.values()]
        assert np.array_equal(np.unique(polygon_counts_per_location), [5, 6])

        location_info = {}
        for location_id, polygon_data_array in polygon_data_array_per_location.items():
            multipoly = shapely.geometry.MultiPolygon([p.geometry for p in polygon_data_array])
            max_bounding_radius = max(
                [shapely.minimum_bounding_radius(p._reproj_geometry) for p in polygon_data_array]  # noqa
            )
            polygon_data_combos = list(itertools.combinations(polygon_data_array, 2))
            max_distance_meters = max(
                [p1._reproj_geometry.distance(p2._reproj_geometry) for p1, p2 in polygon_data_combos]  # noqa
            )
            scatter_ratio = max_distance_meters / (max_bounding_radius * 2)
            if self.max_scatter_threshold is not None and scatter_ratio >= self.max_scatter_threshold:
                logger.info(
                    f"bad location: {location_id}\n\t"
                    f"(did not meet maximum scatter threshold;"
                    f" got {scatter_ratio:.2f}, needed less than {self.max_scatter_threshold:.2f})"
                )
                continue
            # below, we draw a preview of the field mask w/ default settings to compute the mask ratio
            field_mask_shapes = [(poly._reproj_geometry, 1) for poly in polygon_data_array]
            field_mask_multipoly = shapely.geometry.MultiPolygon([p._reproj_geometry for p in polygon_data_array])
            field_mask_topleft = (  # based on expected max image shape + expected image resolution
                field_mask_multipoly.centroid.y  # in meter coords due to reproj above
                - (self.metadata.max_image_shape[0] / 2) * self.metadata.approx_px_resolution,
                field_mask_multipoly.centroid.x  # in meter coords due to reproj above
                - (self.metadata.max_image_shape[1] / 2) * self.metadata.approx_px_resolution,
            )
            field_mask_default_transform = rasterio.transform.from_bounds(
                west=field_mask_topleft[1],
                south=field_mask_topleft[0],
                east=(field_mask_topleft[1] + self.metadata.max_image_shape[1] * self.metadata.approx_px_resolution),
                north=(field_mask_topleft[0] + self.metadata.max_image_shape[1] * self.metadata.approx_px_resolution),
                width=self.metadata.max_image_shape[1],
                height=self.metadata.max_image_shape[0],
            )
            field_mask = rasterio.features.rasterize(
                shapes=field_mask_shapes,
                out_shape=self.metadata.max_image_shape,
                transform=field_mask_default_transform,
            ).astype(bool)
            mask_valid_px_ratio = np.count_nonzero(field_mask) / field_mask.size
            if (
                self.min_valid_pixel_ratio_in_masks is not None
                and mask_valid_px_ratio < self.min_valid_pixel_ratio_in_masks
            ):
                logger.info(
                    f"bad location: {location_id}\n\t"
                    f"(did not meet minimum mask pixel ratio threshold;"
                    f" got {mask_valid_px_ratio:.5f}, needed more than {self.min_valid_pixel_ratio_in_masks:.5f})"
                )
            matched_split_locations = [
                split_data.iloc[geom_idx]
                for geom_idx, geom in enumerate(split_data.geometry)
                if geom.distance(multipoly) <= self.max_split_location_distance
            ]
            if len(matched_split_locations) > 1:  # that's pretty close, but should be in same subset
                if len({loc["fold"] for loc in matched_split_locations}) != 1:
                    matched_image_ids = [str(loc["image_id"]) for loc in matched_split_locations]
                    logger.warning(
                        "matched more than one loc in more than one subset with the current dist threshold"
                        f"\n\t({location_id} matched split csv ids: {', '.join(matched_image_ids)})"
                    )
                    if self.discard_subset_assignment_on_overlap:
                        # we will drop this subset assignment in order to avoid any leak
                        matched_split_locations = []
                # we'll sort the results by distance and take the closest
                matched_split_locations = sorted(
                    matched_split_locations,
                    key=lambda loc: loc.geometry.distance(multipoly),
                )
            matched_split_subset = None
            if matched_split_locations:
                matched_split_subset = matched_split_locations[0]["fold"]
            location_info[location_id] = LocationData(
                identifier=location_id,
                polygons=polygon_data_array,
                centroid=multipoly.centroid,
                max_polygon_distance=max_distance_meters,
                max_polygon_radius=max_bounding_radius,
                scatter_ratio=scatter_ratio,
                mask_valid_px_ratio=mask_valid_px_ratio,
                subset=matched_split_subset,
            )
        return location_info

    def _parse_order_info(self) -> typing.Tuple[typing.Dict[str, OrderInfo], int]:
        """Parses and returns the orders available on disk + the number of bands in all rasters."""
        raw_data_root_path = self.data_root_path / "raw_data"
        assert raw_data_root_path.is_dir(), f"invalid dataset path: {raw_data_root_path}"
        order_manifest_paths = list(raw_data_root_path.rglob("manifest.json"))
        assert len(order_manifest_paths) > 0, f"no order manifests found in: {raw_data_root_path}"
        target_crs = rasterio.crs.CRS.from_string(self.metadata.crs)  # noqa
        logger.info(f"parsing order manifests and data for {len(order_manifest_paths)} orders")
        order_info, raster_band_count = {}, None
        for order_manifest_path in order_manifest_paths:
            order_result_dir = order_manifest_path.parent
            order_id = order_result_dir.name
            with open(order_manifest_path) as order_manifest_fd:
                order_manifest = json.load(order_manifest_fd)
            planet_item_id, skip_order = None, False
            for expected_file_suffix in self.metadata.psscene_file_names:
                assert any([s["path"].endswith(expected_file_suffix) for s in order_manifest["files"]])
            for order_file_info in order_manifest["files"]:
                assert order_file_info["annotations"]["planet/item_type"] == "PSScene"
                if planet_item_id is None:
                    planet_item_id = order_file_info["annotations"]["planet/item_id"]
                else:
                    assert planet_item_id == order_file_info["annotations"]["planet/item_id"]
                order_file_path = order_result_dir / order_file_info["path"]
                if not order_file_path.is_file():
                    logger.info(f"bad order: {order_id}\n\t(missing file: {order_file_path.name})")
                    skip_order = True
                    break
                expected_md5sum = order_file_info["digests"]["md5"]
                md5sum = ssl4rs.utils.filesystem.get_file_hash(order_file_path)
                assert md5sum == expected_md5sum, f"MD5 mismatch for {order_file_path}"
                order_file_name = order_file_path.name
                is_expected = any([order_file_name.endswith(s) for s in self.metadata.psscene_file_names])
                assert is_expected, f"unexpected order file: {order_file_path}"
            if skip_order:
                continue  # reason given in info message above, i.e. missing file(s)
            # full id is the prefix dir(s) + order id + product type + planet item id
            full_id = str(order_result_dir.relative_to(raw_data_root_path) / "PSScene" / planet_item_id)
            order_product_path = order_result_dir / "PSScene"
            order_metadata_path = order_product_path / (planet_item_id + "_metadata.json")
            with open(order_metadata_path) as order_metadata_fd:
                order_metadata = json.load(order_metadata_fd)
            raster_xml_metadata_path = order_product_path / (planet_item_id + "_3B_AnalyticMS_metadata_clip.xml")
            with open(raster_xml_metadata_path) as raster_xml_metadata_fd:
                raster_xml_metadata = raster_xml_metadata_fd.read()
            udm2_mask_path = order_product_path / (planet_item_id + "_3B_udm2_clip.tif")
            with rasterio.open(udm2_mask_path) as udm2:
                raster_data_path = order_product_path / (planet_item_id + "_3B_AnalyticMS_clip.tif")
                with rasterio.open(raster_data_path) as raster:
                    if raster_band_count is None:
                        raster_band_count = raster.count
                    assert udm2.crs == raster.crs
                    assert udm2.height == raster.height
                    assert udm2.width == raster.width
                    assert udm2.bounds == raster.bounds
                    assert udm2.shape == raster.shape
                    assert udm2.transform == raster.transform
                    assert raster.nodata == self.metadata.nodata_val
                    assert raster.count == raster_band_count
                    assert all([dt == self.metadata.raster_dtype for dt in raster.dtypes])
                    raster_data = raster.read()
                    raster_data_shape = raster_data.shape[1:]
                    raster_data_transform = raster.transform
                    raster_data_bounds = raster.bounds
                    raster_crs = raster.crs
                    udm2_data = udm2.read()
                    assert udm2_data.shape == (8, *raster_data_shape)
                    usable_data_mask = np.logical_and(
                        np.any(raster_data != self.metadata.nodata_val, axis=0),
                        udm2_data[7] == 0,  # band #8 in udm2 = unusable pixel bits due to anomalies
                    ).flatten()
                    raster_data_valid_ratio = np.count_nonzero(usable_data_mask) / usable_data_mask.size
                    if (
                        self.min_valid_pixel_ratio_in_orders is not None
                        and raster_data_valid_ratio < self.min_valid_pixel_ratio_in_orders
                    ):
                        logger.info(
                            f"bad order: {order_id}\n\t"
                            f"(did not meet minimum valid ratio;"
                            f" got {raster_data_valid_ratio:.2%} valid pixels,"
                            f" needed more than {self.min_valid_pixel_ratio_in_orders:.2%})"
                        )
                        continue
                    raster_bbox = shapely.geometry.box(*raster_data_bounds)
                    if raster_crs != target_crs:
                        raster_bbox_geojson = rasterio.warp.transform_geom(
                            src_crs=raster_crs,
                            dst_crs=target_crs,
                            geom=raster_bbox,
                        )
                        raster_bbox = shapely.geometry.shape(raster_bbox_geojson)
                    raster_centroid = raster_bbox.centroid
            order_timestamp = datetime.datetime.strptime(
                order_metadata["properties"]["acquired"],
                "%Y-%m-%dT%H:%M:%S.%fZ",
            )
            order_info[order_id] = OrderInfo(
                identifier=order_id,
                full_id=full_id,
                root_dir=order_result_dir,
                planet_item_id=planet_item_id,
                order_metadata=order_metadata,
                order_timestamp=order_timestamp,
                raster_xml_metadata=raster_xml_metadata,
                raster_centroid=raster_centroid,
                raster_bbox=raster_bbox,
                raster_udm2_path=udm2_mask_path,
                raster_data_path=raster_data_path,
                raster_data_valid_ratio=raster_data_valid_ratio,
                raster_orig_data_shape=raster_data_shape,
                raster_orig_transform=raster_data_transform,
                raster_orig_bounds=raster_data_bounds,
                raster_orig_crs=raster_crs,
            )
        return order_info, raster_band_count

    def __init__(
        self,
        dataset_root_path: typing.Union[typing.AnyStr, pathlib.Path],
        max_scatter_threshold: typing.Optional[float] = 10.0,
        max_order_count: typing.Optional[int] = 12,
        max_split_location_distance: float = 0.05,  # in degrees
        min_valid_pixel_ratio_in_orders: typing.Optional[float] = 0.50,
        min_valid_pixel_ratio_in_masks: typing.Optional[float] = 0.001,
        min_time_delta_between_orders: typing.Optional[datetime.timedelta] = datetime.timedelta(days=1),
        discard_subset_assignment_on_overlap: bool = False,
    ):
        """Parses the dataset structure and makes sure all the data is present and valid.

        This is where the filtering of locations and orders will be done, if required. The min/max
        values provided to the constructor are all optional and control this filtering.

        Args:
            dataset_root_path: path to the directory containing all the DISA data.
            max_scatter_threshold: the maximum scatter ratio value that can be allowed for
                locations in the dataset, beyond which the location is skipped. The 'scatter
                ratio' defines how far away polygons are from each other for a single location.
            max_order_count: the maximum number of orders to export per location in the dataset,
                beyond which some orders will be ignored. If `None`, all orders are kept.
            max_split_location_distance: maximum distance allowed between parsed locations
                in shapefile and in split CSV to allow for subset matching (in degrees).
            min_valid_pixel_ratio_in_orders: the minimum ratio of valid pixels that needs to be
                found in order data so that the order is kept. If `None`, all orders are kept.
            min_valid_pixel_ratio_in_masks: the minimum ratio of valid pixels needed in field masks
                so that the location is kept. If `None`, all locations are kept.
            min_time_delta_between_orders: the minimum time allowed between two orders
                matched to the same location. If `None`, all orders are kept.
            discard_subset_assignment_on_overlap: specifies whether to discard location subset
                assignments in cases where the distance between two or more locations across
                different subsets is less than the `max_split_location_distance` argument.
        """
        super().__init__()
        assert max_scatter_threshold is None or max_scatter_threshold > 0, "invalid threshold"
        self.max_scatter_threshold = max_scatter_threshold
        assert max_order_count is None or max_order_count > 0, "invalid count"
        self.max_order_count = max_order_count
        assert (
            min_valid_pixel_ratio_in_orders is None or 0 <= min_valid_pixel_ratio_in_orders <= 1
        ), "invalid minimum valid pixel ratio for order data"
        self.min_valid_pixel_ratio_in_orders = min_valid_pixel_ratio_in_orders
        assert (
            min_valid_pixel_ratio_in_masks is None or 0 <= min_valid_pixel_ratio_in_masks <= 1
        ), "invalid minimum valid pixel ratio for field masks"
        self.min_valid_pixel_ratio_in_masks = min_valid_pixel_ratio_in_masks
        assert min_time_delta_between_orders is None or isinstance(
            min_time_delta_between_orders, datetime.timedelta
        ), "invalid minimum time delta between orders"
        self.min_time_delta_between_orders = min_time_delta_between_orders
        assert max_split_location_distance > 0, "invalid maximum split location distance"
        self.max_split_location_distance = max_split_location_distance
        self.discard_subset_assignment_on_overlap = discard_subset_assignment_on_overlap
        self.data_root_path = pathlib.Path(dataset_root_path)
        assert self.data_root_path.exists(), f"invalid dataset path: {self.data_root_path}"
        self.location_info = self._parse_location_info()
        loc_with_subset_count = sum([loc.subset is not None for loc in self.location_info.values()])
        logger.info(f"valid locations: {len(self.location_info)} (with subset: {loc_with_subset_count})")
        self.order_info, self.raster_band_count = self._parse_order_info()
        logger.info(f"valid orders: {len(self.order_info)}")
        logger.info(f"dataset rasters have {self.raster_band_count} bands")

        # we match locations to their orders by looking for intersections across their geometries
        orders_with_matches, locations_with_matches = [], []
        self.output_samples = []
        for location_id, location in tqdm.tqdm(self.location_info.items(), desc="matching orders"):
            # get the id of all orders whose bounds contain the centroid of the location's polygons
            matched_order_ids = [
                order_id for order_id, order in self.order_info.items() if location.centroid.within(order.raster_bbox)
            ]
            # sort the matched orders according to their timestamp (oldest to newest)
            matched_order_ids = list(sorted(matched_order_ids, key=lambda oid_: self.order_info[oid_].order_timestamp))
            # next, filter out orders that might be too close together in time
            if self.min_time_delta_between_orders is not None and len(matched_order_ids) > 1:
                kept_order_ids, last_timestamp = [], self.order_info[matched_order_ids[0]].order_timestamp
                for oid in matched_order_ids[1:]:
                    if (self.order_info[oid].order_timestamp - last_timestamp) > self.min_time_delta_between_orders:
                        kept_order_ids.append(oid)
                    last_timestamp = self.order_info[oid].order_timestamp
                matched_order_ids = kept_order_ids
            # finally, if we have too many matched orders for this location, we will drop some of them
            if self.max_order_count is not None and len(matched_order_ids) > self.max_order_count:
                # we drop the orders with the lowest valid ratios first
                kept_order_ids = list(
                    sorted(
                        matched_order_ids,
                        key=lambda oid_: self.order_info[oid_].raster_data_valid_ratio,
                        reverse=True,
                    )
                )[: self.max_order_count]
                matched_order_ids = [oid for oid in matched_order_ids if oid in kept_order_ids]
            # if we have not found any orders for this location, it is skipped
            if not matched_order_ids:
                continue
            locations_with_matches.append(location_id)
            orders_with_matches.extend(matched_order_ids)
            output_orders = [self.order_info[oid] for oid in matched_order_ids]
            # orders should still be sorted here, oldest acquisition to newest acquisition
            last_timestamp = output_orders[0].order_timestamp
            for order in output_orders[1:]:
                assert last_timestamp <= order.order_timestamp
                last_timestamp = order.order_timestamp
            self.output_samples.append(
                self._OutputSampleData(
                    location=location,
                    orders=output_orders,
                )
            )

        useless_orders = [oid for oid in self.order_info if oid not in orders_with_matches]
        logger.info(f"found {len(useless_orders)} orders without location")
        useless_orders_str = "\n\t".join(useless_orders)
        logger.debug(f"orders without location:\n\t{useless_orders_str}")
        orderless_locations = [lid for lid in self.location_info if lid not in locations_with_matches]
        logger.info(f"found {len(orderless_locations)} locations without orders")
        orderless_locations_str = "\n\t".join(orderless_locations)
        logger.debug(f"locations without orders:\n\t{orderless_locations_str}")
        logger.info(f"final sample (location-orders pair) count: {len(self.output_samples)}")
        output_pairs_distrib = np.unique([len(s.orders) for s in self.output_samples], return_counts=True)
        output_pairs_distrib_str = "\n\t".join(
            [
                f"{loc_count} location(s) with {order_count} order(s)"
                for order_count, loc_count in zip(*output_pairs_distrib)
            ]
        )
        logger.info(f"distribution of orders per location:\n\t{output_pairs_distrib_str}")
        assert len(self.output_samples) > 0, "no valid samples found?"

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Fetches and returns a data sample for a particular location of this dataset."""
        assert 0 <= item < len(self), f"invalid data sample index being queried: {item}"
        location_data = self.output_samples[item].location
        orders_info = self.output_samples[item].orders
        location_id = location_data.identifier
        location_subset_label = location_data.subset if location_data.subset is not None else "none"
        field_geoms = [list(p.geometry.exterior.coords) for p in location_data.polygons]
        field_centroid = np.asarray((location_data.centroid.x, location_data.centroid.y))  # (lon, lat)
        field_scatter = location_data.scatter_ratio
        image_count = len(orders_info)
        image_order_ids = [o.identifier for o in orders_info]
        image_metadata = [copy.deepcopy(o.order_metadata) for o in orders_info]
        target_crs = rasterio.crs.CRS.from_string(self.metadata.crs)  # noqa

        # we will keep the shape/transform of the LARGEST raster across all matched orders
        max_image_shape_order = max(orders_info, key=lambda o: np.prod(o.raster_orig_data_shape))
        output_transform, output_width, output_height = rasterio.warp.calculate_default_transform(
            src_crs=max_image_shape_order.raster_orig_crs,
            dst_crs=target_crs,
            # height, width = shape
            width=max_image_shape_order.raster_orig_data_shape[1],
            height=max_image_shape_order.raster_orig_data_shape[0],
            # left, bottom, right, top = bounds
            left=max_image_shape_order.raster_orig_bounds[0],
            bottom=max_image_shape_order.raster_orig_bounds[1],
            right=max_image_shape_order.raster_orig_bounds[2],
            top=max_image_shape_order.raster_orig_bounds[3],
        )
        output_image_shape = (output_height, output_width)
        assert output_height <= self.metadata.max_image_shape[0], f"invalid image height: {output_height}"
        assert output_width <= self.metadata.max_image_shape[1], f"invalid image width: {output_width}"

        # the output arrays can be preallocated based on the shape we determined above
        image_data = np.full(
            shape=(len(orders_info), self.raster_band_count, output_height, output_width),
            fill_value=self.metadata.nodata_val,
            dtype=np.float64,  # will be rounded and cast to np.uint16 later
        )
        image_roi = np.full(
            shape=(len(orders_info), output_height, output_width),
            fill_value=0,
            dtype=np.uint8,
        )
        image_udm2 = np.full(
            shape=(len(orders_info), 8, output_height, output_width),
            fill_value=0,
            dtype=np.uint8,
        )

        # next, we fill in the metadata dict and output arrays based on the data we read+reproject from rasters
        for order_idx, order_info in enumerate(orders_info):
            image_metadata[order_idx]["full_id"] = order_info.full_id
            image_metadata[order_idx]["planet_item_id"] = order_info.planet_item_id
            image_metadata[order_idx]["raster_centroid"] = shapely.geometry.mapping(order_info.raster_centroid)
            image_metadata[order_idx]["raster_bbox"] = shapely.geometry.mapping(order_info.raster_bbox)
            image_metadata[order_idx]["raster_xml_metadata"] = order_info.raster_xml_metadata
            with rasterio.open(order_info.raster_data_path) as raster:
                image_metadata[order_idx]["orig_crs"] = raster.crs.to_dict()
                image_metadata[order_idx]["orig_shape"] = raster.shape
                image_metadata[order_idx]["orig_bounds"] = tuple(raster.bounds)
                image_metadata[order_idx]["orig_transform"] = tuple(raster.transform)
                reprojector = functools.partial(
                    rasterio.warp.reproject,
                    src_transform=raster.transform,
                    src_crs=raster.crs,
                    dst_transform=output_transform,
                    dst_crs=target_crs,
                )
                raster_data = raster.read()
                with rasterio.open(order_info.raster_udm2_path) as udm2:
                    udm2_data = udm2.read()
                    usable_data_mask = np.logical_and(
                        np.any(raster_data != self.metadata.nodata_val, axis=0),
                        udm2_data[7] == 0,  # band #8 in udm2 = unusable pixel bits due to anomalies
                    )
            reprojector(
                source=usable_data_mask.astype(np.uint8),
                destination=image_roi[order_idx],
                resampling=rasterio.warp.Resampling.min,
            )
            reprojector(
                source=udm2_data,
                destination=image_udm2[order_idx],
                resampling=rasterio.warp.Resampling.nearest,
            )
            reprojector(
                source=raster_data,
                destination=image_data[order_idx],
                src_nodata=self.metadata.nodata_val,
                dst_nodata=self.metadata.nodata_val,
                resampling=rasterio.warp.Resampling.bilinear,
            )

        # finally, create the field mask using the same shape/transform as the arrays above
        field_mask_shapes = [(poly.geometry, 1) for poly in location_data.polygons]
        field_mask = rasterio.features.rasterize(
            shapes=field_mask_shapes,
            out_shape=output_image_shape,
            transform=output_transform,
        ).astype(bool)

        return dict(
            location_id=location_id,
            location_subset=location_subset_label,
            location_preview_image=self._generate_preview_image(image_data),
            location_preview_roi=image_roi[0].astype(bool),
            field_geoms=field_geoms,
            field_mask=field_mask,
            field_centroid=field_centroid,
            field_scatter=field_scatter,
            image_count=image_count,
            image_transform=np.asarray(output_transform),
            image_order_ids=image_order_ids,
            image_metadata=image_metadata,
            image_data=np.round(image_data).astype(np.uint16),
            image_roi=image_roi.astype(bool),
            image_udm2=image_udm2,
        )

    def _generate_preview_image(
        self,
        image_raw: np.ndarray,
        transpose_to_hwc: bool = True,
    ) -> np.ndarray:
        """Generates a preview RGB image of the given raster image stack."""
        # note: not using global mean/std stats for this! (computing them using the full stack)
        assert image_raw.ndim == 4
        order_count, ch_count, height, width = image_raw.shape
        # channel definitions info: https://developers.planet.com/docs/data/psscene/
        if ch_count == 3:
            # use as-is, it's already a red-green-blue data stack
            pass
        elif ch_count == 4:
            image_raw = image_raw[:, [2, 1, 0], :, :]  # drops the NIR channel
        elif ch_count == 8:
            image_raw = image_raw[:, [5, 3, 1], :, :]  # drops the all but the RGB channels
        valid_px_mask = np.any(image_raw != self.metadata.nodata_val, axis=1)
        normalized_data = np.zeros_like(image_raw, dtype=np.float64)
        # normalize in a bandwise fashion while ignoring nodata values
        for bidx in range(3):
            band_mean = image_raw[:, bidx][valid_px_mask].mean()
            band_std = max(image_raw[:, bidx][valid_px_mask].std(), 1)
            normalized_data[:, bidx][valid_px_mask] = (image_raw[:, bidx][valid_px_mask] - band_mean) / band_std
        # clamp to two std (95% of all data)
        normalized_data[normalized_data > 2] = 2
        normalized_data[normalized_data < -2] = -2
        # keep only the first image of the stack, and scale to 8-bit range
        preview_image = (((normalized_data[0] + 2) / 4) * 255).astype(np.uint8)
        if transpose_to_hwc:
            preview_image = np.transpose(preview_image, (1, 2, 0))  # move ch dim to last position
        return preview_image

    @staticmethod
    def _convert_location_id(sample_idx: int) -> str:
        """Converts a 'sample index' from Sherrie Wang's shapefile dataset to a unique str id."""
        # note: this is used to simplify debugging and help comprehension for dataset users
        return f"shp_location_{sample_idx:04}"

    @dataclasses.dataclass
    class _OutputSampleData:
        """Output sample data structure, used internally do tied locations to orders."""

        location: LocationData
        orders: typing.List[OrderInfo]


def _repackage_disa(dataset_root_path: pathlib.Path):
    assert dataset_root_path.is_dir(), f"missing dataset root directory: {dataset_root_path}"
    repackager = DeepLakeRepackager(dataset_root_path)
    output_path = dataset_root_path / ".deeplake"
    repackager.export(output_path, overwrite=True, num_workers=0)
    assert pathlib.Path(output_path).exists()


if __name__ == "__main__":
    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    config_ = ssl4rs.utils.config.init_hydra_and_compose_config()
    _repackage_disa(pathlib.Path(config_.utils.data_root_dir) / "ai4h-disa" / "india")
