import pathlib
import typing

import numpy as np
import rasterio
import rasterio.crs
import rasterio.io
import rasterio.mask
import rasterio.warp
import shapely.geometry

import ssl4rs.utils.logging

logger = ssl4rs.utils.logging.get_logger(__name__)


def compute_raster_extent_from_affine_transform(
    affine_transform: rasterio.Affine,
    raster_shape: typing.Tuple[int, int],
) -> typing.Tuple[float, float, float, float]:
    """Computes the geographical extent (bounds) of a raster based on its affine transform + shape.

    Args:
        affine_transform: the affine transformation (georeferencing matrix) of the raster.
        raster_shape: the shape of the raster (number of rows, number of columns).

    Returns:
        A tuple of four floats that correspond to the geographical extent (bounds) of the raster
        in the form (left, bottom, right, top).
    """
    n_rows, n_cols = raster_shape
    left = affine_transform.c
    bottom = affine_transform.f + n_rows * affine_transform.e
    right = affine_transform.c + n_cols * affine_transform.a
    top = affine_transform.f
    return left, bottom, right, top


class GeoRasterCropper:
    """Provides crops from a georeferenced raster image.

    The raster will be opened and kept in memory using rasterio. The crop location must
    be specified as a geometry whose bounds can be buffered to create an output array. If
    the specified geometry is not found inside the raster, an empty crop will be returned.

    Note: if the raster is warped to a new CRS, the result will be entirely held in memory.
    This means that if large datasets all have their rasters warped using this class, it might
    require a lot of RAM, as the rasters are kept opened in memory until the cropping is done.
    """

    def __init__(
        self,
        raster_path_or_object: typing.Union[typing.AnyStr, pathlib.Path, rasterio.DatasetReader],
        new_crs: typing.Optional[typing.AnyStr] = None,
    ):
        """Validates the input raster and warps it to a new CRS if needed."""
        if isinstance(raster_path_or_object, (pathlib.Path, str)):
            raster_path = pathlib.Path(raster_path_or_object)
            assert raster_path.exists(), f"invalid path: {raster_path}"
            self.raster = rasterio.open(raster_path)
            self._must_close = True
        else:
            assert isinstance(
                raster_path_or_object, rasterio.DatasetReader
            ), f"invalid raster obj: {type(raster_path_or_object)}"
            self.raster = raster_path_or_object
            self._must_close = False
        self._raster_memfile = None
        if new_crs is not None and new_crs != self.crs:
            raster_left, raster_bottom, raster_right, raster_top = self.bounds
            dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                src_crs=self.crs,
                dst_crs=new_crs,
                width=self.width,
                height=self.height,
                left=raster_left,
                bottom=raster_bottom,
                right=raster_right,
                top=raster_top,
            )
            self._raster_memfile = rasterio.io.MemoryFile()
            dst_raster_kwargs = self.metadata.copy()
            dst_raster_kwargs.update(
                {
                    "crs": new_crs,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                }
            )
            with self._raster_memfile.open(**dst_raster_kwargs) as dst:
                for i in range(1, self.band_count + 1):
                    rasterio.warp.reproject(
                        source=rasterio.band(self.raster, i),
                        destination=rasterio.band(dst, i),
                        src_transform=self.transform,
                        src_crs=self.crs,
                        dst_transform=dst_transform,
                        dst_crs=new_crs,
                        resampling=rasterio.warp.Resampling.nearest,
                    )
            self.raster = self._raster_memfile.open()
            self._must_close = True

    def __del__(self):
        """Makes sure all raster fds are closed before the memfile is closed."""
        if self._must_close:
            self.raster.close()
            self.raster = None
        if self._raster_memfile is not None:
            self._raster_memfile.close()
            self._raster_memfile = None

    @property
    def bounds(self) -> typing.Tuple[float, float, float, float]:
        """Returns the bounds (left, bottom, right, top) of the wrapped raster object."""
        # (lower left x, lower left y, upper right x, upper right y)
        return self.raster.bounds

    @property
    def width(self) -> int:
        """Returns the width of the wrapped raster object."""
        return self.raster.width

    @property
    def height(self) -> int:
        """Returns the height of the wrapped raster object."""
        return self.raster.height

    @property
    def transform(self) -> rasterio.transform.Affine:
        """Returns the wrapped raster object's georeferencing transformation matrix.

        This transform maps pixel row/column coordinates to coordinates in the dataset's CRS.
        """
        return self.raster.transform

    @property
    def band_count(self) -> int:
        """Returns the number of raster bands in the wrapped raster object."""
        return self.raster.count

    @property
    def band_descriptions(self) -> typing.List[str]:
        """Returns the description of raster bands in the wrapped raster object."""
        return self.raster.descriptions

    @property
    def crs(self) -> rasterio.crs.CRS:
        """Returns the CRS (possibly updated since opening) of the wrapped raster dataset."""
        return self.raster.crs

    @property
    def metadata(self) -> typing.Dict[str, typing.Any]:
        """Returns the base metadata of the wrapped raster object."""
        return self.raster.meta

    @property
    def nodata_value(self) -> float:
        """Returns the 'nodata' (invalid) value used when padding/filling partial raster data."""
        return self.raster.nodata

    @property
    def pixel_resolution(self) -> typing.Tuple[float, float]:
        """Returns the (width, height) of pixels in the units of the raster CRS."""
        return self.raster.res

    def crop(
        self,
        geom: shapely.geometry.base.BaseGeometry,
        buffer_size: float = 0,
        all_touched: bool = False,
        invert: bool = False,
        nodata: typing.Optional[typing.Union[int, float]] = None,
        filled: bool = True,
        pad: bool = False,
        pad_width: float = 0.5,
        indexes: typing.Optional[typing.Union[typing.List[int], int]] = None,
        return_crop_transform: bool = False,
    ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, typing.Tuple]]:
        """Generated an axis-aligned crop of the raster centered on the given geometry.

        We assume that the geometry is given in the correct CRS, that its bounds are valid,
        and that it is OK to use padding if the geometry goes outside the raster bounds.

        Args:
            geom: geometry dictating the bounds of the crop to generate. Only the bounds
                of this geometry will be used to generate the crop.
            buffer_size: buffer size to expand the geometry bounds before apply the crop.
            all_touched: include a pixel in the mask if it touches any of the shapes.
                If False (default), include a pixel only if its center is within one of the shapes,
                or if it is selected by Bresenham's line algorithm.
            invert: if False (default) pixels outside shapes will be masked. If True, pixels inside
                shape will be masked.
            nodata:  value representing nodata within each raster band. If not set, defaults to the
                nodata value for the input raster. If there is no set nodata value for the raster,
                it defaults to 0.
            filled: if True, the pixels outside the features will be set to nodata. If False, the
                output array will contain the original pixel data, and only the mask will be based
                on shapes. Defaults to True.
            pad: if True, the features will be padded in each direction by one half of a pixel
                prior to cropping raster. Defaults to False.
            pad_width: see above. Undocumented in rasterio.
            indexes: if `indexes` is a list, the result is a 3D array, but is a 2D array if it
                is a band index number.
            return_crop_transform: defines whether the affine transform of the generated crop should
                be returned as a 2nd output alongside the crop or not.
        """
        assert buffer_size >= 0, f"invalid buffer size: {buffer_size}"
        minx, miny, maxx, maxy = geom.bounds
        bbox = shapely.geometry.box(
            minx=minx - buffer_size,
            miny=miny - buffer_size,
            maxx=maxx + buffer_size,
            maxy=maxy + buffer_size,
        )
        out_image, out_transform = rasterio.mask.mask(
            dataset=self.raster,
            shapes=[bbox],
            all_touched=all_touched,
            invert=invert,
            nodata=nodata,
            filled=filled,
            crop=True,
            pad=pad,
            pad_width=pad_width,
            indexes=indexes,
        )
        if return_crop_transform:
            return out_image, out_transform
        else:
            return out_image


if __name__ == "__main__":
    import logging

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import rasterio.plot

    import ssl4rs.data.parsers.utils.geopandas_utils as gpd_utils
    import ssl4rs.utils.config
    import ssl4rs.utils.logging

    np.random.seed(0)

    ssl4rs.utils.logging.setup_logging_for_analysis_script(logging.INFO)
    data_root = ssl4rs.utils.config.get_data_root_dir()
    demo_data_root = data_root / "demo" / "OpenAI-Tanzania-BFSC"
    geometries_path = demo_data_root / "grid_001.geojson"
    raster_path = demo_data_root / "5afeda152b6a08001185f11b.tif"

    geom_parser = gpd_utils.GeoPandasParser(
        dataset_path_or_object=geometries_path,
        convert_tensors_to_base_type=False,
    )
    logger.info(f"Loaded {len(geom_parser)} geometries")

    raster_cropper = GeoRasterCropper(
        raster_path_or_object=raster_path,
        new_crs=geom_parser.crs,
    )
    rwidth, rheight, bcount = raster_cropper.width, raster_cropper.height, raster_cropper.band_count
    logger.info(f"Loaded {rwidth}x{rheight} raster with {bcount} bands")

    geom_idxs = np.random.permutation(len(geom_parser))
    for geom_idx in geom_idxs:
        geometry = geom_parser[geom_idx]["geometry"]
        crop, crop_transform = raster_cropper.crop(
            geometry,
            return_crop_transform=True,
        )
        logger.info(f"Created crop with shape: {crop.shape}")
        crop_extent = compute_raster_extent_from_affine_transform(crop_transform, crop.shape[1:])
        cminx, cminy, cmaxx, cmaxy = crop_extent
        cbands, cheight, cwidth = crop.shape
        fig, ax = plt.subplots(figsize=plt.figaspect(cheight / cwidth), dpi=250)
        ax = rasterio.plot.show(
            crop[0],
            cmap="gray",
            extent=[cminx, cmaxx, cminy, cmaxy],
            ax=ax,
        )
        geoms = gpd.GeoSeries([geometry], crs=geom_parser.crs)
        geoms.plot(ax=ax, edgecolor="red", facecolor="none")
        fig.show()
