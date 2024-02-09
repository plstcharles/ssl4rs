"""Defines static metadata for the DISA dataset.

For information on the Planet PSScene-specific attributes below, see:
    https://developers.planet.com/docs/data/psscene/

TODO: validate this once again once finalized!
"""

import numpy as np

raster_dtype = np.dtype(np.uint16)
"""Data type for the raster bands."""

nodata_val = 0.0
"""No-data (invalid) pixel value in all raster bands."""

three_band_descriptions = ["red", "green", "blue"]
"""When using 3-band Planet PSScene rasters, this is the description of those bands."""

four_band_descriptions = ["blue", "green", "red", "nir"]
"""When using 4-band Planet PSScene rasters, this is the description of those bands."""

eight_band_descriptions = [
    "coastal blue",
    "blue",
    "green i",
    "green",
    "yellow",
    "red",
    "red edge",
    "nir",
]
"""When using 8-band Planet PSScene rasters, this is the description of those bands."""

crs = "EPSG:4326"
"""EPSG code for the CRS of all rasters (post-repackaging)."""

_crs_with_meter_units = "EPSG:24378"
"""EPSG code used internally to compute areas/distances for this dataset."""

shapefile_md5sum = "7e23ab737c08104927353c542a0bf4ce"
"""MD5 checksum for Sherrie Wang's field geometries shapefile.

Visit https://zenodo.org/records/7315090 for more information.
"""

shapefile_location_count = 2002
"""Number of locations with annotated polygons (pre-cleanup).

Note: 1999 locations have 5 polygons, and 3 locations have 6 polygons.
"""

shapefile_polygon_count = 10013
"""Number of annotated polygons (pre-cleanup).

Note: 1999 locations have 5 polygons, and 3 locations have 6 polygons.
"""

psscene_file_names = [
    "3B_AnalyticMS_clip.tif",
    "3B_AnalyticMS_metadata_clip.xml",
    "3B_udm2_clip.tif",
    "metadata.json",
]
"""List of file suffixes that are expected in the raw data folders for Planet PSScene orders."""
