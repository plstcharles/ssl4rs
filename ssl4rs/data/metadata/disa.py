"""Defines static metadata for the DISA dataset.

TODO: validate this once again once finalized!
"""

import numpy as np

band_count = 4
"""Number of bands in all rasters."""

raster_dtype = np.dtype(np.uint16)
"""Data type for the raster bands."""

nodata_val = 0.0
"""No-data (invalid) pixel value in all raster bands."""

crs = "EPSG:4326"
"""EPSG code for the CRS of all rasters."""
