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

three_band_descriptions = ("red", "green", "blue")
"""When using 3-band Planet PSScene rasters, this is the description of those bands."""

four_band_descriptions = ("blue", "green", "red", "nir")
"""When using 4-band Planet PSScene rasters, this is the description of those bands."""

eight_band_descriptions = (
    "coastal blue",
    "blue",
    "green i",
    "green",
    "yellow",
    "red",
    "red edge",
    "nir",
)
"""When using 8-band Planet PSScene rasters, this is the description of those bands."""

band_count_to_descriptions = {
    3: three_band_descriptions,
    4: four_band_descriptions,
    8: eight_band_descriptions,
}
"""Band count (int) to band descriptions map."""

supported_band_counts = tuple(band_count_to_descriptions.keys())
"""Supported band counts for different versions of the dataset."""

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

psscene_file_names = (
    "3B_AnalyticMS_clip.tif",
    "3B_AnalyticMS_metadata_clip.xml",
    "3B_udm2_clip.tif",
    "metadata.json",
)
"""List of file suffixes that are expected in the raw data folders for Planet PSScene orders."""

tensor_info_dicts = dict(
    location_id=dict(htype="text", sample_compression=None),
    location_preview_image=dict(htype="image.rgb", dtype=np.uint8, sample_compression="jpg"),
    location_preview_roi=dict(htype="binary_mask", dtype=bool, sample_compression="lz4"),
    field_geoms=dict(htype="polygon", dtype=np.float64, sample_compression=None),
    field_mask=dict(htype="binary_mask", dtype=bool, sample_compression="lz4"),
    field_centroid=dict(htype="generic", dtype=np.float64, sample_compression=None),
    field_scatter=dict(htype="generic", dtype=np.float64, sample_compression=None),
    image_count=dict(htype="generic", dtype=int, sample_compression=None),
    image_transform=dict(htype="generic", dtype=np.float64, sample_compression=None),
    image_order_ids=dict(htype="sequence[text]", sample_compression=None),
    image_metadata=dict(htype="sequence[json]", sample_compression=None),
    image_data=dict(htype="generic", dtype=np.uint16, sample_compression=None),
    image_roi=dict(htype="binary_mask", dtype=bool, sample_compression="lz4"),
    image_udm2=dict(htype="generic", dtype=np.uint8, sample_compression="lz4"),
)
"""Returns the dictionary of tensor declarations used during repackaging.

Note: the arguments used for each key in the returned dictionary should be usable directly
in the `deeplake.Dataset.create_tensor` function; see
https://api-docs.activeloop.ai/#deeplake.Dataset.create_tensor for more information.

Tensor descriptions:
    location_id: unique identifier (string) used to tag each target location, derived from
        Sherrie Wang's original shapefile sample index.
    location_preview_image: RGB image used to give a quick preview of each location, derived
        from the raster of the first order found for each location. Each image is stored as
        a (height, width, channels) array, where channels are in RGB order.
    location_preview_roi: binary mask of valid pixels in the above preview image. Each mask
        is stored as a (height, width) array of boolean values.
    field_geoms: Sherrie Wang's field geometries (polygons) based on October 2020 SPOT
        imagery. Each sample is stored as (#polygons, #points, #dimensions), where all
        points are 2D (so #dimensions=2).
    field_mask: binary mask encoding whether each pixel in all the registered images lies
        in one of the above field polygons. The dimensions of this mask should be the same
        as those of the image tensors for a given sample, i.e. (height, width).
    field_centroid: coordinates of the centroid of all field polygons computed via shapely
        and weighted according to each field's area, in the target CRS. This is given as
        (longitude, latitude), or (x, y)-coordinates.
    field_scatter: the 'scatter ratio' value defining how far away field polygons are from
        each other with respect to their maximum diameter. Lower is a tighter group, larger
        is more spread out. May have been used to filter out some very-spread-out locations.
    image_count: total number of images available based on orders matched for each location.
    image_transform: the transform used to map each pixel in the image data to a geospatial
        location (derived from rasterio's Affine transform).
    image_order_ids: list of unique order identifiers (strings) matched for each location
        which resulted in the raster data available for each sample.
    image_metadata: list of jsons (dicts) that contain the metadata associated with each
        order and its raster/mask (note: includes acquisition timestamps).
    image_data: stack of multi-band raster data for this location. Given as an array of
        dimensions (#orders, #bands, height, width) with uint16 values, where all rasters
        have already been registered to the same extent+CRS.
    image_roi: stack of multi-band binary masks of valid pixels in the above image stack,
        with dimensions (#orders, height, width). Similar to the udm2 array below, but
        where all bands have been aggregated into a single band (or channel) indicating
        whether any of the original image bands is usable or not, for any reason.
    image_udm2: stack of usable data masks (UDM2) associated with each image in the above
        image stack, with dimensions (#orders, 8, height, width). Note that this data is
        reprojected (with nearest resampling) from its original CRS. The UDM2 format is
        described here: https://developers.planet.com/docs/data/udm-2/
"""

tensor_names = tuple(tensor_info_dicts.keys())
"""List of tensors repackaged by the deeplake dataset repackager and expected by the parser."""

tensor_names_to_collate_manually = (
    "location_id",  # string/identifier
    "field_geoms",  # since both polygon counts and point counts can vary across locations
    "image_order_ids",  # list of strings/identifiers
    "image_metadata",  # list of dicts with lots of more-or-less useful stuff in them
    "image_data",  # array whose first dimension varies (it's the number of matched orders)
    "image_roi",  # array whose first dimension varies (it's the number of matched orders)
    "image_udm2",  # array whose first dimension varies (it's the number of matched orders)
)
"""List of tensors that will surely break pytorch if not handled manually during batch collate."""

tensor_names_to_pad = (
    "location_preview_image",
    "location_preview_roi",
    "field_mask",
    "field_boundary_mask",
    "image_data",
    "image_roi",
    "image_udm2",
)
"""List of tensors that will surely need to be padded during batch collate."""

tensor_names_to_convert = (
    "location_preview_image",
    "location_preview_roi",
    "field_mask",
    "image_data",
    "image_roi",
    "image_udm2",
)
"""List of tensors that will need to be converted from numpy arrays to torch.Tensor."""

tensor_names_to_transpose_ch = (
    "location_preview_image",
    # note: image_data does not need to be transposed (it's already in NxCxHxW format)
)
"""List of tensors that will need to have their channels transposed during batch collate."""


tensor_normalization_stats = {
    "location_preview_image": dict(  # (red, green, blue) stats from imagenet
        mean=np.asarray((0.485, 0.456, 0.406), dtype=np.float32) * 255,
        std=np.asarray((0.229, 0.224, 0.225), dtype=np.float32) * 255,
    ),
    "image_data": {  # band_count to stats mapping
        4: dict(
            mean=np.asarray(
                (  # (blue, green, red, nir) stats from `disa.ipynb` notebook (2024-02-16)
                    8209.919921875,
                    7170.8564453125,
                    5660.4619140625,
                    7837.09423828125,
                ),
                dtype=np.float32,
            ),
            std=np.asarray(
                (
                    566.794677734375,
                    614.297119140625,
                    754.4950561523438,
                    1062.7650146484375,
                ),
                dtype=np.float32,
            ),
        ),
    },
    # todo: compute more stats if we want to normalize other raw values, e.g. latlon coords
}

tensor_names_to_normalize = tuple(tensor_normalization_stats.keys())
"""List of tensors that should have their channels normalized during tensor conversion."""

dontcare_label = -1
"""Value that is used inside classification masks to indicate dontcare pixels (based on ROI)."""
