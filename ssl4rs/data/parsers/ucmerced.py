"""Implements a data parser for the UC Merced Land Use dataset.

See the following URL(s) for more info on this dataset:
http://weegee.vision.ucmerced.edu/datasets/landuse.html
http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
"""

import ssl4rs.data.metadata.ucmerced
import ssl4rs.data.parsers.utils


class DeepLakeParser(ssl4rs.data.parsers.utils.DeepLakeParser):
    """UCMerced does not require any special handling on top of the base deeplake parser."""

    metadata = ssl4rs.data.metadata.ucmerced
