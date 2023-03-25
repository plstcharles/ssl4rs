"""Implements a data parser for the UC Merced Land Use dataset.

See the following URL(s) for more info on this dataset:
http://weegee.vision.ucmerced.edu/datasets/landuse.html
https://www.tensorflow.org/datasets/catalog/uc_merced
http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
"""

import ssl4rs.data.parsers
import ssl4rs.data.repackagers.ucmerced


class DeepLakeParser(ssl4rs.data.parsers.DeepLakeParser):
    """UCMerced does not require any special handling on top of the base deeplake parser.

    This means that apart from the utility attributes/defines, this class is empty.
    """

    class_distrib = ssl4rs.data.repackagers.ucmerced.DeepLakeRepackager.class_distrib
    class_names = ssl4rs.data.repackagers.ucmerced.DeepLakeRepackager.class_names
    image_shape = ssl4rs.data.repackagers.ucmerced.DeepLakeRepackager.image_shape
    ground_sampling_distance = ssl4rs.data.repackagers.ucmerced.DeepLakeRepackager.ground_sampling_distance
