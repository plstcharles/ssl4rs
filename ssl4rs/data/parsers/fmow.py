"""Implements a deep lake data parser for the Functional Map of the World (FMoW) dataset.

See the following URLs for more info on this dataset:
https://arxiv.org/abs/1711.07846
https://github.com/fMoW/dataset
https://spacenet.ai/iarpa-functional-map-of-the-world-fmow/
"""

import ssl4rs.data.parsers
import ssl4rs.data.repackagers.fmow


class DeepLakeParser(ssl4rs.data.parsers.DeepLakeParser):
    """FMoW does not require any special handling on top of the base deeplake parser.

    This means that apart from the utility attributes/defines, this class is empty.
    """

    class_names = ssl4rs.data.repackagers.fmow.DeepLakeRepackager.class_names
