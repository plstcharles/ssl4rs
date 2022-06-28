"""Implements a data parser for the AID dataset.

See the following URL for more info on this dataset:
    https://captain-whu.github.io/AID/
"""

import ssl4rs.data.parsers
import ssl4rs.data.repackagers.aid

# TODO: add a regular parser that does not rely on Hub?


class HubParser(ssl4rs.data.parsers.HubParser):
    """The AID dataset does not require any special handling on top of the base Hub parser.

    This means that apart from the utility attributes/defines, this class is empty.
    """

    class_distrib = ssl4rs.data.repackagers.aid.AIDRepackager.class_distrib
    class_names = ssl4rs.data.repackagers.aid.AIDRepackager.class_names
    image_shape = ssl4rs.data.repackagers.aid.AIDRepackager.image_shape
