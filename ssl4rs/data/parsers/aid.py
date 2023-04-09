"""Implements a data parser for the AID dataset.

See the following URL for more info on this dataset: https://captain-whu.github.io/AID/
"""

import ssl4rs.data.metadata.aid
import ssl4rs.data.parsers.utils


class DeepLakeParser(ssl4rs.data.parsers.utils.DeepLakeParser):
    """The AID dataset does not require any special handling on top of the base deeplake parser."""

    metadata = ssl4rs.data.metadata.aid
