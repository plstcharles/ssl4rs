"""Implements a data parser for the DISA dataset."""

import ssl4rs.data.metadata.disa
import ssl4rs.data.parsers.utils


class DeepLakeParser(ssl4rs.data.parsers.utils.DeepLakeParser):
    """The DISA dataset does not require any special handling on top of the base deeplake parser."""

    metadata = ssl4rs.data.metadata.disa
