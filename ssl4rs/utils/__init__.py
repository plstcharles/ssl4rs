import ssl4rs.utils.compression
import ssl4rs.utils.config
import ssl4rs.utils.filesystem
import ssl4rs.utils.imgproc
import ssl4rs.utils.logging
import ssl4rs.utils.patch_coord

from ssl4rs.utils.patch_coord import PatchCoord
from ssl4rs.utils.logging import get_logger

getLogger = get_logger  # for convenience, to more easily replace classic logging calls
