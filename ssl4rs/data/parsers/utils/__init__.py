import torch.utils.data

import ssl4rs.data.parsers.utils.deeplake
import ssl4rs.data.parsers.utils.wrappers
from ssl4rs.data.parsers.utils.deeplake import DeepLakeParser
from ssl4rs.data.parsers.utils.wrappers import ParserWrapper

# todo: define a base interface w/ metadata, transform map, filter, select, cache, tensor names, ...?
DataParser = torch.utils.data.Dataset
