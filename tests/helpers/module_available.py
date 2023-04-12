"""Module/library/environment/tool availability evaluator.

Adapted from:
https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/utilities/imports.py
"""

import os
import pathlib
import platform
from importlib.util import find_spec


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except ModuleNotFoundError:
        # Python 3.7+
        return False


_IS_WINDOWS = platform.system() == "Windows"
_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _module_available("deepspeed")
_RPC_AVAILABLE = not _IS_WINDOWS and _module_available("torch.distributed.rpc")
_IS_SLURM_AVAILABLE = pathlib.Path("/opt/slurm/bin/sinfo").is_file() and "SLURM_NTASKS" not in os.environ
_IS_LMOD_AVAILABLE = os.getenv("LMOD_ROOT", default=None)
_IS_ON_MILA_CLUSTER = _IS_SLURM_AVAILABLE and _IS_LMOD_AVAILABLE and "mila.quebec" in os.getenv("LMOD_ROOT")
