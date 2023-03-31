import subprocess
import typing

import ssl4rs.utils.filesystem as fs_utils


def run(
    args: typing.List[str],
    check=False,
    capture_output=True,
    text=True,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Launches a subprocess from the framework root using the provided command arguments list."""
    assert len(args) > 0
    fw_root_dir = fs_utils.get_framework_root_dir()
    assert fw_root_dir.is_dir(), "could not locate framework root dir"
    with fs_utils.WorkDirectoryContextManager(fw_root_dir):
        result = subprocess.run(args, check=check, capture_output=capture_output, text=text, **kwargs)
    return result
