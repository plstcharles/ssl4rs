"""Contains utilities related to filesystem ops (file download, extraction, sync, ...)."""
import hashlib
import io
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import time
import typing
import unicodedata

_hook_start_time = time.time()


def download_file(
    url: typing.AnyStr,
    root: typing.Union[typing.AnyStr, pathlib.Path],
    filename: typing.AnyStr,
    md5: typing.Optional[typing.AnyStr] = None,
) -> pathlib.Path:
    """Downloads a file from a given URL to a local destination.

    Args:
        url: path to query for the file (query will be based on urllib).
        root: destination folder where the file should be saved.
        filename: destination name for the file.
        md5: optional, for md5 integrity check.

    Returns:
        The path to the downloaded file.
    """
    # inspired from torchvision.datasets.utils.download_url; no dep check
    from six.moves import urllib
    root = pathlib.Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    fpath = root / filename
    if not fpath.is_file():  # only download if we dont have it already
        urllib.request.urlretrieve(str(url), str(fpath), reporthook)
        sys.stdout.write("\r")
        sys.stdout.flush()
    if md5 is not None:  # once we have the file, checksum it (if possible)
        md5o = hashlib.md5()
        with open(str(fpath), "rb") as fd:
            for chunk in iter(lambda: fd.read(1024 * 1024), b''):
                md5o.update(chunk)
        md5c = md5o.hexdigest()
        assert md5c == md5, f"md5 check failed for downloaded file: {fpath}"
    return fpath


def extract_tar(
    filepath: typing.Union[typing.AnyStr, pathlib.Path],
    root: typing.Union[typing.AnyStr, pathlib.Path],
    flags: typing.AnyStr = "r:gz",
) -> None:
    """Extracts the content of a tar file to a specific location.

    Args:
        filepath: location of the tar archive.
        root: where to extract the archive's content.
        flags: extra flags passed to ``tarfile.open``.
    """
    class _FileWrapper(io.FileIO):
        def __init__(self, path, *args, **kwargs):
            self.start_time = time.time()
            self._size = os.path.getsize(path)
            super().__init__(path, *args, **kwargs)

        def read(self, *args, **kwargs):
            duration = time.time() - self.start_time
            progress_size = self.tell()
            speed = str(int(progress_size / (1024 * duration))) if duration > 0 else "?"
            percent = min(int(progress_size * 100 / self._size), 100)
            progress_size_mb = progress_size / (1024 * 1024)
            sys.stdout.write(
                f"\r\t=> extracted {percent}%% ({progress_size_mb} MB) @ {speed} KB/s..."
            )
            sys.stdout.flush()
            return io.FileIO.read(self, *args, **kwargs)

    cwd = os.getcwd()
    tar = tarfile.open(fileobj=_FileWrapper(str(filepath)), mode=str(flags))
    root = pathlib.Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(str(root))
    tar.extractall()
    tar.close()
    os.chdir(cwd)
    sys.stdout.write("\r")
    sys.stdout.flush()


def reporthook(
    count: int,
    block_size: int,
    total_size: int,
) -> None:
    """Report hook used to display a download progression bar when using urllib requests."""
    global _hook_start_time
    if count == 0:
        _hook_start_time = time.time()
        return
    duration = time.time() - _hook_start_time
    progress_size = int(count * block_size)
    speed = str(int(progress_size / (1024 * duration))) if duration > 0 else "?"
    percent = min(int(count * block_size * 100 / total_size), 100)
    progress_size_mb = progress_size / (1024 * 1024)
    sys.stdout.write(
        f"\r\t=> downloaded {percent}%% ({progress_size_mb} MB) @ {speed} KB/s..."
    )
    sys.stdout.flush()


def read_in_chunks(
    file_object: typing.Any,
    chunk_size: int = 1024
) -> typing.Any:
    """Read a file object in chunks of size chunk_size.

    Args:
        file_object: an open file object/handle.
        chunk_size: size of the chunk to read.

    Yields:
        The read data.
    """
    assert chunk_size > 0, "invalid chunk size!"
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def get_file_hash(
    file_path: typing.Union[typing.AnyStr, pathlib.Path],
) -> str:
    """Computes and returns the hash (md5 checksum) of a file as a string of hex digits.

    Args:
        file_path: location of the file to hash.

    Returns:
        The hashing result as a string of hexadecimal digits.
    """
    with open(str(file_path), "rb") as f:
        file_hash = hashlib.md5()
        for chunk in read_in_chunks(f):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def get_slurm_tmpdir() -> typing.Optional[pathlib.Path]:
    """Returns the local SLURM_TMPDIR path if available, or ``None`` otherwise."""
    slurm_tmpdir = os.getenv("SLURM_TMPDIR")
    if slurm_tmpdir is not None:
        slurm_tmpdir = pathlib.Path(slurm_tmpdir)
        assert slurm_tmpdir.is_dir(), f"invalid SLURM_TMPDIR path: {slurm_tmpdir}"
        assert os.access(str(slurm_tmpdir), os.W_OK), f"SLURM_TMPDIR not writable: {slurm_tmpdir}"
    return slurm_tmpdir


def rsync_folder(
    source: typing.Union[typing.AnyStr, pathlib.Path],
    target: typing.Union[typing.AnyStr, pathlib.Path],
) -> None:
    """Uses rsync to copy the content of source into target."""
    target = pathlib.Path(target).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["rsync", "-avzq", str(source), str(target)])


def slugify(
    in_str: typing.AnyStr,
    allow_unicode: bool = False,
) -> str:
    """Converts a provided string into a file-name-compatible string.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py

    Will convert the input string to ASCII if 'allow_unicode' is False. Will convert spaces or
    repeated dashes to single dashes. Will remove characters that aren't alphanumerics, underscores,
    or hyphens. Will convert to lowercase. Will also strip leading and trailing whitespace, dashes,
    and underscores.
    """
    value = str(in_str)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")
