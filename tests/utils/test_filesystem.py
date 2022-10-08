import os
import pathlib
import typing

import ssl4rs.utils.filesystem as fs_utils


def _write_dummy_file(file_path: typing.Union[typing.AnyStr, pathlib.Path]) -> str:
    with open(str(file_path), "wb") as fd:
        fd.write(b"some static stuff")
    hash = fs_utils.get_file_hash(file_path)
    return hash


def test_get_file_hash(tmpdir):
    file_path = os.path.join(tmpdir, "file.bin")
    hash = _write_dummy_file(file_path)
    assert os.path.isfile(file_path)
    assert hash == "8c92ff476d38070f08169c80cbd00d4a"


def test_get_package_root():
    pkg_root_dir = fs_utils.get_package_root_dir()
    assert pkg_root_dir.is_dir()
    expected_init_module_path = pkg_root_dir / "__init__.py"
    assert expected_init_module_path.is_file()
    expected_version_module_path = pkg_root_dir / "_version.py"
    assert expected_version_module_path.is_file()


def test_get_framework_root_dir():
    # note: if we're running tests, it means the 'framework directory' exists!
    fw_root_dir = fs_utils.get_framework_root_dir()
    assert fw_root_dir.is_dir()
    expected_test_dir_path = fw_root_dir / "tests"
    assert expected_test_dir_path.is_dir()


def test_rsync_folder(tmpdir):
    source_dir_path = pathlib.Path(tmpdir) / "source"
    source_dir_path.mkdir()
    source_file_path = source_dir_path / "file.bin"
    source_file_hash = _write_dummy_file(source_file_path)
    assert source_file_path.is_file()
    dest_dir_path = pathlib.Path(tmpdir) / "destination"
    assert not dest_dir_path.exists()
    fs_utils.rsync_folder(source_dir_path, dest_dir_path)
    expected_dir_copy_path = dest_dir_path / "source"
    assert expected_dir_copy_path.is_dir()
    expected_file_copy_path = expected_dir_copy_path / "file.bin"
    assert expected_file_copy_path.is_file()
    dest_file_hash = fs_utils.get_file_hash(expected_file_copy_path)
    assert source_file_hash == dest_file_hash


def test_work_dir_context_manager(tmpdir):
    orig_cwdir = os.path.abspath(os.path.curdir)
    assert pathlib.Path(orig_cwdir) != pathlib.Path(tmpdir)
    with fs_utils.WorkDirectoryContextManager(tmpdir):
        new_cwdir = os.path.abspath(os.path.curdir)
        assert pathlib.Path(new_cwdir) == pathlib.Path(tmpdir)
        rel_file_path = pathlib.Path("test.txt")
        assert not rel_file_path.exists()
        rel_file_path.touch()
        assert rel_file_path.exists()
        abs_file_path = pathlib.Path(tmpdir) / "test.txt"
        assert abs_file_path.is_file()
    new_cwdir = os.path.abspath(os.path.curdir)
    assert pathlib.Path(new_cwdir) == pathlib.Path(orig_cwdir)
    assert abs_file_path.is_file()

