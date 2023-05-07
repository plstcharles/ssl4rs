import pytest

import ssl4rs.utils.config


@pytest.fixture
def package_root_dir():
    """Returns the path to this package's root directory (i.e. where its modules are located)."""
    return ssl4rs.utils.config.get_package_root_dir()


@pytest.fixture
def framework_root_dir():
    """Returns the path to this framework's root directory (i.e. where the source code is
    located)."""
    return ssl4rs.utils.config.get_framework_root_dir()


@pytest.fixture
def data_root_dir():
    """Returns the data root directory for the current environment/config setup."""
    return ssl4rs.utils.config.get_data_root_dir()


@pytest.fixture
def global_cfg_cleaner():
    """Will automatically clear the global config dict in case one in saved in the parent test."""
    yield
    ssl4rs.utils.config.clear_global_config()
