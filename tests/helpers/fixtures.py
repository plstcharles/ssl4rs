import pytest

import ssl4rs.utils.config


@pytest.fixture
def global_cfg_cleaner():
    """Will automatically clear the global config dict in case one in saved in the parent test."""
    yield
    ssl4rs.utils.config.clear_global_config()
