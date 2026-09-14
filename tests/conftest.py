import os
import tempfile

import pytest

from caught.config import Config
from caught.store import Store


@pytest.fixture
def cfg(tmp_path):
    c = Config(data_dir=str(tmp_path / "data"))
    c.ensure_dirs()
    return c


@pytest.fixture
def store(cfg):
    s = Store(cfg.db_path, cfg.samples_path)
    yield s
    s.close()
