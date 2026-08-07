import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gaitscreen.config import Config  # noqa: E402


@pytest.fixture()
def cfg() -> Config:
    return Config.load()
