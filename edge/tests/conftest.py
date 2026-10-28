import sys
from pathlib import Path

import pytest

EDGE = Path(__file__).resolve().parents[1]
REPO = EDGE.parent
for path in (EDGE, REPO / "src", REPO / "tools", EDGE / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SAMPLE = REPO / "app" / "samples" / "vishal_normal_95spm.mp4"


@pytest.fixture
def settings_factory(tmp_path):
    """Settings for a device whose storage lives in a temp folder."""
    import yaml

    from gaitedge import settings as settings_module

    def make(**overrides):
        data = {"device": {"id": "test-device"}, "storage": {"root": str(tmp_path / "root")},
                "camera": {"backend": "file", "source": str(SAMPLE), "realtime": False,
                           "width": 640, "height": 360, "fps": 60},
                # Port 0: any free port, so tests never collide with a real service.
                "preview": {"port": 0}}
        for dotted, value in overrides.items():
            node = data
            parts = dotted.split("__")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        path = tmp_path / "edge.yaml"
        path.write_text(yaml.safe_dump(data))
        return settings_module.load(path)

    return make
