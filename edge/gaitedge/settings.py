"""Device settings, read from one YAML file.

Kept separate from the gait-analysis config (``config/default.yaml`` in the
main package): that one describes *how a walk is measured* and must be the same
on every device and on the server, while this one describes *this device* --
its camera, where it sends things, how much disk it may use.

The one place the two meet is ``analysis_overrides``, merged over the analysis
config. It exists for the pose model choice, which is a speed/accuracy
trade-off that genuinely differs between a Pi and a server; anything else set
there makes this device measure differently from the rest, so ``describe()``
reports it in every result.
"""
from __future__ import annotations

import copy
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

#: Pose model variants MediaPipe publishes, smallest last. All three take the
#: same input and give the same 33 landmarks; they differ in network size.
MODEL_VARIANTS = ("heavy", "full", "lite")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")

DEFAULTS: dict[str, Any] = {
    "device": {
        "id": None,              # defaults to the hostname
        "site": "",              # free text: where this camera is installed
    },
    "camera": {
        # picamera2 for a Raspberry Pi camera on the ribbon connector, opencv
        # for a USB webcam, file to replay a recording (testing).
        "backend": "picamera2",
        "index": 0,              # opencv: which /dev/video device
        "source": None,          # file: path to replay
        "width": 1280,
        "height": 720,
        "fps": 60,
        "rotate": 0,             # 0/90/180/270, for a camera mounted on its side
        "realtime": True,        # file: replay at recorded speed
    },
    "trigger": {
        # motion: record whenever something walks through; manual: only on
        # request (`gaitedge record`); continuous: fixed-length clips forever.
        "mode": "motion",
        "analysis_width": 160,   # motion is judged on a thumbnail this wide
        "min_moving_frac": 0.02, # share of the thumbnail that must change
        "start_frames": 6,       # consecutive moving frames to start
        # Stillness before a clip is closed. Long enough to hold a recording
        # open through a turn at the end of the walkway, when the walker is
        # out of frame for several seconds: closing there split a four-pass
        # pilot walk into four clips of 2-4 strides each, too few for
        # step-to-step timing. The cost is that two people passing within
        # this time share a clip.
        "stop_after_s": 8.0,
        "pre_roll_s": 1.5,       # kept from before motion started
        "min_clip_s": 4.0,       # shorter clips are discarded (passers-by)
        "max_clip_s": 90.0,      # a clip is cut here regardless
        "warmup_s": 3.0,         # ignore motion while the camera settles
    },
    "record": {
        "codec": "h264",         # h264 (small files) or mjpeg (light on CPU)
        "crf": 23,
        "preset": "ultrafast",
    },
    "processing": {
        # local: analyse on this device and send results; remote: send the
        # recording for the server to analyse; auto: local if the last
        # benchmark says this device keeps up, otherwise remote.
        "mode": "auto",
        "model": "full",         # heavy | full | lite (see `gaitedge bench`)
        "make_overlay": True,    # draw the annotated video on the device
        "coverage_m": None,      # floor length the camera sees; enables speed
        "max_realtime_factor": 6.0,  # auto: local only if a walk takes <= this
                                     # many times its own length to analyse
    },
    "upload": {
        "server_url": None,      # e.g. https://gait.example.org
        "token": None,           # per-device secret, sent as a bearer token
        "timeout_s": 120,
        "retry_base_s": 15,      # backoff starts here and doubles
        "retry_max_s": 900,
        "verify_tls": True,
    },
    "storage": {
        "root": "/var/lib/gaitedge",
        "max_disk_gb": 20.0,     # oldest finished jobs are pruned past this
        "keep_raw_after_upload": False,  # privacy: delete raw walks once safe
        "keep_days": 14,
    },
    "preview": {
        # A live view at http://<host>:<port>/, opened on an attached screen
        # by deploy/preview_kiosk.sh. 127.0.0.1: only from the device itself.
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8080,
        "fps": 8,                # frames a second sent to the view
        "width": 960,            # the view is downscaled to this
    },
    "analysis_overrides": {},
}


@dataclass
class Settings:
    data: dict[str, Any]
    source: Optional[Path] = None
    analysis_config_path: Optional[Path] = None

    # -- access ----------------------------------------------------------
    def __getitem__(self, dotted: str) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            node = node[part]
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return self[dotted]
        except (KeyError, TypeError):
            return default

    @property
    def device_id(self) -> str:
        return self.data["device"]["id"] or socket.gethostname()

    @property
    def root(self) -> Path:
        return Path(self.data["storage"]["root"])

    # -- the analysis config this device measures with ----------------------
    def analysis_config(self):
        """The shared gait-analysis config, with this device's overrides."""
        from gaitscreen.config import Config

        cfg = Config.load(self.analysis_config_path) if self.analysis_config_path \
            else Config.load()
        overrides = copy.deepcopy(self.data.get("analysis_overrides") or {})
        variant = self.data["processing"]["model"]
        if variant in MODEL_VARIANTS:
            overrides = _deep_merge(model_overrides(variant, self.root), overrides)
        return cfg.with_overrides(overrides) if overrides else cfg

    def describe(self) -> dict[str, Any]:
        """What a server needs to know about how this device measured a walk."""
        return {
            "device_id": self.device_id,
            "site": self.data["device"].get("site") or "",
            "pose_model": self.data["processing"]["model"],
            "analysis_overrides": self.data.get("analysis_overrides") or {},
        }


def model_overrides(variant: str, root: Path) -> dict[str, Any]:
    """Analysis-config overrides that select one pose model variant.

    Models are cached under the device's storage root, next to everything
    else it writes, so a read-only install directory still works.
    """
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown pose model {variant!r}; use one of {MODEL_VARIANTS}")
    return {"pose": {
        "model_path": str(root / "models" / f"pose_landmarker_{variant}.task"),
        "model_url": MODEL_URL.format(v=variant),
        "auto_download": True,
    }}


def load(path: Optional[str | Path] = None) -> Settings:
    """Defaults, then the YAML file at ``path`` merged over them."""
    data = copy.deepcopy(DEFAULTS)
    source = None
    analysis_path = None
    if path is not None:
        source = Path(path)
        with open(source, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        analysis_path = loaded.pop("analysis_config", None)
        data = _deep_merge(data, loaded)
        if analysis_path:
            analysis_path = (source.parent / analysis_path).resolve()
    _validate(data)
    return Settings(data=data, source=source,
                    analysis_config_path=Path(analysis_path) if analysis_path else None)


def _validate(data: dict) -> None:
    problems = []
    if data["camera"]["backend"] not in ("picamera2", "opencv", "file"):
        problems.append("camera.backend must be picamera2, opencv or file")
    if data["camera"]["backend"] == "file" and not data["camera"]["source"]:
        problems.append("camera.source is required for the file backend")
    if data["trigger"]["mode"] not in ("motion", "manual", "continuous"):
        problems.append("trigger.mode must be motion, manual or continuous")
    if data["processing"]["mode"] not in ("local", "remote", "auto"):
        problems.append("processing.mode must be local, remote or auto")
    if data["processing"]["model"] not in MODEL_VARIANTS:
        problems.append(f"processing.model must be one of {MODEL_VARIANTS}")
    if data["record"]["codec"] not in ("h264", "mjpeg"):
        problems.append("record.codec must be h264 or mjpeg")
    if data["camera"]["rotate"] not in (0, 90, 180, 270):
        problems.append("camera.rotate must be 0, 90, 180 or 270")
    if problems:
        raise ValueError("invalid gaitedge settings: " + "; ".join(problems))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
