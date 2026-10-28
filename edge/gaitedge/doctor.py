"""`gaitedge doctor`: check an install end to end, in plain pass/fail lines.

Meant for the moment a new device is set up, when the useful question is not
"is the code right" but "will this unit work where it is": does the camera
deliver frames at the rate asked for, is there disk, can it reach the server.
"""
from __future__ import annotations

import shutil
import time

from .settings import Settings


def run_checks(settings: Settings) -> bool:
    results: list[tuple[bool, str, str]] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn() or ""
            results.append((True, name, str(detail)))
        except Exception as exc:  # noqa: BLE001
            results.append((False, name, str(exc)))

    check("python packages", _imports)
    if settings["camera.backend"] == "picamera2":
        check("picamera2", _picamera2)
    check("ffmpeg (annotated video)", _ffmpeg)
    check("storage", lambda: _storage(settings))
    check("pose model", lambda: _model(settings))
    check("camera frame rate", lambda: _camera(settings))
    if settings["upload.server_url"]:
        check("server", lambda: _server(settings))
    check("benchmark", lambda: _bench(settings))

    for ok, name, detail in results:
        print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return all(ok for ok, _, _ in results)


def _imports() -> str:
    import av
    import cv2
    import mediapipe
    import numpy

    import gaitscreen  # noqa: F401

    return (f"mediapipe {mediapipe.__version__}, opencv {cv2.__version__}, "
            f"av {av.__version__}, numpy {numpy.__version__}")


def _picamera2() -> str:
    from picamera2 import Picamera2

    cams = Picamera2.global_camera_info()
    if not cams:
        raise RuntimeError("no camera detected on the ribbon connector")
    return ", ".join(c.get("Model", "?") for c in cams)


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("not installed; `sudo apt install ffmpeg`")
    return path


def _storage(settings: Settings) -> str:
    root = settings.root
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".write-test"
    probe.write_text("ok")
    probe.unlink()
    free = shutil.disk_usage(root).free / 1e9
    allowance = float(settings["storage.max_disk_gb"])
    if free < allowance * 0.25:
        raise RuntimeError(f"only {free:.1f} GB free under {root}")
    return f"{root}, {free:.1f} GB free, allowance {allowance:g} GB"


def _model(settings: Settings) -> str:
    from gaitscreen.pose.landmarker import _resolve_model

    path = _resolve_model(settings.analysis_config(), settings.root)
    return f"{settings['processing.model']} ({path.stat().st_size / 1e6:.0f} MB)"


def _camera(settings: Settings) -> str:
    from .camera import open_source

    want = float(settings["camera.fps"])
    with open_source(settings) as source:
        stamps = []
        start = time.monotonic()
        for _, t in source.frames():
            stamps.append(t)
            if len(stamps) >= int(want * 3) or time.monotonic() - start > 10:
                break
    if len(stamps) < 10:
        raise RuntimeError(f"only {len(stamps)} frames in 10 s")
    fps = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-9)
    detail = f"{source.width}x{source.height} at {fps:.1f} fps (asked for {want:g})"
    if fps < 0.9 * want:
        raise RuntimeError(detail + "; the camera or its mode cannot reach the rate set")
    return detail


def _server(settings: Settings) -> str:
    from .uploader import Uploader

    if not Uploader(settings).heartbeat({"doctor": True, "device": settings.describe()}):
        raise RuntimeError(f"no answer from {settings['upload.server_url']}")
    return settings["upload.server_url"]


def _bench(settings: Settings) -> str:
    from .bench import load_latest

    latest = load_latest(settings)
    if not latest:
        raise RuntimeError("not run yet: `gaitedge bench --clip <a recorded walk>`")
    rec = latest["recommendation"]
    return f"{latest['measured_at']}: process {rec['mode']}" + (
        f" with {rec['model']}" if rec.get("model") else "")
