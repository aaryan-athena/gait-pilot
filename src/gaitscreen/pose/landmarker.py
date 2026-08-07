"""MediaPipe Pose landmark extraction.

Uses the MediaPipe Tasks ``PoseLandmarker``. The legacy ``mediapipe.solutions``
module has been removed in current releases (0.10.3x), so the Tasks API is the
only option and a ``.task`` model bundle must be present on disk -- see
``models/`` and the README for the download command.

Running mode is VIDEO rather than IMAGE: it keeps MediaPipe's internal tracking
between frames, which markedly improves landmark continuity on limbs that are
briefly self-occluded by the near leg in a sagittal view.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")

from mediapipe import Image, ImageFormat  # noqa: E402
from mediapipe.tasks.python import BaseOptions  # noqa: E402
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions  # noqa: E402
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (  # noqa: E402
    VisionTaskRunningMode,
)

from ..config import Config  # noqa: E402
from ..io import video as video_io  # noqa: E402
from ..pose.schema import N_LANDMARKS  # noqa: E402
from ..types import RawLandmarks, VideoInfo  # noqa: E402

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

#: Below this the file is a truncated download or an error page, not a model.
_MIN_MODEL_BYTES = 1_000_000


def _resolve_model(cfg: Config, project_root: str | Path) -> Path:
    """Locate the pose model, fetching it on first use if allowed.

    The bundle is ~30 MB, so it is not committed to the repository. A hosted
    deployment has no opportunity to run a setup command between cloning the
    repo and starting the app, so the download has to happen here or the app
    cannot work there at all. Locally this replaces the one-off curl step.
    """
    model_path = cfg.resolve_path("pose.model_path", project_root)

    if model_path.exists():
        if model_path.stat().st_size >= _MIN_MODEL_BYTES:
            return model_path
        # A partial download left behind by an interrupted first run would
        # otherwise fail with an opaque error on every subsequent run.
        model_path.unlink()

    url = str(cfg.get("pose.model_url") or MODEL_URL)
    if not bool(cfg.get("pose.auto_download", True)):
        raise FileNotFoundError(
            f"pose model not found at {model_path} and pose.auto_download is "
            f"disabled.\nDownload it with:\n  curl -sSL -o {model_path} {url}"
        )

    return _download_model(url, model_path)


def _download_model(url: str, destination: Path) -> Path:
    """Fetch the model to a temporary file, then move it into place.

    Downloading straight to the destination would leave a truncated file that
    looks valid if the process is interrupted -- and on a hosted deployment,
    something that breaks only on restart is the worst kind of failure.
    """
    import shutil
    import tempfile
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gaitscreen] downloading pose model (~30 MB) to {destination} …")

    handle, temporary = tempfile.mkstemp(
        dir=destination.parent, suffix=".task.partial"
    )
    os.close(handle)
    temporary_path = Path(temporary)

    try:
        with urllib.request.urlopen(url, timeout=120) as response, \
                open(temporary_path, "wb") as out:
            shutil.copyfileobj(response, out)

        if temporary_path.stat().st_size < _MIN_MODEL_BYTES:
            raise OSError(
                f"downloaded file is only {temporary_path.stat().st_size} bytes; "
                "the model URL may have moved"
            )
        # os.replace is atomic on the same filesystem, so a second process can
        # never observe a half-written model.
        os.replace(temporary_path, destination)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download the pose model from {url}: {exc}\n"
            f"Fetch it manually with:\n  curl -sSL -o {destination} {url}"
        ) from exc

    print(f"[gaitscreen] pose model ready ({destination.stat().st_size / 1e6:.0f} MB)")
    return destination


def extract(
    info: VideoInfo,
    cfg: Config,
    *,
    project_root: str | Path = ".",
    progress: Optional[callable] = None,
) -> RawLandmarks:
    """Run pose estimation over a whole video and return the raw archive.

    Undetected frames are kept as NaN rather than dropped: the downstream
    filtering and peak detection both assume a uniformly spaced time base, and
    silently closing gaps would shorten stride times.
    """
    model_path = _resolve_model(cfg, project_root)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=int(cfg["pose.num_poses"]),
        min_pose_detection_confidence=float(cfg["pose.min_detection_confidence"]),
        min_tracking_confidence=float(cfg["pose.min_tracking_confidence"]),
        min_pose_presence_confidence=float(cfg["pose.min_presence_confidence"]),
    )

    max_frames = int(cfg.get("video.max_frames", 0) or 0)
    xy_rows: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []
    vis_rows: list[np.ndarray] = []
    detected: list[bool] = []

    nan_xy = np.full((N_LANDMARKS, 2), np.nan)
    nan_1d = np.full(N_LANDMARKS, np.nan)

    with PoseLandmarker.create_from_options(options) as landmarker:
        last_ts = -1
        for index, rgb in video_io.frames(info.path, max_frames=max_frames, to_rgb=True):
            # VIDEO mode requires strictly increasing integer millisecond stamps.
            ts_ms = max(int(round(index * 1000.0 / info.fps)), last_ts + 1)
            last_ts = ts_ms

            result = landmarker.detect_for_video(
                Image(image_format=ImageFormat.SRGB, data=rgb), ts_ms
            )
            if result.pose_landmarks:
                marks = result.pose_landmarks[0]
                xy_rows.append(np.array([[m.x, m.y] for m in marks], dtype=float))
                z_rows.append(np.array([m.z for m in marks], dtype=float))
                vis_rows.append(np.array([m.visibility for m in marks], dtype=float))
                detected.append(True)
            else:
                xy_rows.append(nan_xy.copy())
                z_rows.append(nan_1d.copy())
                vis_rows.append(np.zeros(N_LANDMARKS))
                detected.append(False)

            if progress is not None and index % 25 == 0:
                progress(index)

    n = len(xy_rows)
    if n == 0:
        raise ValueError(f"no frames could be read from {info.path}")

    # Trust the frames we actually read over the container's frame count.
    info.n_frames = n

    return RawLandmarks(
        t=np.arange(n, dtype=float) / info.fps,
        xy=np.stack(xy_rows),
        z=np.stack(z_rows),
        visibility=np.stack(vis_rows),
        detected=np.array(detected, dtype=bool),
        video=info,
    )


def subject_boxes(
    raw: RawLandmarks,
) -> list[Optional[tuple[float, float, float, float]]]:
    """Per-frame subject bounding boxes in image pixels (y down).

    Used to mask the subject out of camera-motion estimation.
    """
    width, height = raw.video.width, raw.video.height
    boxes: list[Optional[tuple[float, float, float, float]]] = []
    for i in range(raw.n_frames):
        if not raw.detected[i]:
            boxes.append(None)
            continue
        xs = raw.xy[i, :, 0] * width
        ys = raw.xy[i, :, 1] * height
        if not np.isfinite(xs).any():
            boxes.append(None)
            continue
        boxes.append(
            (float(np.nanmin(xs)), float(np.nanmin(ys)),
             float(np.nanmax(xs)), float(np.nanmax(ys)))
        )
    return boxes
