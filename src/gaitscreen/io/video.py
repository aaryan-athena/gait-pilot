"""Video ingestion: probing, frame iteration, and camera-motion estimation."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ..config import Config
from ..types import VideoInfo


def probe(path: str | Path, cfg: Config) -> VideoInfo:
    """Read container metadata and record any quality warnings.

    Warnings are attached rather than raised: a low-fps recording is still worth
    processing, it just cannot support every metric. The fps warning matters
    more than it looks -- see :func:`fps_warnings`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    warnings: list[str] = []
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"video reports an unusable frame rate ({fps}): {path}")
    if fps > 240:
        warnings.append(
            f"reported frame rate {fps:g} fps is implausible; container metadata "
            "may be wrong, which would scale every timing metric"
        )
    if n_frames <= 0:
        warnings.append("frame count unavailable from container; will count while reading")

    warnings.extend(fps_warnings(fps, cfg))

    if height < cfg["video.min_frame_height"]:
        warnings.append(
            f"frame height {height}px is below the recommended "
            f"{cfg['video.min_frame_height']}px; foot landmarks will be noisier"
        )

    return VideoInfo(
        path=path, width=width, height=height, fps=fps,
        n_frames=max(n_frames, 0), warnings=warnings,
    )


def fps_warnings(fps: float, cfg: Config) -> list[str]:
    """Frame-rate warnings, including the one that limits variability metrics.

    Stride-time variability is the headline fall-risk metric, and it is the one
    most damaged by temporal quantisation. Healthy elderly stride time is around
    1.1 s with a CV of 2-3%, i.e. a standard deviation of roughly 25-33 ms. At
    25 fps one frame is 40 ms -- larger than the entire quantity being measured.
    Sub-frame event interpolation recovers a good deal of this, but not all of
    it, so the limitation is surfaced rather than hidden.
    """
    out: list[str] = []
    warn_below = cfg["video.fps_warn_below"]
    var_min = cfg["video.fps_variability_min"]
    if fps < warn_below:
        out.append(
            f"{fps:g} fps is below the recommended {warn_below:g} fps; "
            "event timing and joint-angle curves will be coarser"
        )
    if fps < var_min:
        out.append(
            f"{fps:g} fps (frame interval {1000 / fps:.0f} ms) is below the "
            f"{var_min:g} fps needed for a trustworthy stride-time variability "
            "measurement: the frame interval is comparable to the stride-time "
            "standard deviation itself. Variability will be reported as "
            "low-confidence. Record at 60 fps for this metric."
        )
    return out


def frames(
    path: str | Path, *, max_frames: int = 0, to_rgb: bool = False
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, image)`` pairs. Images are BGR unless ``to_rgb``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and index >= max_frames:
                break
            yield index, (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if to_rgb else frame)
            index += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Camera motion
# --------------------------------------------------------------------------
@dataclass
class CameraMotion:
    """Background displacement, used to detect a panning or tracking camera.

    A camera that follows the subject cancels the subject's image-space
    translation, which would otherwise be the basis of the gait-speed
    measurement. ``determinate`` is False when the background has too little
    texture to track at all (a plain green screen or blank wall), in which case
    motion is unknown rather than zero.
    """

    net_px: float  # net signed background shift, absolute value
    path_px: float  # total accumulated absolute shift
    determinate: bool
    n_samples: int
    note: Optional[str] = None


_FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7)
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def estimate_camera_motion(
    path: str | Path,
    cfg: Config,
    subject_boxes: Optional[Sequence[Optional[tuple[float, float, float, float]]]] = None,
    *,
    max_frames: int = 0,
) -> CameraMotion:
    """Track background features to measure how much the camera itself moved.

    ``subject_boxes`` are per-frame ``(x0, y0, x1, y1)`` boxes in image
    coordinates (y down) that are masked out before feature selection, so the
    subject's own motion is not mistaken for camera motion.
    """
    step = max(1, int(cfg["speed.camera_motion_sample_every"]))
    prev_gray: Optional[np.ndarray] = None
    prev_index = -1
    net = 0.0
    path_total = 0.0
    n_samples = 0
    n_featureless = 0

    for index, frame in frames(path, max_frames=max_frames):
        if index % step:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray, prev_index = gray, index
            continue

        mask = _background_mask(gray.shape, subject_boxes, prev_index)
        pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **_FEATURE_PARAMS)
        if pts is None or len(pts) < 8:
            n_featureless += 1
            prev_gray, prev_index = gray, index
            continue

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **_LK_PARAMS)
        if nxt is None:
            prev_gray, prev_index = gray, index
            continue
        good = status.ravel() == 1
        if good.sum() < 8:
            n_featureless += 1
            prev_gray, prev_index = gray, index
            continue

        dx = float(np.median((nxt[good] - pts[good])[:, 0, 0]))
        net += dx
        path_total += abs(dx)
        n_samples += 1
        prev_gray, prev_index = gray, index

    if n_samples == 0:
        return CameraMotion(
            net_px=float("nan"), path_px=float("nan"), determinate=False, n_samples=0,
            note=(
                "background has too little texture to track (plain or chroma-key "
                "backdrop), so camera motion cannot be determined"
            ),
        )

    note = None
    if n_featureless > n_samples:
        note = (
            f"background texture was trackable in only {n_samples} of "
            f"{n_samples + n_featureless} sampled intervals"
        )
    return CameraMotion(
        net_px=abs(net), path_px=path_total, determinate=True,
        n_samples=n_samples, note=note,
    )


def _background_mask(
    shape: tuple[int, int],
    subject_boxes: Optional[Sequence[Optional[tuple[float, float, float, float]]]],
    index: int,
) -> Optional[np.ndarray]:
    """Full-frame mask with the subject's (padded) bounding box excluded."""
    if not subject_boxes or index >= len(subject_boxes):
        return None
    box = subject_boxes[index]
    if box is None:
        return None
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    pad_x, pad_y = 0.15 * width, 0.05 * height
    x0 = int(max(0, box[0] - pad_x))
    y0 = int(max(0, box[1] - pad_y))
    x1 = int(min(width, box[2] + pad_x))
    y1 = int(min(height, box[3] + pad_y))
    mask[y0:y1, x0:x1] = 0
    if mask.mean() < 20:  # subject fills the frame; nothing left to track
        return None
    return mask
