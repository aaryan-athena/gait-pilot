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
    """Yield ``(frame_index, image)`` pairs. Images are BGR unless ``to_rgb``.

    ``frame_index`` counts decoded frames. On a clip where the camera dropped
    frames that is not the row in the landmark grid; see
    :attr:`RawLandmarks.frame_rows`.
    """
    for index, image, _ in timed_frames(path, max_frames=max_frames, to_rgb=to_rgb):
        yield index, image


def timed_frames(
    path: str | Path, *, max_frames: int = 0, to_rgb: bool = False
) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield ``(frame_index, image, timestamp_ms)`` for every decoded frame.

    The timestamp is the one the camera stamped on the frame, read straight
    after decoding it. It is the only record of *when* a frame was captured:
    the frame count cannot say, because a phone in dim light skips frames and
    writes nothing where they would have been.
    """
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
            ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            yield index, (
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if to_rgb else frame
            ), ts_ms
            index += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Timing integrity
# --------------------------------------------------------------------------
@dataclass
class FrameGrid:
    """Where each decoded frame belongs on the camera's nominal frame grid."""

    rows: np.ndarray  # (n_decoded,) int, strictly increasing, rows[0] == 0
    fps: float  # nominal grid rate, e.g. 59.94 -- not the container average
    from_timestamps: bool  # False if the timestamps were unusable

    @property
    def n_rows(self) -> int:
        return int(self.rows[-1]) + 1 if self.rows.size else 0

    @property
    def dropped(self) -> int:
        return self.n_rows - int(self.rows.size)

    @property
    def max_gap(self) -> int:
        return int(np.max(np.diff(self.rows)) - 1) if self.rows.size > 1 else 0


def nominal_interval_ms(timestamps_ms: np.ndarray) -> Optional[float]:
    """The camera's nominal frame interval, measured from its own timestamps.

    Not the container's frame rate, for two reasons. On a variable-rate phone
    clip the container reports the *average* rate -- 43.2 fps on a pilot clip
    shot at 60 -- which is the one number guaranteed to be wrong for every
    individual frame. And the nominal rate cannot be assumed either: many
    phones shoot at 59.94, and treating that as 60 misplaces a frame every
    sixteen seconds, which over a minute-long walk is a clock error larger than
    the stride-time variability being measured.

    Two stages. The commonest interval gives a coarse value: skipped frames
    only ever lengthen an interval, so as long as most frames arrived the
    nominal interval is the mode. (Not the minimum, nor the shortest few: with
    timestamp jitter the shortest intervals are the nominal one minus the
    jitter, and a grid built on them invents skipped frames everywhere.) Then
    it is refined by least squares against the absolute timestamps, which
    averages the jitter out over the whole clip and resolves 59.94 from 60.
    """
    ts = np.asarray(timestamps_ms, dtype=float)
    deltas = np.diff(ts)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size < 10:
        return None

    candidates = np.quantile(deltas, np.linspace(0.02, 0.6, 59))
    support = [np.count_nonzero(np.abs(deltas - c) <= 0.15 * c) for c in candidates]
    mode = float(candidates[int(np.argmax(support))])
    # Every candidate near the mode scores alike, so the winner can sit a few
    # percent off it. The median of the cluster it picks out does not.
    interval = float(np.median(deltas[np.abs(deltas - mode) <= 0.25 * mode]))

    # A 1% error in the interval puts a frame on the wrong row every hundred
    # frames, and a least-squares fit over rows assigned with that error just
    # reproduces it. So the fit starts over the first stretch of the clip,
    # where the coarse value is still good enough to count frames, and extends.
    elapsed = ts - ts[0]
    for share in (0.05, 0.2, 0.5, 1.0):
        span = elapsed[: max(10, int(share * elapsed.size))]
        steps = np.round(span / interval)
        if not np.any(steps):
            return None
        interval = float(np.dot(steps, span) / np.dot(steps, steps))
    return interval


def frame_grid(
    timestamps_ms: np.ndarray, container_fps: float, *, max_drop_fraction: float = 0.6
) -> FrameGrid:
    """Place each decoded frame on the nominal grid by its timestamp.

    Pure function, so the arithmetic is testable without a video file. Frames
    the camera skipped show up as rows no decoded frame lands on.

    Falls back to one row per decoded frame at the container rate -- exactly
    what the pipeline assumed before -- when the timestamps cannot be used: a
    backend that reports zeros, time running backwards, or a grid that would
    claim most of the clip is missing. A fallback can under-report drops but
    never invent them.
    """
    ts = np.asarray(timestamps_ms, dtype=float)
    n = ts.size

    def uniform() -> FrameGrid:
        return FrameGrid(rows=np.arange(n), fps=float(container_fps),
                         from_timestamps=False)

    if n < 2 or not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
        return uniform()
    interval = nominal_interval_ms(ts)
    if interval is None or interval <= 0:
        return uniform()

    rows = np.round((ts - ts[0]) / interval).astype(int)
    # Jitter larger than half an interval could put two frames on one row.
    # Keep order, and push the later one to the next free row.
    for i in range(1, n):
        if rows[i] <= rows[i - 1]:
            rows[i] = rows[i - 1] + 1

    grid = FrameGrid(rows=rows, fps=1000.0 / interval, from_timestamps=True)
    if grid.dropped > max_drop_fraction * grid.n_rows:
        return uniform()
    return grid


def describe_drops(info: VideoInfo, cfg: Config) -> list[str]:
    """Plain-language warning for camera-dropped frames, if worth one."""
    if info.dropped_frames <= 0:
        return []
    rate = info.drop_rate_pct
    if rate < float(cfg["video.timing_warn_drop_rate_pct"]):
        return []
    longest_ms = info.max_gap_frames * 1000.0 / info.fps if info.fps else 0.0
    return [
        f"the camera skipped {info.dropped_frames} frames ({rate:.1f}% of the "
        f"recording, longest gap {longest_ms:.0f} ms) -- usually a sign of dim "
        "light. Timing was rebuilt from the camera's own timestamps, so stride "
        "times are measured on the real clock; the skipped frames were filled "
        "in the same way as a briefly hidden ankle."
    ]


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
