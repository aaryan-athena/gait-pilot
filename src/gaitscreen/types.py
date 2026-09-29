"""Core data structures shared across the pipeline.

Coordinate conventions -- worth reading once, because getting these wrong is
the single easiest way to produce plausible nonsense:

* :class:`RawLandmarks` is a faithful archive of what MediaPipe returned:
  normalised coordinates, ``y`` increasing **downward**, ``x`` normalised by
  width and ``y`` by height (so anisotropic -- not directly comparable).
* :class:`PixelSeries` is what every downstream module actually uses: pixel
  units on both axes (isotropic), ``x`` increasing right, ``y`` increasing
  **upward** from a bottom-left origin, matching biomechanics convention so
  "higher" means larger ``y``.
* Direction of progression is resolved per pass and stored as ``direction``
  (+1 = subject moves toward +x). Downstream code uses *anterior* offsets,
  never raw ``x``, so it works in both directions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from .pose.schema import PL, SIDE_LANDMARKS

Side = Literal["left", "right"]
EventKind = Literal["heel_strike", "toe_off"]

#: The five core screening metrics, in the priority order from the brief, plus
#: the secondary temporal-asymmetry variant. Single source of truth for
#: storage columns, flagging rules and report ordering.
CORE_METRICS: tuple[str, ...] = (
    "gait_speed_mps",
    "stride_time_cv_pct",
    "step_length_asymmetry_pct",
    "cadence_spm",
    "double_support_pct",
    "trunk_ap_sway_norm",
    # Coronal-only; null on side-on sessions, which is handled the same way as
    # any other unavailable metric rather than by keeping a second list.
    "step_width_norm",
    "trunk_lateral_sway_norm",
)

#: Direction of clinical deterioration for each metric: -1 means a *decrease*
#: is worse (slower walking), +1 means an *increase* is worse (more
#: variability, more asymmetry, longer double support, more sway).
DETERIORATION_DIRECTION: dict[str, int] = {
    "gait_speed_mps": -1,
    "stride_time_cv_pct": +1,
    "step_length_asymmetry_pct": +1,
    "cadence_spm": -1,
    "double_support_pct": +1,
    "trunk_ap_sway_norm": +1,
    # A wider base and more side-to-side sway are both what people do when they
    # feel unsteady, so an increase in either is the direction of concern.
    "step_width_norm": +1,
    "trunk_lateral_sway_norm": +1,
}


# --------------------------------------------------------------------------
# Video / landmarks
# --------------------------------------------------------------------------
@dataclass
class VideoInfo:
    """Container metadata plus any ingestion warnings."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int
    warnings: list[str] = field(default_factory=list)

    # Timing integrity, filled in by pose extraction from the camera's own
    # per-frame timestamps. Phones record at a variable rate: in dim light they
    # silently skip frames, and the container then reports an *average* rate
    # (43 fps on one pilot clip nominally shot at 60). ``fps`` becomes the
    # nominal grid rate, ``n_frames`` the length of that grid, and each skipped
    # frame a NaN row -- so ``arange(n) / fps`` is the true clock.
    container_fps: Optional[float] = None
    dropped_frames: int = 0
    max_gap_frames: int = 0

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def drop_rate_pct(self) -> float:
        """Share of the frame grid the camera never delivered."""
        return 100.0 * self.dropped_frames / self.n_frames if self.n_frames else 0.0

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / self.fps if self.fps else float("nan")


@dataclass
class RawLandmarks:
    """Unmodified MediaPipe output on a uniform frame index.

    Frames where no pose was detected are present but flagged in ``detected``
    and filled with NaN, so the time base stays uniform -- filtering and peak
    detection both assume evenly spaced samples.
    """

    t: np.ndarray  # (n,) seconds from first frame
    xy: np.ndarray  # (n, 33, 2) normalised, y DOWN
    z: np.ndarray  # (n, 33) non-metric depth, archived only
    visibility: np.ndarray  # (n, 33)
    detected: np.ndarray  # (n,) bool
    video: VideoInfo
    # (n,) bool -- False on rows the camera never delivered a frame for. None
    # means every row is a real frame, which is true of anything recorded
    # before timing was rebuilt from timestamps, and of the synthetic fixtures.
    observed: Optional[np.ndarray] = None

    @property
    def n_frames(self) -> int:
        return int(self.t.shape[0])

    @property
    def observed_mask(self) -> np.ndarray:
        if self.observed is None:
            return np.ones(self.n_frames, dtype=bool)
        return self.observed

    @property
    def frame_rows(self) -> np.ndarray:
        """Row index of each frame actually decoded from the file, in order.

        Anything that walks the video file frame by frame -- the camera-motion
        check, the annotated overlay -- must go through this, because once
        dropped frames are materialised the n-th decoded frame is no longer
        row n.
        """
        return np.flatnonzero(self.observed_mask)

    @property
    def detection_rate(self) -> float:
        """Share of *delivered* frames in which a pose was found.

        Rows for frames the camera dropped are excluded: nothing was there to
        detect, and counting them would blame the tracker for the lighting.
        """
        seen = self.observed_mask
        return float(np.mean(self.detected[seen])) if seen.any() else 0.0


@dataclass
class PixelSeries:
    """Filtered, isotropic, y-up landmark trajectories in pixels."""

    t: np.ndarray  # (n,)
    xy: np.ndarray  # (n, 33, 2) pixels, y UP
    visibility: np.ndarray  # (n, 33)
    valid: np.ndarray  # (n, 33) bool -- False where interpolated or missing
    video: VideoInfo
    # (n,) bool -- rows the camera never delivered a frame for. Still False in
    # ``valid``, because nothing was measured there, but not evidence that the
    # tracker lost the subject either; see :meth:`trusted`.
    camera_gap: Optional[np.ndarray] = None

    @property
    def n_frames(self) -> int:
        return int(self.t.shape[0])

    @property
    def fps(self) -> float:
        return self.video.fps

    def point(self, landmark: PL | int, frames: slice | np.ndarray | None = None) -> np.ndarray:
        """Trajectory of one landmark, shape (n, 2)."""
        series = self.xy[:, int(landmark), :]
        return series if frames is None else series[frames]

    def joint(self, side: str, name: str) -> np.ndarray:
        """Trajectory of a named joint on one side, e.g. ``joint("left", "heel")``."""
        return self.point(SIDE_LANDMARKS[side][name])

    def midpoint(self, a: PL | int, b: PL | int) -> np.ndarray:
        return 0.5 * (self.point(a) + self.point(b))

    def trusted(self, frames: slice, landmarks: Sequence[int]) -> np.ndarray:
        """Samples that can be relied on: measured, or bridging a camera skip.

        ``valid`` is False in two very different situations. When the tracker
        loses a foot it still reports one, inferred from the rest of the body,
        so the neighbouring samples are suspect too -- that is what the event
        confidence gate exists to catch. When the camera skips a frame the
        samples either side are genuine detections and the gap is a few frames
        of a signal with nothing above 6 Hz, so filling it is close to exact.
        On the pilot metronome clip, strides rejected only because of skipped
        frames matched the metronome exactly as well as the rest.
        """
        mask = self.valid[frames][:, list(landmarks)]
        if self.camera_gap is None:
            return mask
        gap = self.camera_gap[frames][:, None]
        filled = np.isfinite(self.xy[frames][:, list(landmarks), 0])
        return mask | (gap & filled)

    def slice_frames(self, start: int, stop: int) -> "PixelSeries":
        return PixelSeries(
            t=self.t[start:stop],
            xy=self.xy[start:stop],
            visibility=self.visibility[start:stop],
            valid=self.valid[start:stop],
            video=self.video,
            camera_gap=None if self.camera_gap is None else self.camera_gap[start:stop],
        )


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
@dataclass
class WalkPass:
    """One continuous walk in a single direction, turns excluded."""

    start_frame: int
    end_frame: int  # exclusive
    direction: int  # +1 if subject moves toward +x, else -1
    camera_side: Optional[Side] = None  # body side nearer the camera, if known

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class GaitEvent:
    """A gait event at sub-frame time resolution.

    ``t`` is interpolated between frames on purpose: at 25-30 fps the frame
    interval (33-40 ms) is as large as the entire stride-time standard
    deviation being measured, so integer frame indices would quantise the
    signal away. ``frame`` is retained for traceability.
    """

    t: float
    kind: EventKind
    side: Side
    frame: int
    confidence: float = 1.0
    method: str = "zeni"


@dataclass
class GaitCycle:
    """One stride: heel-strike to next heel-strike on the same side."""

    side: Side
    t_start: float
    t_end: float
    start_frame: int
    end_frame: int
    pass_index: int
    valid: bool = True
    exclusion_reason: Optional[str] = None
    #: Opposite-limb toe-off / heel-strike times inside this cycle, when found.
    contra_toe_off_t: Optional[float] = None
    contra_heel_strike_t: Optional[float] = None
    ipsi_toe_off_t: Optional[float] = None

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


# --------------------------------------------------------------------------
# Feasibility / quality
# --------------------------------------------------------------------------
@dataclass
class SpeedFeasibility:
    """Whether image-space gait speed is measurable from this recording.

    Treadmill, walking-in-place and camera-tracked footage all produce near-zero
    subject translation in image space. Speed is refused in those cases rather
    than estimated, because a fabricated speed would flow straight into the
    trend baseline and corrupt it permanently.
    """

    feasible: bool
    subject_translation_px: float
    subject_translation_frac: float
    camera_motion_px: float
    camera_motion_frac: float
    reason: Optional[str] = None


@dataclass
class QualityReport:
    score: float
    low_confidence: bool
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    assistive_device: Optional[str] = None
    assistive_device_source: Optional[str] = None  # "metadata" | "heuristic"


# --------------------------------------------------------------------------
# Session output
# --------------------------------------------------------------------------
@dataclass
class SessionMetrics:
    """Scalar metrics for one session.

    Any metric can be ``None``, with the reason recorded in ``unavailable``.
    A missing metric is reported as missing -- never silently substituted with a
    default, which in a screening tool would read as a normal result.
    """

    gait_speed_mps: Optional[float] = None
    stride_time_cv_pct: Optional[float] = None
    step_length_asymmetry_pct: Optional[float] = None
    step_time_asymmetry_pct: Optional[float] = None
    cadence_spm: Optional[float] = None
    double_support_pct: Optional[float] = None
    trunk_ap_sway_norm: Optional[float] = None

    # Coronal-only. A side-on recording cannot see either of these -- one leg
    # hides the other, and side-to-side trunk motion is projected away -- so
    # they stay None on every sagittal session rather than being estimated.
    step_width_norm: Optional[float] = None
    trunk_lateral_sway_norm: Optional[float] = None

    stride_time_mean_s: Optional[float] = None
    stride_time_sd_s: Optional[float] = None
    n_strides_total: int = 0
    n_strides_valid: int = 0
    n_passes: int = 0

    unavailable: dict[str, str] = field(default_factory=dict)
    low_confidence_metrics: dict[str, str] = field(default_factory=dict)

    def value(self, metric: str) -> Optional[float]:
        return getattr(self, metric, None)

    def mark_unavailable(self, metric: str, reason: str) -> None:
        setattr(self, metric, None)
        self.unavailable[metric] = reason


@dataclass
class Flag:
    """One triggered alert, with a plain-language explanation.

    ``message`` is written for a caregiver to read; ``detail`` carries the raw
    numbers so a clinician can check the arithmetic.
    """

    code: str
    metric: str
    severity: Literal["moderate", "high"]
    trigger: Literal["absolute", "trend", "quality"]
    message: str
    detail: dict[str, float | str | None] = field(default_factory=dict)
    confirmed: bool = False


@dataclass
class AngleCurves:
    """Joint angle curves resampled to 0-100% of the gait cycle.

    Stored per side as (n_cycles, n_points) so both the mean curve and its
    cycle-to-cycle spread survive to the report stage.
    """

    percent: np.ndarray  # (n_points,)
    curves: dict[str, np.ndarray] = field(default_factory=dict)  # "left_knee" -> (n_cycles, n_points)

    def mean_curve(self, key: str) -> np.ndarray:
        return np.nanmean(self.curves[key], axis=0)
