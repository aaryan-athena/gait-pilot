"""Measurement from a towards-camera (coronal) recording.

This is a separate, smaller measurement -- not a fallback that fills in for the
sagittal one. The two views see different things, and the honest response to a
coronal clip is to report the few quantities it genuinely supports and refuse
the rest, rather than to produce a full-looking result from a view that cannot
support it.

What a coronal view *cannot* give, at all:

* step length and gait speed -- the walking direction points along the camera
  axis, so a 70 cm step projects to a handful of pixels of scale change;
* heel strike and toe-off by the Zeni rule, which reads anterior position, the
  one axis that has been projected away;
* double support, which is built from toe-off;
* step-length asymmetry, for the same reason.

What it gives that a side view *cannot*:

* **step width** -- how far apart the feet are placed side to side. A side view
  hides this completely, one leg behind the other. It is the measure people
  widen when they feel unsteady, so it is worth having.
* **lateral trunk sway** -- side-to-side motion of the upper body over the
  pelvis. The original brief wanted this and settled for anterior-posterior
  lean from the side view as a substitute; a coronal clip measures the real
  thing.

And what it gives redundantly, but as a useful cross-check:

* **cadence and mean stride time**, from the vertical bob of each ankle
  relative to the pelvis. Feet rise and fall in every view.

Stride-time *variability* is deliberately not reported. The period is stable
enough to average over a pass, but individual heel strikes cannot be located
precisely enough in this view to time one stride against the next, and a CV
computed from imprecise event times measures the detector rather than the
person.

Everything here is normalised per frame by the subject's own leg length, which
in a coronal clip changes by a factor of three between the near and far end of
the walk. A single whole-clip scale, as the sagittal path uses, would leave
every measurement dominated by the subject's approach rather than their gait.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..pose.schema import HIPS, SHOULDERS
from ..quality.view import frame_scale
from ..segmentation.events import _detrend
from ..types import PixelSeries


@dataclass
class CoronalPass:
    """One approach to or retreat from the camera, turns excluded."""

    start_frame: int
    stop_frame: int
    towards_camera: bool

    def duration_s(self, fps: float) -> float:
        return (self.stop_frame - self.start_frame) / fps


@dataclass
class CoronalAnalysis:
    """What a towards-camera recording supports, and nothing beyond it."""

    passes: list[CoronalPass] = field(default_factory=list)
    step_width_norm: float | None = None
    trunk_lateral_sway_norm: float | None = None
    cadence_spm: float | None = None
    stride_time_mean_s: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_passes(self) -> int:
        return len(self.passes)


_DISTANCE_REASON = (
    "the camera was facing the walk rather than side-on to it, so the direction "
    "of travel points away from the camera and distances along it cannot be "
    "measured"
)

_EVENT_REASON = (
    "this needs the exact moment each foot lifts off the ground, which is read "
    "from the foot's forward motion and so needs a side-on view; the camera was "
    "facing the walk instead"
)


#: Metrics that exist only because the walk was recorded side-on: on a coronal
#: clip every one must be refused, not estimated, and this says why. They do not
#: all fail for the same reason, and saying they did would send someone looking
#: for the wrong fix: stride-time variability is not a distance measurement, and
#: no amount of calibration would recover it from this view.
SAGITTAL_ONLY_METRICS: dict[str, str] = {
    "gait_speed_mps": _DISTANCE_REASON,
    "step_length_asymmetry_pct": _DISTANCE_REASON,
    "step_time_asymmetry_pct": _EVENT_REASON,
    "double_support_pct": _EVENT_REASON,
    "stride_time_cv_pct": (
        "timing one stride against the next needs each heel strike located to "
        "within a frame or two, which needs a side-on view; this recording was "
        "filmed towards the camera, where the foot's forward motion is hidden"
    ),
    "trunk_ap_sway_norm": (
        "forward-and-back trunk lean is measured across the image in a side-on "
        "recording; filmed towards the camera it points at the lens and cannot "
        "be seen. Side-to-side sway is reported instead"
    ),
}


def find_passes(series: PixelSeries, cfg: Config) -> list[CoronalPass]:
    """Split a coronal recording into approaches and retreats.

    A person walking back and forth in front of the camera spends part of the
    clip turning round, and a turn is not gait: the shoulders rotate through
    ninety degrees, the feet cross, and -- because left and right swap sides of
    the image -- the distance between the ankles passes through zero. Pooling
    turns in with the walking would corrupt step width and sway far more than
    it would add data.

    Direction comes from whether the subject is growing or shrinking in the
    frame, smoothed over more than a stride so the scale wobble from knee
    flexion does not read as a change of direction. A margin is then trimmed
    from both ends of each run, because the subject is decelerating into the
    turn and accelerating out of it, and neither is representative walking --
    the same reason the sagittal path drops strides at pass edges.
    """
    section = cfg.section("coronal")
    fps = series.fps
    scale = frame_scale(series)

    window = max(3, int(round(float(section["direction_smooth_s"]) * fps)))
    rate = np.convolve(np.nan_to_num(np.gradient(scale)),
                       np.ones(window) / window, mode="same")
    direction = np.sign(rate)

    min_frames = max(2, int(round(float(section["min_pass_s"]) * fps)))
    margin = int(round(float(section["pass_edge_trim_s"]) * fps))

    passes: list[CoronalPass] = []
    start = 0
    for index in range(1, series.n_frames + 1):
        if index < series.n_frames and direction[index] == direction[start]:
            continue
        if index - start >= min_frames:
            inner_start, inner_stop = start + margin, index - margin
            if inner_stop - inner_start >= min_frames // 2:
                passes.append(CoronalPass(
                    start_frame=inner_start, stop_frame=inner_stop,
                    # The subject grows in the frame as they approach, so a
                    # rising apparent scale means towards the camera.
                    towards_camera=bool(direction[start] > 0),
                ))
        start = index
    return passes


def analyse_coronal(series: PixelSeries, cfg: Config) -> CoronalAnalysis:
    """Measure what a towards-camera recording supports."""
    result = CoronalAnalysis()
    result.passes = find_passes(series, cfg)

    if not result.passes:
        result.notes.append(
            "no steady approach or retreat could be found in this recording, so "
            "nothing could be measured from it; the subject may have turned "
            "continuously, or the walk may be too short"
        )
        return result

    scale = frame_scale(series)
    left_ankle = series.joint("left", "ankle")
    right_ankle = series.joint("right", "ankle")
    pelvis = series.midpoint(*HIPS)
    shoulders = series.midpoint(*SHOULDERS)

    widths: list[np.ndarray] = []
    sways: list[np.ndarray] = []
    periods: dict[str, list[float]] = {"left": [], "right": []}

    for walk in result.passes:
        span = slice(walk.start_frame, walk.stop_frame)
        unit = scale[span]

        # Distance between the ankles across the image. Absolute, because the
        # two legs trade sides of the frame when the subject turns round, and a
        # signed value would cancel between an approach and a retreat.
        widths.append(np.abs(left_ankle[span, 0] - right_ankle[span, 0]) / unit)

        # Shoulders relative to hips: the trunk leaning side to side, with the
        # subject's own translation across the frame already removed.
        lateral = (shoulders[span, 0] - pelvis[span, 0]) / unit
        sways.append(lateral - np.nanmedian(lateral))

        for side, ankle in (("left", left_ankle), ("right", right_ankle)):
            lift = (ankle[span, 1] - pelvis[span, 1]) / unit
            period = _dominant_period(lift, series.fps, cfg)
            if period is not None:
                periods[side].append(period)

    pooled_width = np.concatenate(widths)
    pooled_sway = np.concatenate(sways)
    if int(np.isfinite(pooled_width).sum()) >= series.fps:
        result.step_width_norm = float(np.nanmedian(pooled_width))
    if int(np.isfinite(pooled_sway).sum()) >= series.fps:
        result.trunk_lateral_sway_norm = float(np.nanstd(pooled_sway))

    _resolve_cadence(result, periods, cfg)

    towards = sum(1 for p in result.passes if p.towards_camera)
    result.notes.append(
        f"measured from {len(result.passes)} walking pass(es) towards or away "
        f"from the camera ({towards} towards, {len(result.passes) - towards} "
        "away), with the turns between them excluded"
    )
    return result


def _dominant_period(signal: np.ndarray, fps: float, cfg: Config) -> float | None:
    """Stride period from the vertical bob of one ankle over the pelvis.

    Autocorrelation rather than peak counting. The foot rises once per stride,
    but the height of each rise varies with how the subject is being projected
    as they approach, so prominence gating on peaks keeps some rises and drops
    others -- which halves or doubles the answer. Autocorrelation uses the
    whole pass and does not care that the peaks are unequal.
    """
    lo_s = float(cfg["segmentation.min_stride_time_s"])
    hi_s = float(cfg["segmentation.max_stride_time_s"])

    finite = np.isfinite(signal)
    if int(finite.sum()) < int(2 * hi_s * fps):
        return None
    filled = np.where(finite, signal, np.nanmean(signal))
    # The subject's apparent size still drifts within a pass; without removing
    # it the autocorrelation is dominated by that ramp and simply returns its
    # shortest allowed lag.
    values = _detrend(filled, fps, hi_s)
    values = values - values.mean()

    correlation = np.correlate(values, values, mode="full")[values.size - 1:]
    lo, hi = int(lo_s * fps), min(int(hi_s * fps), correlation.size)
    if hi <= lo:
        return None
    return float((lo + int(np.argmax(correlation[lo:hi]))) / fps)


def _resolve_cadence(result: CoronalAnalysis, periods: dict[str, list[float]],
                     cfg: Config) -> None:
    """Accept a cadence only when the two legs independently agree on it.

    The two ankles are tracked separately but belong to one gait cycle, so
    their stride periods must match. When they do not, the autocorrelation has
    locked onto different things on the two sides -- typically a harmonic on
    one of them -- and the number would be wrong rather than merely imprecise.
    """
    left, right = periods["left"], periods["right"]
    if not left or not right:
        result.notes.append(
            "the feet could not be tracked steadily enough to time the walk, so "
            "cadence is not reported for this recording"
        )
        return

    left_med, right_med = float(np.median(left)), float(np.median(right))
    mean_period = 0.5 * (left_med + right_med)
    disagreement = abs(left_med - right_med) / mean_period
    tolerance = float(cfg.section("coronal")["max_leg_period_disagreement"])

    if disagreement > tolerance:
        result.notes.append(
            f"the two legs gave stride times {disagreement:.0%} apart "
            f"({left_med:.2f}s and {right_med:.2f}s), so the walk could not be "
            "timed reliably and cadence is not reported"
        )
        return

    result.stride_time_mean_s = mean_period
    result.cadence_spm = 120.0 / mean_period
