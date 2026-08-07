"""Gait event detection: heel strike and toe-off.

Primary method is the coordinate-based rule of Zeni et al. (2008), which is the
standard for marker-less and marker-based kinematics alike:

* **Heel strike** = local *maximum* of the heel's anterior position relative to
  the pelvis. At contact the foot is at its most forward point in the gait cycle.
* **Toe-off** = local *minimum* of the toe's anterior position relative to the
  pelvis. At push-off the foot is at its most rearward point.

Working in the pelvis reference frame is what makes this robust: it removes the
subject's own forward travel, so the same rule applies to overground and
treadmill walking, and to a camera that is following the subject.

The brief proposed detecting heel strike from ankle vertical position or velocity
minima. That marks foot-flat rather than initial contact -- it occurs after heel
strike, by a variable amount that depends on ankle stiffness -- and it relies on
the noisiest landmarks MediaPipe produces. It is retained here as a cross-check
(``ankle_velocity``), and the agreement between the two methods is reported as a
diagnostic, but it is not what the metrics are built on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..pose.schema import HIPS, SIDE_LANDMARKS, SIDES
from ..signal.events import find_extrema, fractional_time, refine_extremum
from ..types import GaitEvent, PixelSeries, WalkPass


@dataclass
class EventSet:
    """All events found in one pass, plus method-agreement diagnostics."""

    events: list[GaitEvent]
    method: str
    agreement_ms: float | None = None  # median |zeni - fallback| heel-strike offset
    note: str | None = None

    def of(self, kind: str, side: str | None = None) -> list[GaitEvent]:
        return [
            e for e in self.events
            if e.kind == kind and (side is None or e.side == side)
        ]


def detect_events(
    series: PixelSeries, walk_pass: WalkPass, cfg: Config
) -> EventSet:
    """Find heel strikes and toe-offs for both legs within one pass."""
    method = str(cfg["segmentation.method"])
    window = series.slice_frames(walk_pass.start_frame, walk_pass.end_frame)

    if method == "zeni":
        events = _zeni_events(window, walk_pass, cfg)
    elif method == "ankle_velocity":
        events = _ankle_velocity_events(window, walk_pass, cfg)
    else:
        raise ValueError(f"unknown segmentation method: {method!r}")

    agreement, note = _cross_check(window, walk_pass, cfg, events, method)
    events.sort(key=lambda e: e.t)
    return EventSet(events=events, method=method, agreement_ms=agreement, note=note)


# --------------------------------------------------------------------------
# Zeni coordinate method
# --------------------------------------------------------------------------
def _anterior(series: PixelSeries, side: str, joint: str, direction: int) -> np.ndarray:
    """Landmark position along the anterior axis, relative to the pelvis."""
    point_x = series.point(SIDE_LANDMARKS[side][joint])[:, 0]
    hip_x = series.midpoint(*HIPS)[:, 0]
    return direction * (point_x - hip_x)


def estimate_stride_period(signal: np.ndarray, fps: float, cfg: Config) -> float | None:
    """Dominant stride period from the autocorrelation of an anterior signal.

    A fixed minimum separation cannot serve both a brisk walker (0.9 s strides)
    and a slow one (1.8 s): set it low and one heel strike gets counted twice,
    set it high and every second strike is merged into a double-length stride.
    Estimating the period first and scaling the separation to it removes that
    trade-off, and costs one autocorrelation per limb.
    """
    min_s = float(cfg["segmentation.min_stride_time_s"])
    max_s = float(cfg["segmentation.max_stride_time_s"])

    values = np.asarray(signal, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return None

    # Detrend first, with a conservative window of one maximum stride. A slow
    # drift -- from a panning camera, or from the subject's pixel scale changing
    # as they approach -- adds a large low-frequency term whose autocorrelation
    # swamps the gait rhythm entirely, leaving no periodic peak to find. The
    # window is refined once the period is known; this pass only needs to remove
    # the trend, not to be optimal.
    values = _detrend(values, fps, max_s)
    centred = np.where(np.isfinite(values), values - np.nanmean(values), 0.0)

    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1:]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]

    low = max(1, int(round(min_s * fps)))
    high = min(int(round(max_s * fps)), correlation.size - 2)
    if high <= low:
        return None

    lag = low + int(np.argmax(correlation[low:high]))
    if correlation[lag] < 0.2:
        return None  # no consistent rhythm to lock onto
    return float(refine_extremum(correlation, lag)) / fps


def _detrend(values: np.ndarray, fps: float, period_s: float) -> np.ndarray:
    """Remove slow drift over a window longer than one stride.

    Zeni's rule assumes the pelvis-relative foot signal is stationary. It is not
    when the camera pans, when the subject's distance from the camera changes
    (so pixel scale drifts), or when the pelvis landmark wanders. That drift
    makes successive peaks unequal in height, and prominence gating then keeps
    some and discards others -- which is what produces alternate-strike
    detection and doubled stride times.
    """
    window = int(round(1.5 * period_s * fps)) | 1
    if window < 5 or window >= values.size:
        return values
    filled = np.where(np.isfinite(values), values, np.nanmean(values))
    padded = np.pad(filled, window // 2, mode="edge")
    baseline = np.convolve(padded, np.ones(window) / window, mode="valid")[: values.size]
    return values - baseline


def _zeni_events(
    window: PixelSeries, walk_pass: WalkPass, cfg: Config
) -> list[GaitEvent]:
    min_stride = float(cfg["segmentation.min_stride_time_s"])
    prominence = float(cfg["segmentation.peak_prominence_frac"])
    refine = bool(cfg["segmentation.subframe_refine"])
    events: list[GaitEvent] = []

    for side in SIDES:
        heel = _anterior(window, side, "heel", walk_pass.direction)
        toe = _anterior(window, side, "foot_index", walk_pass.direction)

        period = estimate_stride_period(heel, window.fps, cfg)
        if period is None:
            period = estimate_stride_period(toe, window.fps, cfg)
        separation = max(min_stride, 0.6 * period) if period else min_stride

        for signal, kind, extremum in (
            (heel, "heel_strike", "max"),
            (toe, "toe_off", "min"),
        ):
            search = _detrend(signal, window.fps, period) if period else signal
            indices = find_extrema(
                search, fps=window.fps, kind=extremum,
                min_separation_s=separation, prominence_frac=prominence,
            )
            for index in indices:
                events.append(
                    _make_event(window, walk_pass, search, index, kind, side,
                                refine=refine, method="zeni")
                )
    return events


# --------------------------------------------------------------------------
# Ankle-velocity cross-check
# --------------------------------------------------------------------------
def _ankle_velocity_events(
    window: PixelSeries, walk_pass: WalkPass, cfg: Config
) -> list[GaitEvent]:
    """Heel strike at ankle vertical minima; kept as a cross-check only."""
    min_stride = float(cfg["segmentation.min_stride_time_s"])
    prominence = float(cfg["segmentation.peak_prominence_frac"])
    refine = bool(cfg["segmentation.subframe_refine"])
    events: list[GaitEvent] = []

    for side in SIDES:
        ankle_y = window.point(SIDE_LANDMARKS[side]["ankle"])[:, 1]
        indices = find_extrema(
            ankle_y, fps=window.fps, kind="min",
            min_separation_s=min_stride, prominence_frac=prominence,
        )
        for index in indices:
            events.append(
                _make_event(window, walk_pass, ankle_y, index, "heel_strike", side,
                            refine=refine, method="ankle_velocity")
            )
    return events


def _make_event(
    window: PixelSeries, walk_pass: WalkPass, signal: np.ndarray, index: int,
    kind: str, side: str, *, refine: bool, method: str,
) -> GaitEvent:
    fractional = refine_extremum(signal, int(index)) if refine else float(index)
    t = fractional_time(window.t, fractional)

    # Confidence reflects whether the landmarks were genuinely observed around
    # the event, rather than interpolated across a dropout.
    low = max(0, int(index) - 2)
    high = min(window.n_frames, int(index) + 3)
    indices = [int(SIDE_LANDMARKS[side][j]) for j in ("heel", "ankle", "foot_index")]
    confidence = float(np.mean(window.valid[low:high, indices]))

    return GaitEvent(
        t=float(t), kind=kind, side=side,
        frame=walk_pass.start_frame + int(round(fractional)),
        confidence=confidence, method=method,
    )


def _cross_check(
    window: PixelSeries, walk_pass: WalkPass, cfg: Config,
    events: list[GaitEvent], method: str,
) -> tuple[float | None, str | None]:
    """Compare the primary detector against the configured fallback.

    Large disagreement usually means the foot landmarks are unreliable in this
    recording, which is worth surfacing before anyone reads the numbers.
    """
    fallback = str(cfg.get("segmentation.fallback_method") or "")
    if not fallback or fallback == method:
        return None, None

    try:
        other = (
            _ankle_velocity_events(window, walk_pass, cfg)
            if fallback == "ankle_velocity"
            else _zeni_events(window, walk_pass, cfg)
        )
    except Exception:  # noqa: BLE001 - a cross-check must never break the run
        return None, None

    offsets = []
    for side in SIDES:
        primary = sorted(e.t for e in events if e.kind == "heel_strike" and e.side == side)
        secondary = sorted(e.t for e in other if e.kind == "heel_strike" and e.side == side)
        if not primary or not secondary:
            continue
        for t in primary:
            offsets.append(min(abs(t - s) for s in secondary))

    if not offsets:
        return None, (
            f"the {fallback} cross-check found no comparable events; foot landmark "
            "quality may be poor"
        )

    # A *systematic* offset between the two methods is expected, not a fault:
    # ankle vertical minimum marks foot-flat, which genuinely occurs some tens of
    # milliseconds after initial contact. What indicates trouble is an offset
    # that varies from step to step, since that means at least one detector is
    # not locked to the gait rhythm.
    offsets = np.asarray(offsets)
    median_ms = float(np.median(offsets) * 1000)
    spread_ms = float(np.subtract(*np.percentile(offsets, [75, 25])) * 1000)

    # Scale the tolerance to the stride rather than fixing it in milliseconds.
    # The foot-flat instant genuinely varies with walking speed and surface, so
    # a slow walker's offsets spread wider in absolute terms without anything
    # being wrong. A fixed threshold fires on every normal recording, and a
    # warning that always fires carries no information.
    strikes = sorted(e.t for e in events if e.kind == "heel_strike")
    intervals = np.diff(strikes) if len(strikes) > 2 else np.array([])
    stride_ms = float(2 * np.median(intervals) * 1000) if intervals.size else 1100.0
    tolerance_ms = 0.15 * stride_ms

    note = None
    if spread_ms > tolerance_ms:
        note = (
            f"the two event-detection methods disagree inconsistently from step to "
            f"step (inter-quartile spread {spread_ms:.0f} ms around a "
            f"{median_ms:.0f} ms median offset); event timing in this recording is "
            "unstable, so treat the timing metrics with caution"
        )
    return median_ms, note


def independent_cadence_spm(series: PixelSeries, cfg: Config) -> float | None:
    """A second cadence estimate that shares no machinery with Zeni detection.

    The horizontal distance between the two ankles peaks once per step, when the
    legs are maximally split at contact. Counting those peaks gives cadence
    without pelvis-relative signals, without autocorrelation, and -- because it
    uses the *absolute* difference -- without depending on left and right being
    correctly distinguished.

    That independence is the point. It cannot replace event detection (it yields
    no per-limb events, so no stride times, asymmetry or double support), but
    when it disagrees with the main pipeline the main pipeline is wrong, and that
    is a failure which is otherwise invisible because every individual metric
    still looks plausible.
    """
    left = series.point(SIDE_LANDMARKS["left"]["ankle"])[:, 0]
    right = series.point(SIDE_LANDMARKS["right"]["ankle"])[:, 0]
    separation = np.abs(left - right)
    finite = np.isfinite(separation)
    if finite.sum() < int(2 * series.fps):
        return None

    separation = np.where(finite, separation, 0.0)
    span = float(np.ptp(separation[finite]))
    if span <= 0:
        return None

    # Half the minimum stride is the shortest physically plausible step.
    min_step_s = 0.5 * float(cfg["segmentation.min_stride_time_s"])
    peaks = find_extrema(
        separation, fps=series.fps, kind="max",
        min_separation_s=min_step_s, prominence_frac=0.25,
    )
    if len(peaks) < 3:
        return None

    duration = float(series.t[-1] - series.t[0])
    if duration <= 0:
        return None
    return float(60.0 * len(peaks) / duration)
