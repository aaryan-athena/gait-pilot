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
from ..pose.to_pixels import leg_length_px
from ..signal.events import (find_extrema, fractional_time, refine_extremum,
                             refine_zero_crossing)
from ..signal.resample import _true_runs
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
    prefer_contact = str(cfg["segmentation.toe_off_method"]) == "foot_contact"
    events: list[GaitEvent] = []
    periods: dict[str, float | None] = {}

    for side in SIDES:
        heel = _anterior(window, side, "heel", walk_pass.direction)
        toe = _anterior(window, side, "foot_index", walk_pass.direction)

        period = estimate_stride_period(heel, window.fps, cfg)
        if period is None:
            period = estimate_stride_period(toe, window.fps, cfg)
        periods[side] = period
        separation = max(min_stride, 0.6 * period) if period else min_stride

        # Heel strike keeps the Zeni rule. The heel's anterior maximum is a
        # sharp, well-conditioned marker of initial contact, and every measure
        # built on it -- stride time, variability, cadence, asymmetry -- has been
        # validated against it.
        search = _detrend(heel, window.fps, period) if period else heel
        for index in find_extrema(
            search, fps=window.fps, kind="max",
            min_separation_s=separation, prominence_frac=prominence,
        ):
            events.append(
                _make_event(window, walk_pass, search, index, "heel_strike", side,
                            refine=refine, method="zeni")
            )

    # Toe-off is decided for the pass as a whole, never per leg. Double support
    # is the overlap between the two legs' contacts, so measuring one leg from
    # ground contact and the other from an anterior minimum -- two definitions
    # that sit several frames apart -- would put a spurious asymmetry straight
    # into the result.
    contact = (
        {side: _contact_toe_offs(window, side, periods[side], cfg) for side in SIDES}
        if prefer_contact else {side: None for side in SIDES}
    )
    if all(value is not None for value in contact.values()):
        for value in contact.values():
            events.extend(value)
        return events

    # Fall back to the anterior minimum for both legs -- see
    # :func:`_contact_toe_offs` for when ground contact cannot be identified.
    for side in SIDES:
        period = periods[side]
        separation = max(min_stride, 0.6 * period) if period else min_stride
        toe = _anterior(window, side, "foot_index", walk_pass.direction)
        search = _detrend(toe, window.fps, period) if period else toe
        for index in find_extrema(
            search, fps=window.fps, kind="min",
            min_separation_s=separation, prominence_frac=prominence,
        ):
            events.append(
                _make_event(window, walk_pass, search, index, "toe_off", side,
                            refine=refine, method="zeni")
            )
    return events


def _foot_speed(
    window: PixelSeries, side: str, period: float | None
) -> np.ndarray | None:
    """Foot speed in leg lengths per stride.

    Dividing by leg length removes how big the subject is in frame; multiplying
    by the stride period removes how fast they walk. A planted foot then reads
    near zero and a swinging foot around three, for any subject at any pace, so
    a single fixed threshold serves every recording.
    """
    if not period or period <= 0:
        return None
    leg = leg_length_px(window)
    if not np.isfinite(leg) or leg <= 0:
        return None

    foot = 0.5 * (window.joint(side, "heel") + window.joint(side, "foot_index"))
    if int(np.isfinite(foot).all(axis=1).sum()) < 10:
        return None
    speed = np.linalg.norm(np.gradient(foot, axis=0), axis=1) * window.fps
    return speed * period / leg


def _contact_toe_offs(
    window: PixelSeries, side: str, period: float | None, cfg: Config
) -> list[GaitEvent] | None:
    """Toe-off as the end of the foot's ground contact, measured from foot speed.

    The anterior-minimum rule is badly conditioned for toe-off, and the raw
    signal shows why. While the foot is planted the pelvis keeps travelling over
    it, so the foot's *pelvis-relative* anterior position slides steadily
    backwards for the whole of stance with no distinct minimum: on pilot footage
    it fell from +47 px to -28 px across forty frames of a foot that was
    demonstrably stationary. The minimum therefore lands wherever noise and the
    pelvis estimate happen to put it, and the error grows as the walk speeds up
    and the plateau shortens.

    Ground contact itself is unambiguous in a fixed-camera recording: a planted
    foot barely moves in the image -- 2-10 px/s against 300-450 px/s in swing --
    so taking toe-off as the end of that stationary interval measures the event
    instead of inferring it from a shape.

    Returns ``None`` when the stationary assumption does not hold, so the caller
    falls back. That is judged from the signal rather than from metadata: on
    treadmill, in-place or camera-tracked footage a planted foot still travels
    across the image, and the detected contacts then occupy an implausible share
    of the cycle.
    """
    speed = _foot_speed(window, side, period)
    if speed is None:
        return None

    section = cfg.section("segmentation")
    threshold = float(section["stance_speed_threshold"])
    min_fraction = float(section["min_stance_fraction"])
    max_fraction = float(section["max_stance_fraction"])

    stance = np.isfinite(speed) & (speed < threshold)

    # Close brief gaps before measuring runs. A planted foot is not perfectly
    # still -- it rolls from heel to toe, and the heel lifts before the toe
    # does -- so the speed flickers above the threshold mid-contact and splits
    # one contact into several. Without this the run count swings between 0 and
    # 5 on identical footage purely from where the flickers land.
    gap = max(1, int(round(float(section["stance_merge_gap_fraction"])
                           * period * window.fps)))
    for start, stop in _true_runs(~stance):
        if (stop - start) <= gap and start > 0 and stop < stance.size:
            stance[start:stop] = True

    minimum_frames = max(1, int(round(
        float(section["min_contact_fraction"]) * period * window.fps)))
    runs = [(a, b) for a, b in _true_runs(stance) if (b - a) >= minimum_frames]
    if len(runs) < 2:
        return None

    # Self-validation: stance occupies roughly 60% of the cycle in any real
    # walk, more when walking slowly. Far outside that band means the foot was
    # never actually stationary -- a moving camera, a treadmill -- and this
    # measurement does not apply to the recording.
    fractions = [(b - a) / (period * window.fps) for a, b in runs]
    if not min_fraction <= float(np.median(fractions)) <= max_fraction:
        return None

    foot_indices = [int(SIDE_LANDMARKS[side][joint])
                    for joint in ("heel", "foot_index")]
    events: list[GaitEvent] = []
    for _, end in runs:
        if end >= speed.size:
            continue
        # Speed crosses the threshold steeply as the foot leaves the ground, so
        # the crossing locates toe-off far more precisely than a frame index.
        fractional = refine_zero_crossing(speed - threshold, end - 1)
        events.append(
            GaitEvent(
                t=float(fractional_time(window.t, fractional)),
                kind="toe_off",
                side=side,
                frame=window.n_frames and int(round(fractional)),
                confidence=float(np.mean(
                    window.valid[max(0, end - 2):end + 2, foot_indices]
                )),
                method="foot_contact",
            )
        )
    return events or None


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
