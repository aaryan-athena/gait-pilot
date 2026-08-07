"""Temporal and spatial gait measures derived from events and cycles.

Everything here except gait speed is calibration-free, which matters: in pilot
use most recordings will have no metric calibration, and timing metrics remain
fully valid without one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..pose.schema import HIPS, PL, SIDE_LANDMARKS

PL_LEFT_ANKLE, PL_RIGHT_ANKLE = PL.LEFT_ANKLE, PL.RIGHT_ANKLE
from ..pose.to_pixels import leg_length_px
from ..types import GaitCycle, GaitEvent, PixelSeries, WalkPass


@dataclass
class StepMeasures:
    """Per-step spatial and temporal measures, kept per side for asymmetry."""

    step_times_s: dict[str, list[float]] = field(
        default_factory=lambda: {"left": [], "right": []}
    )
    step_lengths_px: dict[str, list[float]] = field(
        default_factory=lambda: {"left": [], "right": []}
    )
    leg_length_px: float = float("nan")

    def all_step_times(self) -> np.ndarray:
        return np.array(self.step_times_s["left"] + self.step_times_s["right"])

    def mean_per_side(self, values: dict[str, list[float]]) -> dict[str, float]:
        return {
            side: float(np.mean(v)) if v else float("nan")
            for side, v in values.items()
        }


def stride_times(cycles: list[GaitCycle]) -> dict[str, np.ndarray]:
    """Valid stride durations, per side."""
    return {
        side: np.array([c.duration_s for c in cycles if c.valid and c.side == side])
        for side in ("left", "right")
    }


def cadence_spm(events: list[GaitEvent], cycles: list[GaitCycle]) -> float | None:
    """Steps per minute, from the mean interval between successive heel strikes.

    Derived from step intervals rather than (step count / video duration): the
    recording usually contains standing still at either end, and dividing by the
    full duration would understate cadence by however long the person hesitated.
    """
    strikes = sorted(
        e.t for e in events
        if e.kind == "heel_strike" and _within_valid_cycle(e.t, cycles)
    )
    if len(strikes) < 3:
        return None
    intervals = np.diff(strikes)
    # Guard against a missed contralateral strike doubling an interval.
    median = np.median(intervals)
    kept = intervals[(intervals > 0.5 * median) & (intervals < 1.8 * median)]
    if kept.size == 0:
        return None
    return float(60.0 / kept.mean())


def double_support_pct(cycles: list[GaitCycle]) -> tuple[float | None, int]:
    """Percentage of the gait cycle with both feet on the ground.

    Requires toe-off events, which is why heel-strike-only segmentation is not
    sufficient. Each stride has two double-support phases: from this foot's heel
    strike until the other foot lifts, and from the other foot's heel strike
    until this foot lifts.
    """
    values = []
    for cycle in cycles:
        if not cycle.valid:
            continue
        if cycle.contra_toe_off_t is None or cycle.contra_heel_strike_t is None:
            continue
        if cycle.ipsi_toe_off_t is None:
            continue
        first = cycle.contra_toe_off_t - cycle.t_start
        second = cycle.ipsi_toe_off_t - cycle.contra_heel_strike_t
        if first < 0 or second < 0:
            continue
        fraction = 100.0 * (first + second) / cycle.duration_s
        # Physiologically double support is roughly 15-35% of the cycle; well
        # outside that means an event was misassigned.
        if 2.0 < fraction < 60.0:
            values.append(fraction)

    if not values:
        return None, 0
    return float(np.median(values)), len(values)


def step_measures(
    series: PixelSeries,
    events: list[GaitEvent],
    cycles: list[GaitCycle],
    passes: list[WalkPass],
) -> StepMeasures:
    """Step time and step length for each step, attributed to the leading foot.

    Step length is the anterior separation between the two heels at the moment of
    heel strike, measured in pixels. It is converted to metres only where a valid
    calibration exists; the left/right *ratio* needs no calibration at all.
    """
    measures = StepMeasures(leg_length_px=leg_length_px(series))
    strikes = sorted(
        (e for e in events if e.kind == "heel_strike" and _within_valid_cycle(e.t, cycles)),
        key=lambda e: e.t,
    )

    for previous, current in zip(strikes, strikes[1:]):
        if previous.side == current.side:
            continue  # a contralateral strike was missed; this is not one step
        measures.step_times_s[current.side].append(current.t - previous.t)

        direction = _direction_at(current.t, series, passes)
        length = _heel_separation_px(series, current.t, current.side, direction)
        if length is not None and np.isfinite(length):
            measures.step_lengths_px[current.side].append(abs(length))

    return measures


def gait_speed_mps(
    series: PixelSeries,
    passes: list[WalkPass],
    cfg: Config,
    calibration,
) -> tuple[float | None, str | None]:
    """Mean walking speed over the steady-state portion of each pass.

    The leading and trailing fraction of every pass is trimmed
    (``speed.steady_state_trim_frac``) because those contain the subject starting
    and stopping, which would drag the mean down and mimic slow walking.

    Which body point is tracked depends on the calibration. A two-point scale is
    a single number, so the pelvis is used -- stable and well tracked. A
    homography maps the *floor plane* specifically, so it is applied to the ankle
    midpoint, which sits on that plane; projecting the pelvis through a floor
    homography would place it wherever the floor plane happens to intersect its
    line of sight, which is not where the person is.
    """
    trim = float(cfg["speed.steady_state_trim_frac"])
    use_homography = getattr(calibration, "method", None) == "homography"
    track = (
        series.midpoint(PL_LEFT_ANKLE, PL_RIGHT_ANKLE) if use_homography
        else series.midpoint(*HIPS)
    )
    speeds, weights = [], []

    for walk_pass in passes:
        margin = int(round(trim * walk_pass.n_frames))
        start = walk_pass.start_frame + margin
        stop = walk_pass.end_frame - margin
        if stop - start < max(4, int(0.5 * series.fps)):
            continue

        window = track[start:stop]
        t = series.t[start:stop]
        finite = np.isfinite(window).all(axis=1)
        if finite.sum() < 4:
            continue

        first, last = window[finite][0], window[finite][-1]
        elapsed = t[finite][-1] - t[finite][0]
        if elapsed <= 0:
            continue

        if use_homography:
            height = series.video.height
            metres = calibration.distance_m(
                (first[0], height - first[1]), (last[0], height - last[1])
            )
        else:
            metres = abs(last[0] - first[0]) * calibration.scale_m_per_px

        speeds.append(metres / elapsed)
        weights.append(elapsed)

    if not speeds:
        return None, (
            "no pass was long enough to measure steady-state speed after trimming "
            "the acceleration and deceleration phases"
        )
    return float(np.average(speeds, weights=weights)), None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _within_valid_cycle(t: float, cycles: list[GaitCycle]) -> bool:
    return any(c.valid and c.t_start <= t <= c.t_end for c in cycles)


def _direction_at(t: float, series: PixelSeries, passes: list[WalkPass]) -> int:
    frame = int(np.searchsorted(series.t, t))
    for walk_pass in passes:
        if walk_pass.start_frame <= frame < walk_pass.end_frame:
            return walk_pass.direction
    return passes[0].direction if passes else 1


def _heel_separation_px(
    series: PixelSeries, t: float, leading_side: str, direction: int
) -> float | None:
    """Anterior distance between the two heels at a heel strike."""
    frame = int(np.searchsorted(series.t, t))
    if frame >= series.n_frames:
        frame = series.n_frames - 1
    trailing = "right" if leading_side == "left" else "left"

    lead_x = series.point(SIDE_LANDMARKS[leading_side]["heel"])[frame, 0]
    trail_x = series.point(SIDE_LANDMARKS[trailing]["heel"])[frame, 0]
    if not np.isfinite([lead_x, trail_x]).all():
        return None
    return direction * (lead_x - trail_x)
