"""Building gait cycles from detected events, and deciding which to trust.

A cycle is one stride: heel strike to the next heel strike on the same foot.
Cycles are rejected rather than silently included when they cannot support a
measurement, and each rejection carries a reason so the app can explain why a
session yielded fewer strides than the video appears to contain.

Rejection rules, and why each exists:

* **Implausible duration** -- a stride outside 0.6-2.5 s is a missed or spurious
  detection, not a slow walker.
* **Low landmark confidence** -- events derived from interpolated landmarks have
  uncertain timing.
* **First and last stride of each pass** -- these contain the acceleration and
  deceleration phases. They are systematically different from steady-state
  strides, and including them inflates stride-time variability, which is the
  metric most sensitive to exactly this kind of contamination.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import GaitCycle, GaitEvent, PixelSeries, WalkPass
from .events import EventSet


def build_cycles(
    series: PixelSeries,
    walk_pass: WalkPass,
    pass_index: int,
    event_set: EventSet,
    cfg: Config,
) -> list[GaitCycle]:
    """Pair consecutive same-side heel strikes into strides and gate them."""
    min_stride = float(cfg["segmentation.min_stride_time_s"])
    max_stride = float(cfg["segmentation.max_stride_time_s"])
    edge_exclusion = int(cfg["segmentation.exclude_strides_at_pass_edges"])

    cycles: list[GaitCycle] = []
    for side in ("left", "right"):
        strikes = sorted(event_set.of("heel_strike", side), key=lambda e: e.t)
        for previous, current in zip(strikes, strikes[1:]):
            cycle = GaitCycle(
                side=side,
                t_start=previous.t,
                t_end=current.t,
                start_frame=previous.frame,
                end_frame=current.frame,
                pass_index=pass_index,
            )
            _attach_contralateral(cycle, event_set, side)
            _gate_duration(cycle, min_stride, max_stride)
            _gate_confidence(cycle, previous, current)
            cycles.append(cycle)

    cycles.sort(key=lambda c: c.t_start)
    _exclude_pass_edges(cycles, edge_exclusion)
    return cycles


def _attach_contralateral(cycle: GaitCycle, event_set: EventSet, side: str) -> None:
    """Record the opposite-limb events inside this stride.

    Double-support time is the interval when both feet are down, so it needs the
    contralateral toe-off and heel strike -- which heel-strike detection alone
    cannot provide. This is why toe-off detection is not optional.
    """
    other = "right" if side == "left" else "left"

    def first_between(kind: str, source_side: str) -> float | None:
        candidates = [
            e.t for e in event_set.of(kind, source_side)
            if cycle.t_start < e.t < cycle.t_end
        ]
        return min(candidates) if candidates else None

    cycle.contra_toe_off_t = first_between("toe_off", other)
    cycle.contra_heel_strike_t = first_between("heel_strike", other)
    cycle.ipsi_toe_off_t = first_between("toe_off", side)


def _gate_duration(cycle: GaitCycle, min_stride: float, max_stride: float) -> None:
    duration = cycle.duration_s
    if duration < min_stride:
        cycle.valid = False
        cycle.exclusion_reason = (
            f"stride of {duration:.2f}s is shorter than the {min_stride:.2f}s "
            "minimum, so it is probably a double-counted event"
        )
    elif duration > max_stride:
        cycle.valid = False
        cycle.exclusion_reason = (
            f"stride of {duration:.2f}s exceeds the {max_stride:.2f}s maximum, so "
            "an intervening heel strike was probably missed"
        )


def _gate_confidence(cycle: GaitCycle, start: GaitEvent, end: GaitEvent) -> None:
    if not cycle.valid:
        return
    confidence = min(start.confidence, end.confidence)
    if confidence < 0.5:
        cycle.valid = False
        cycle.exclusion_reason = (
            f"foot landmarks were observed in only {confidence:.0%} of the frames "
            "around this stride's heel strikes, so its timing is unreliable"
        )


def _exclude_pass_edges(cycles: list[GaitCycle], n_edge: int) -> None:
    """Drop the first and last strides of the pass, per side."""
    if n_edge <= 0:
        return
    for side in ("left", "right"):
        side_cycles = [c for c in cycles if c.side == side]
        if len(side_cycles) <= 2 * n_edge:
            # Too few strides to trim without discarding the whole pass; keep them
            # and let the minimum-stride gate decide whether the session counts.
            continue
        for cycle in side_cycles[:n_edge]:
            _exclude(cycle, "first stride of the pass (acceleration phase)")
        for cycle in side_cycles[-n_edge:]:
            _exclude(cycle, "last stride of the pass (deceleration phase)")


def _exclude(cycle: GaitCycle, reason: str) -> None:
    if cycle.valid:
        cycle.valid = False
        cycle.exclusion_reason = reason


def summarise(cycles: list[GaitCycle]) -> dict:
    """Counts and exclusion reasons, for display in the app."""
    valid = [c for c in cycles if c.valid]
    reasons: dict[str, int] = {}
    for cycle in cycles:
        if not cycle.valid and cycle.exclusion_reason:
            key = cycle.exclusion_reason.split(",")[0]
            reasons[key] = reasons.get(key, 0) + 1

    durations = np.array([c.duration_s for c in valid])
    return {
        "n_total": len(cycles),
        "n_valid": len(valid),
        "n_valid_left": sum(1 for c in valid if c.side == "left"),
        "n_valid_right": sum(1 for c in valid if c.side == "right"),
        "exclusions": reasons,
        "mean_duration_s": float(durations.mean()) if durations.size else None,
    }
