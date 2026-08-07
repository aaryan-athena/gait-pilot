"""Left/right asymmetry of step length and step time.

Asymmetry is a ratio, so it needs no calibration -- which makes it the most
readily available spatial metric in an uncalibrated pilot deployment.

It is not, however, free of the single-camera problem. The far limb is occluded
behind the near limb through much of stance, so its landmarks are systematically
noisier, and that noise does not cancel in a ratio. Where the near limb is known
(``camera_side``), the result is annotated so the reader knows a one-sided view
produced it. The protocol fix is to record one pass in each direction, giving each
limb a turn as the near limb.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spatiotemporal import StepMeasures


@dataclass
class AsymmetryResult:
    step_length_pct: float | None
    step_time_pct: float | None
    step_length_per_side_px: dict[str, float]
    step_time_per_side_s: dict[str, float]
    n_steps: dict[str, int]
    unavailable_reason: str | None = None
    note: str | None = None


def symmetry_index(left: float, right: float) -> float | None:
    """Absolute difference as a percentage of the mean of the two sides.

    Sign is dropped deliberately: which side is longer is clinically interesting
    but not what the flagging logic acts on, and keeping it signed would let a
    left-dominant and a right-dominant session average to zero across a trend.
    """
    if not np.isfinite([left, right]).all():
        return None
    mean = 0.5 * (left + right)
    if mean <= 0:
        return None
    return float(100.0 * abs(left - right) / mean)


def compute(measures: StepMeasures, *, camera_side: str | None = None) -> AsymmetryResult:
    lengths = measures.mean_per_side(measures.step_lengths_px)
    times = measures.mean_per_side(measures.step_times_s)
    counts = {side: len(v) for side, v in measures.step_times_s.items()}

    result = AsymmetryResult(
        step_length_pct=symmetry_index(lengths["left"], lengths["right"]),
        step_time_pct=symmetry_index(times["left"], times["right"]),
        step_length_per_side_px=lengths,
        step_time_per_side_s=times,
        n_steps=counts,
    )

    # At least two steps per side, otherwise a single mis-detected step is the
    # entire measurement.
    if min(counts.values()) < 2:
        result.step_length_pct = None
        result.step_time_pct = None
        result.unavailable_reason = (
            f"only {counts['left']} left and {counts['right']} right steps were "
            "detected; at least two per side are needed to compare them"
        )
        return result

    if camera_side:
        result.note = (
            f"measured from a single view with the {camera_side} side nearer the "
            f"camera, so the {'right' if camera_side == 'left' else 'left'} limb is "
            "partly occluded and noisier. Record one pass in each direction to "
            "remove this bias."
        )
    return result
