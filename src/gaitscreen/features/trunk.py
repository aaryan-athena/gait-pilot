"""Trunk sway.

The brief specifies lateral (side-to-side) trunk deviation, which requires a
frontal view. This deployment has no frontal camera, so the measure implemented
here is the sagittal substitute: **anterior-posterior trunk lean**, the horizontal
offset of the shoulder midpoint from the hip midpoint along the direction of
travel, normalised by leg length so it is comparable across subjects and camera
distances.

This is a different quantity from lateral sway, not a proxy for it, and it is
named ``trunk_ap_sway_norm`` rather than ``trunk_sway`` to stop anyone comparing
it against published lateral-sway norms. What it does capture is real and
relevant: increased forward lean and larger cycle-to-cycle trunk excursion both
appear in cautious and unstable gait.

Two numbers are produced. The *amplitude* is the within-cycle peak-to-peak
excursion, which is the sway analogue. The *mean lean* is the average offset,
which reflects posture rather than sway; it is stored separately so a stooped but
steady walker is not scored as unstable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose.schema import HIPS, SHOULDERS
from ..pose.to_pixels import leg_length_px
from ..types import GaitCycle, PixelSeries, WalkPass


@dataclass
class TrunkResult:
    ap_sway_norm: float | None  # within-cycle peak-to-peak, / leg length
    mean_lean_norm: float | None  # mean forward offset, / leg length
    n_cycles: int
    axis: str = "sagittal_ap"
    unavailable_reason: str | None = None
    note: str | None = None


AXIS_NOTE = (
    "no frontal camera in this setup, so lateral trunk sway cannot be measured; "
    "this is anterior-posterior trunk lean from the side view and is not "
    "comparable to published lateral-sway values"
)


def compute(
    series: PixelSeries,
    cycles: list[GaitCycle],
    passes: list[WalkPass],
) -> TrunkResult:
    """Per-cycle trunk lean excursion, summarised across the session."""
    leg = leg_length_px(series)
    if not np.isfinite(leg) or leg <= 0:
        return TrunkResult(
            ap_sway_norm=None, mean_lean_norm=None, n_cycles=0,
            unavailable_reason="leg length could not be measured, so trunk lean "
                               "cannot be normalised",
            note=AXIS_NOTE,
        )

    offset = series.midpoint(*SHOULDERS)[:, 0] - series.midpoint(*HIPS)[:, 0]

    amplitudes, leans = [], []
    for cycle in cycles:
        if not cycle.valid:
            continue
        direction = (
            passes[cycle.pass_index].direction
            if 0 <= cycle.pass_index < len(passes) else 1
        )
        start = int(np.searchsorted(series.t, cycle.t_start))
        stop = int(np.searchsorted(series.t, cycle.t_end)) + 1
        window = direction * offset[start:stop]
        window = window[np.isfinite(window)]
        if window.size < 4:
            continue
        amplitudes.append(float(np.ptp(window)) / leg)
        leans.append(float(np.mean(window)) / leg)

    if not amplitudes:
        return TrunkResult(
            ap_sway_norm=None, mean_lean_norm=None, n_cycles=0,
            unavailable_reason="no valid cycle had enough trunk landmark data",
            note=AXIS_NOTE,
        )

    return TrunkResult(
        ap_sway_norm=float(np.median(amplitudes)),
        mean_lean_norm=float(np.median(leans)),
        n_cycles=len(amplitudes),
        note=AXIS_NOTE,
    )
