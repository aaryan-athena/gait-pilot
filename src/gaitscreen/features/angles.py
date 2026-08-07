"""Sagittal joint angle curves for hip, knee and ankle.

Stored per cycle at 0-100% even though they are not part of the flagging logic
yet: they make the session report far more informative, and having them archived
means a future version can add angle-based rules to historical data without
re-processing video.

Sign conventions (clinical): positive hip flexion is the thigh forward of the
trunk, positive knee flexion is the shank behind the thigh, positive ankle angle
is dorsiflexion. All are resolved against the pass's direction of progression, so
they read the same whether the subject walks left-to-right or right-to-left.

Caveat worth carrying into any interpretation: these are 2D projections from a
single camera. Out-of-plane rotation and the far limb's occlusion both add error,
and ankle angle is the least reliable of the three because it depends on the foot
segment, whose landmarks are the noisiest.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..pose.schema import HIPS, SHOULDERS, SIDE_LANDMARKS
from ..signal.events import resample_to_percent
from ..types import AngleCurves, GaitCycle, PixelSeries, WalkPass

JOINTS = ("hip", "knee", "ankle")


def joint_angle_series(
    series: PixelSeries, side: str, direction: int
) -> dict[str, np.ndarray]:
    """Per-frame hip, knee and ankle angles in degrees."""
    marks = SIDE_LANDMARKS[side]
    hip = series.point(marks["hip"])
    knee = series.point(marks["knee"])
    ankle = series.point(marks["ankle"])
    heel = series.point(marks["heel"])
    toe = series.point(marks["foot_index"])
    shoulder_mid = series.midpoint(*SHOULDERS)
    hip_mid = series.midpoint(*HIPS)

    trunk_down = hip_mid - shoulder_mid
    thigh = knee - hip
    shank_down = ankle - knee
    shank_up = knee - ankle
    foot = toe - heel

    return {
        "hip": direction * _signed_angle(trunk_down, thigh),
        "knee": -direction * _signed_angle(thigh, shank_down),
        "ankle": 90.0 - _unsigned_angle(shank_up, foot),
    }


def cycle_curves(
    series: PixelSeries,
    cycles: list[GaitCycle],
    passes: list[WalkPass],
    cfg: Config,
) -> AngleCurves:
    """Resample each valid cycle's angle curves onto 0-100% of the cycle."""
    n_points = int(cfg["features.angle_curve_points"])
    percent = np.linspace(0.0, 100.0, n_points)
    collected: dict[str, list[np.ndarray]] = {
        f"{side}_{joint}": [] for side in ("left", "right") for joint in JOINTS
    }

    cached: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for cycle in cycles:
        if not cycle.valid:
            continue
        direction = (
            passes[cycle.pass_index].direction
            if 0 <= cycle.pass_index < len(passes) else 1
        )
        key = (cycle.side, direction)
        if key not in cached:
            cached[key] = joint_angle_series(series, cycle.side, direction)

        start = int(np.searchsorted(series.t, cycle.t_start))
        stop = int(np.searchsorted(series.t, cycle.t_end)) + 1
        if stop - start < 4:
            continue
        for joint in JOINTS:
            collected[f"{cycle.side}_{joint}"].append(
                resample_to_percent(cached[key][joint][start:stop], n_points)
            )

    curves = {
        name: (np.vstack(rows) if rows else np.empty((0, n_points)))
        for name, rows in collected.items()
    }
    return AngleCurves(percent=percent, curves=curves)


def range_of_motion(angles: AngleCurves) -> dict[str, float]:
    """Peak-to-peak excursion of each mean angle curve, in degrees.

    A useful summary on its own: reduced knee excursion is one of the more
    visible kinematic changes in cautious or antalgic gait.
    """
    out: dict[str, float] = {}
    for name, rows in angles.curves.items():
        if rows.size == 0:
            continue
        mean_curve = np.nanmean(rows, axis=0)
        if np.isfinite(mean_curve).any():
            out[name] = float(np.nanmax(mean_curve) - np.nanmin(mean_curve))
    return out


def _signed_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angle from ``a`` to ``b`` in degrees, counter-clockwise positive."""
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = np.einsum("ij,ij->i", a, b)
    return np.degrees(np.arctan2(cross, dot))


def _unsigned_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.einsum("ij,ij->i", a, b)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.clip(dot / norms, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))
