"""A walk signature: what about a walk says *who* it is, not how well they walk.

Two kinds of trait, weighted very differently by the matcher:

* **Body proportions** -- torso against leg, thigh against shin, upper arm
  against forearm, head against torso, and (facing the camera) shoulders
  against hips. Ratios of segment lengths in the same frame, so they do not
  depend on how far away the camera is or how it is calibrated, and they do
  not change as a person ages over months. These carry the identification.
* **Gait traits** -- cadence and speed. Distinctive between people, but they
  are exactly what this tool monitors: a person whose walking declines must
  not stop being recognised as themselves. They only break near-ties.

A signature is only compared with others recorded from the same side of the
walker (side-on with side-on), for the same reason trends are: a segment seen
face-on is foreshortened differently from one seen side-on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import Config
from ..pose.schema import PL
from ..types import PixelSeries, SessionMetrics

SIGNATURE_VERSION = 1

BODY_TRAITS = ("torso_to_leg", "thigh_to_shank", "upper_arm_to_forearm",
               "head_to_torso", "shoulder_to_hip_width")
GAIT_TRAITS = ("cadence_spm", "gait_speed_mps")

_SIDES = {
    "left": (PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST, PL.LEFT_HIP,
             PL.LEFT_KNEE, PL.LEFT_ANKLE),
    "right": (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST, PL.RIGHT_HIP,
              PL.RIGHT_KNEE, PL.RIGHT_ANKLE),
}


def walk_signature(series: PixelSeries, metrics: Optional[SessionMetrics],
                   view_kind: Optional[str], cfg: Config) -> dict:
    """The signature of one walk, as plain JSON-ready data.

    A trait the recording cannot support (too few clear frames of the joints
    it needs) is ``None`` rather than a guess; the matcher skips it.
    """
    min_frames = int(cfg.get("identity.walk.min_frames", 30))
    min_vis = float(cfg.get("identity.walk.min_visibility", 0.6))

    def length(a: PL, b: PL, *, extended: bool) -> Optional[float]:
        ok = (series.valid[:, a] & series.valid[:, b]
              & (series.visibility[:, a] >= min_vis) & (series.visibility[:, b] >= min_vis))
        if int(ok.sum()) < min_frames:
            return None
        d = np.linalg.norm(series.xy[ok, a] - series.xy[ok, b], axis=1)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size < min_frames:
            return None
        # A limb is only its full length when straight; the upper percentile
        # finds the extended frames without trusting a single outlier.
        return float(np.percentile(d, 90 if extended else 50))

    def per_side_ratio(num, den) -> Optional[float]:
        ratios = []
        for side in _SIDES.values():
            top = length(*num(side), extended=True)
            bottom = length(*den(side), extended=True)
            if top and bottom:
                ratios.append(top / bottom)
        return float(np.mean(ratios)) if ratios else None

    sh, el, wr, hip, kn, an = range(6)
    body = {
        "thigh_to_shank": per_side_ratio(lambda s: (s[hip], s[kn]), lambda s: (s[kn], s[an])),
        "upper_arm_to_forearm": per_side_ratio(lambda s: (s[sh], s[el]),
                                               lambda s: (s[el], s[wr])),
        "torso_to_leg": per_side_ratio(lambda s: (s[sh], s[hip]), lambda s: (s[hip], s[an])),
        "head_to_torso": _head_to_torso(series, min_vis, min_frames),
        "shoulder_to_hip_width": None,
    }
    if view_kind == "coronal":
        # Shoulder and hip width are only visible facing the camera; side-on
        # they collapse to whatever the tracker guesses for the hidden side.
        shoulders = length(PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, extended=False)
        hips = length(PL.LEFT_HIP, PL.RIGHT_HIP, extended=False)
        if shoulders and hips:
            body["shoulder_to_hip_width"] = shoulders / hips

    gait = {name: _finite(getattr(metrics, name, None)) for name in GAIT_TRAITS}
    return {
        "version": SIGNATURE_VERSION,
        "view": view_kind,
        "body": {k: _round(v) for k, v in body.items()},
        "gait": {k: _round(v) for k, v in gait.items()},
        "n_frames": int(series.n_frames),
    }


def _head_to_torso(series: PixelSeries, min_vis: float, min_frames: int) -> Optional[float]:
    """Nose to mid-shoulder over mid-shoulder to mid-hip, per frame."""
    need = [PL.NOSE, PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, PL.LEFT_HIP, PL.RIGHT_HIP]
    ok = np.all(series.valid[:, need] & (series.visibility[:, need] >= min_vis), axis=1)
    if int(ok.sum()) < min_frames:
        return None
    shoulders = series.midpoint(PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER)[ok]
    hips = series.midpoint(PL.LEFT_HIP, PL.RIGHT_HIP)[ok]
    head = np.linalg.norm(series.point(PL.NOSE)[ok] - shoulders, axis=1)
    torso = np.linalg.norm(shoulders - hips, axis=1)
    keep = torso > 0
    return float(np.median(head[keep] / torso[keep])) if keep.any() else None


def signature_from_result(result, cfg: Config) -> dict:
    """The signature of an analysed :class:`~gaitscreen.pipeline.SessionResult`."""
    view = result.analysis.view or result.extraction.view
    return walk_signature(result.extraction.series, result.analysis.metrics,
                          view.kind if view is not None else None, cfg)


def _finite(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)
