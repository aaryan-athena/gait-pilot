"""Stride-time variability.

Per the geriatric literature this is the single most fall-risk-predictive of the
metrics here, so it gets the most defensive treatment in the codebase.

Two gates protect it, both of which return ``None`` with a reason rather than a
number:

* **Stride count.** A CV from 4-8 strides is dominated by sampling error; the
  literature generally wants 12-30. Below ``features.min_strides_for_cv`` the
  value is withheld.
* **Frame rate.** At 25-30 fps the frame interval (33-40 ms) is comparable to the
  standard deviation being measured (25-35 ms). Sub-frame event refinement
  recovers most of that, but not all, so below ``video.fps_variability_min`` the
  value is returned marked low-confidence.

The CV is computed per leg and then averaged, rather than pooling all strides.
Pooling would fold any left/right *difference* in mean stride time into the
spread, inflating the variability of an asymmetric but perfectly consistent
walker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..types import GaitCycle
from .spatiotemporal import stride_times


@dataclass
class VariabilityResult:
    cv_pct: float | None
    mean_s: float | None
    sd_s: float | None
    n_strides: int
    per_side_cv_pct: dict[str, float | None]
    unavailable_reason: str | None = None
    low_confidence_reason: str | None = None


def stride_time_variability(
    cycles: list[GaitCycle], cfg: Config, fps: float
) -> VariabilityResult:
    """Coefficient of variation of stride time, with both gates applied."""
    per_side = stride_times(cycles)
    n_total = sum(v.size for v in per_side.values())
    minimum = int(cfg["features.min_strides_for_cv"])

    all_strides = np.concatenate([v for v in per_side.values() if v.size]) if n_total else np.array([])
    mean_s = float(all_strides.mean()) if all_strides.size else None
    sd_s = float(all_strides.std(ddof=1)) if all_strides.size > 1 else None

    side_cv: dict[str, float | None] = {}
    for side, values in per_side.items():
        side_cv[side] = (
            float(100.0 * values.std(ddof=1) / values.mean())
            if values.size > 1 and values.mean() > 0 else None
        )

    if n_total < minimum:
        return VariabilityResult(
            cv_pct=None, mean_s=mean_s, sd_s=sd_s, n_strides=n_total,
            per_side_cv_pct=side_cv,
            unavailable_reason=(
                f"only {n_total} valid strides were recovered; at least {minimum} "
                "are needed before a variability figure means anything. Record a "
                "longer walk, or several passes in one session."
            ),
        )

    usable = [v for v in side_cv.values() if v is not None]
    if not usable:
        return VariabilityResult(
            cv_pct=None, mean_s=mean_s, sd_s=sd_s, n_strides=n_total,
            per_side_cv_pct=side_cv,
            unavailable_reason="no side had more than one valid stride",
        )

    cv = float(np.mean(usable))
    low_confidence = None
    variability_min = float(cfg["video.fps_variability_min"])
    if fps < variability_min:
        frame_ms = 1000.0 / fps
        low_confidence = (
            f"recorded at {fps:g} fps, so event times are quantised at {frame_ms:.0f} ms "
            f"before sub-frame refinement -- comparable to the stride-time standard "
            f"deviation itself ({(sd_s or 0) * 1000:.0f} ms here). Treat this figure "
            f"as indicative and record at {variability_min:g} fps for a reliable one."
        )

    return VariabilityResult(
        cv_pct=cv, mean_s=mean_s, sd_s=sd_s, n_strides=n_total,
        per_side_cv_pct=side_cv, low_confidence_reason=low_confidence,
    )
