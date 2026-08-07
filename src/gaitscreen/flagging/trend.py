"""Personal trend checks: has this person changed relative to their own normal?

Three properties distinguish this from a plain "beyond 2 SD" rule.

**Direction awareness.** Only deterioration flags. A person walking faster or more
consistently than their baseline has not developed a fall risk, and flagging it
would add noise to a list that a caregiver has to triage.

**A minimum-detectable-change floor.** A deviation must exceed both the SD
multiple *and* the tool's own test-retest error for that metric. Without this,
a person whose baseline happens to be very consistent gets flagged on
measurement noise -- and the tighter their baseline, the more often it happens,
which is exactly backwards. The MDC values shipped in the config are placeholders
and should be replaced with values measured on the actual capture setup.

**Confirmation, without losing sensitivity.** The brief asks to bias toward
sensitivity, so a single deviating session still raises a flag. A deviation
present in 2 of the last 3 sessions is additionally marked ``confirmed``, which
lets a caregiver triage a long list without any flag being suppressed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..types import DETERIORATION_DIRECTION, Flag
from .absolute import METRIC_LABELS
from .baseline import Baseline


@dataclass
class Deviation:
    metric: str
    value: float
    baseline_centre: float
    baseline_spread: float
    change: float  # signed, value - centre
    n_sds: float
    exceeds_sds: bool
    exceeds_mdc: bool
    deteriorating: bool

    @property
    def triggered(self) -> bool:
        return self.deteriorating and self.exceeds_sds and self.exceeds_mdc


def assess(value: float, baseline: Baseline, cfg: Config) -> Optional[Deviation]:
    """Measure how far a session sits from a person's own normal."""
    if not baseline.available:
        return None

    section = cfg.section("flagging.trend")
    threshold_sds = float(section["deviation_sds"])
    mdc = float(cfg.get(
        f"flagging.trend.minimum_detectable_change.{baseline.metric}", 0.0
    ) or 0.0)

    change = float(value - baseline.centre)
    worsening_sign = DETERIORATION_DIRECTION.get(baseline.metric, 1)
    deteriorating = (change * worsening_sign) > 0

    n_sds = (
        abs(change) / baseline.spread if baseline.spread > 0
        else (float("inf") if abs(change) > 0 else 0.0)
    )

    return Deviation(
        metric=baseline.metric,
        value=float(value),
        baseline_centre=baseline.centre,
        baseline_spread=baseline.spread,
        change=change,
        n_sds=float(n_sds),
        exceeds_sds=bool(n_sds >= threshold_sds),
        exceeds_mdc=bool(abs(change) >= mdc),
        deteriorating=bool(deteriorating),
    )


def to_flag(
    deviation: Deviation, baseline: Baseline, cfg: Config, *, confirmed: bool = False
) -> Flag:
    """Turn a triggered deviation into a caregiver-readable flag."""
    label, unit = METRIC_LABELS.get(deviation.metric, (deviation.metric, ""))
    space = " " if unit else ""
    worsened = "decreased" if deviation.change < 0 else "increased"
    severity = "high" if deviation.n_sds >= 2.0 * float(
        cfg["flagging.trend.deviation_sds"]
    ) else "moderate"

    spread_text = (
        f"{deviation.n_sds:.1f} times their usual variation"
        if np.isfinite(deviation.n_sds)
        else "a change from a previously unvarying measurement"
    )
    confirm_text = (
        " This has now been seen in more than one recent session."
        if confirmed else ""
    )

    return Flag(
        code=f"{deviation.metric}_personal_trend",
        metric=deviation.metric,
        severity=severity,  # type: ignore[arg-type]
        trigger="trend",
        message=(
            f"{label} has {worsened} from this person's usual "
            f"{deviation.baseline_centre:.2f}{space}{unit} to "
            f"{deviation.value:.2f}{space}{unit} -- {spread_text}, based on "
            f"{baseline.n_sessions} previous sessions.{confirm_text}"
        ),
        detail={
            "value": deviation.value,
            "baseline_centre": deviation.baseline_centre,
            "baseline_spread": deviation.baseline_spread,
            "change": deviation.change,
            "n_sds": deviation.n_sds,
            "baseline_sessions": baseline.n_sessions,
            "baseline_method": baseline.method,
        },
        confirmed=confirmed,
    )


def count_recent_deviations(
    history: pd.DataFrame, baseline: Baseline, cfg: Config
) -> int:
    """How many of the most recent prior sessions also deviated.

    Used only to mark a flag ``confirmed``; it never suppresses one.
    """
    section = cfg.section("flagging.trend")
    window = int(section["confirm_window"])
    metric = baseline.metric
    if history.empty or metric not in history:
        return 0

    recent = history.sort_values("session_date").tail(max(window - 1, 0))
    hits = 0
    for value in recent[metric].dropna().astype(float):
        deviation = assess(value, baseline, cfg)
        if deviation is not None and deviation.triggered:
            hits += 1
    return hits
