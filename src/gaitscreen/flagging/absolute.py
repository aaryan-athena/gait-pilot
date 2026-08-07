"""Absolute threshold checks against population-level cut-offs.

These are the "is this person slow in general" checks, as opposed to the personal
trend checks. They are useful on a first session, when no baseline exists yet.

**Every threshold here is illustrative.** They are drawn from general reading of
the geriatric gait literature (the 0.6 m/s and 1.0 m/s speed bands are the most
widely cited), and they are placed in ``config/default.yaml`` rather than in code
precisely so a clinician can review and change them without touching the
pipeline. Validate them against current sources and against your own population
before acting on them.
"""
from __future__ import annotations

from ..config import Config
from ..types import Flag, SessionMetrics

#: Human-readable metric names for flag messages.
METRIC_LABELS = {
    "gait_speed_mps": ("walking speed", "m/s"),
    "stride_time_cv_pct": ("stride-time variability", "%"),
    "step_length_asymmetry_pct": ("step-length asymmetry", "%"),
    "step_time_asymmetry_pct": ("step-time asymmetry", "%"),
    "cadence_spm": ("cadence", "steps/min"),
    "double_support_pct": ("double-support time", "% of cycle"),
    "trunk_ap_sway_norm": ("trunk lean excursion", ""),
}


def evaluate(metrics: SessionMetrics, cfg: Config) -> list[Flag]:
    """Compare each measured metric against its configured risk bands."""
    thresholds = cfg["flagging.absolute"]
    flags: list[Flag] = []

    for metric, bands in thresholds.items():
        value = metrics.value(metric)
        if value is None:
            continue
        label, unit = METRIC_LABELS.get(metric, (metric, ""))

        for severity in ("high", "moderate"):
            triggered, threshold, comparison = _check(bands, severity, value)
            if not triggered:
                continue
            flags.append(
                Flag(
                    code=f"{metric}_{severity}_absolute",
                    metric=metric,
                    severity=severity,  # type: ignore[arg-type]
                    trigger="absolute",
                    message=(
                        f"{label} is {value:.2f}{_space(unit)}{unit}, which is "
                        f"{comparison} the {severity}-concern threshold of "
                        f"{threshold:.2f}{_space(unit)}{unit}."
                    ),
                    detail={
                        "value": float(value),
                        "threshold": float(threshold),
                        "band": severity,
                        "note": "threshold is illustrative; review against current "
                                "clinical literature",
                    },
                )
            )
            break  # high-risk supersedes moderate for the same metric

    return flags


def _check(bands: dict, severity: str, value: float):
    below = bands.get(f"{severity}_risk_below")
    if below is not None and value < below:
        return True, below, "below"
    above = bands.get(f"{severity}_risk_above")
    if above is not None and value > above:
        return True, above, "above"
    return False, None, ""


def _space(unit: str) -> str:
    return " " if unit else ""
