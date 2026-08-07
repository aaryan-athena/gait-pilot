"""Personal baselines: what is normal *for this person*.

The brief asks for a rolling mean and SD from the first 3-5 sessions. Two
adjustments were needed to make that hold up.

**Robust statistics by default.** A standard deviation estimated from four
observations is extremely unstable, and a single poor recording in that window
sets the reference for everything after it. Median and 1.4826 x MAD (the constant
makes MAD comparable to an SD for normally distributed data) are used instead, so
one outlier cannot redefine normal. Configurable via ``flagging.baseline.robust``.

**Window by sessions as well as days.** A 90-day window assumes frequent
screening. At a realistic monthly cadence, 90 days is three sessions -- too few to
estimate a spread from at all. The window is therefore whichever of "last N days"
and "last M sessions" yields more data.

Low-confidence sessions are excluded from baselines but still stored and shown:
a bad recording should not quietly redefine a person's normal, but hiding it
would also hide the fact that data collection is degrading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config

MAD_TO_SD = 1.4826


@dataclass
class Baseline:
    """A person's reference distribution for one metric."""

    metric: str
    centre: float
    spread: float
    n_sessions: int
    method: str  # "robust" | "classical"
    values: list[float]
    unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None


def compute(
    history: pd.DataFrame, metric: str, cfg: Config
) -> Baseline:
    """Build the reference distribution for ``metric`` from prior sessions.

    ``history`` must already exclude the session being evaluated -- a session
    that contributes to its own baseline can never deviate from it.
    """
    section = cfg.section("flagging.baseline")
    min_sessions = int(section["min_sessions"])

    if history.empty or metric not in history:
        return _unavailable(metric, "no previous sessions for this person yet")

    frame = history
    if bool(section["exclude_low_confidence"]) and "low_confidence" in frame:
        frame = frame[frame["low_confidence"] == 0]

    frame = frame.dropna(subset=[metric])
    if frame.empty:
        return _unavailable(
            metric, f"no previous session has a usable {metric} measurement"
        )

    frame = _window(frame, section)
    values = frame[metric].astype(float).to_numpy()

    if values.size < min_sessions:
        return _unavailable(
            metric,
            f"a personal baseline needs at least {min_sessions} previous sessions; "
            f"{values.size} available so far",
        )

    if bool(section["robust"]):
        centre = float(np.median(values))
        spread = float(MAD_TO_SD * np.median(np.abs(values - centre)))
        method = "robust"
    else:
        centre = float(values.mean())
        spread = float(values.std(ddof=1))
        method = "classical"

    if spread <= 0 or not np.isfinite(spread):
        # Identical values across sessions; any deviation is then judged purely
        # by the minimum-detectable-change floor in the trend rule.
        spread = 0.0

    return Baseline(
        metric=metric, centre=centre, spread=spread, n_sessions=int(values.size),
        method=method, values=values.tolist(),
    )


def _window(frame: pd.DataFrame, section: Config) -> pd.DataFrame:
    """Trailing window: whichever of days or session count keeps more data."""
    window_days = int(section["window_days"])
    min_count = int(section["window_min_sessions"])

    ordered = frame.sort_values("session_date")
    by_count = ordered.tail(max(min_count, 1))

    if "session_date" not in ordered:
        return by_count
    latest = ordered["session_date"].max()
    cutoff = latest - pd.Timedelta(days=window_days)
    by_days = ordered[ordered["session_date"] >= cutoff]

    return by_days if len(by_days) >= len(by_count) else by_count


def _unavailable(metric: str, reason: str) -> Baseline:
    return Baseline(
        metric=metric, centre=float("nan"), spread=float("nan"), n_sessions=0,
        method="none", values=[], unavailable_reason=reason,
    )
