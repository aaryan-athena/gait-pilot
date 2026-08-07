"""Combining absolute, trend and quality checks into one flag list.

Both trigger types can fire independently, and both are reported -- a person can
be slow in absolute terms without changing, or can change sharply while still
within population norms. The second is the case a trend tool exists to catch.

A note on the false-alarm rate: six metrics each tested at roughly 1.5 SD gives a
substantial chance that at least one fires in any given session. That is a
deliberate consequence of biasing toward sensitivity, as the brief asks, since a
missed decline is worse than a false alarm. It does mean the flag list should be
read as "look at this", not "something is wrong", and it is why every flag
carries its own plain-language explanation and raw numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import Config
from ..types import CORE_METRICS, Flag, QualityReport, SessionMetrics
from . import absolute as absolute_module
from . import baseline as baseline_module
from . import trend as trend_module


@dataclass
class FlagResult:
    flags: list[Flag] = field(default_factory=list)
    baselines: dict[str, baseline_module.Baseline] = field(default_factory=dict)
    deviations: dict[str, trend_module.Deviation] = field(default_factory=dict)
    baseline_notes: list[str] = field(default_factory=list)

    @property
    def high(self) -> list[Flag]:
        return [f for f in self.flags if f.severity == "high"]

    @property
    def by_trigger(self) -> dict[str, list[Flag]]:
        out: dict[str, list[Flag]] = {}
        for flag in self.flags:
            out.setdefault(flag.trigger, []).append(flag)
        return out


def evaluate(
    metrics: SessionMetrics,
    quality: QualityReport,
    history: pd.DataFrame,
    cfg: Config,
) -> FlagResult:
    """Run every flagging rule for one session.

    ``history`` must contain only sessions strictly before the one being
    evaluated, otherwise a session enters its own baseline.
    """
    result = FlagResult()
    result.flags.extend(absolute_module.evaluate(metrics, cfg))

    for metric in CORE_METRICS:
        base = baseline_module.compute(history, metric, cfg)
        result.baselines[metric] = base
        value = metrics.value(metric)

        if not base.available:
            continue
        if value is None:
            continue

        deviation = trend_module.assess(value, base, cfg)
        if deviation is None:
            continue
        result.deviations[metric] = deviation
        if not deviation.triggered:
            continue

        hits = trend_module.count_recent_deviations(history, base, cfg)
        confirmed = (hits + 1) >= int(cfg["flagging.trend.confirm_min_hits"])
        result.flags.append(
            trend_module.to_flag(deviation, base, cfg, confirmed=confirmed)
        )

    if quality.low_confidence:
        result.flags.append(
            Flag(
                code="low_data_quality",
                metric="quality_score",
                severity="moderate",
                trigger="quality",
                message=(
                    f"Recording quality scored {quality.score:.2f}, below the "
                    "threshold for reliable measurement. The numbers below should "
                    "be treated as provisional, and this session is excluded from "
                    "this person's baseline."
                ),
                detail={"score": quality.score, **{
                    k: v for k, v in quality.components.items()
                }},
            )
        )

    unavailable_baselines = [
        b.unavailable_reason for b in result.baselines.values()
        if not b.available and b.unavailable_reason
    ]
    if unavailable_baselines:
        result.baseline_notes.append(sorted(set(unavailable_baselines))[0])

    order = {"high": 0, "moderate": 1}
    result.flags.sort(key=lambda f: (order.get(f.severity, 2), f.trigger, f.metric))
    return result
