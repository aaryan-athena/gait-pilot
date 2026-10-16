"""One person's session history, summarised for the trends page.

A single session answers "how did this walk look". The trends page has to
answer a different question -- "is anything changing, and does any of it need
attention" -- and the answer is spread across every stored session and every
flag raised on them. This module gathers it into one place.

Like :mod:`.plain`, it is translation only. Concerns come from the flags the
flagging engine already raised and stored; nothing is re-tested here, so the
trends summary can never disagree with what a session reported when it was
analysed. The one judgement it adds is whether the latest value has *moved*,
and that uses the same minimum detectable change the trend flags use, so a
change too small for the tool to measure is called "steady" rather than shown
as a rise or a fall.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..flagging.baseline import compute as compute_baseline
from ..types import CORE_METRICS, DETERIORATION_DIRECTION, Flag
from .plain import PLAIN

Movement = Literal["better", "worse", "steady"]
Tone = Literal["good", "watch", "attention", "insufficient"]

#: A recording problem seen in at least this share of sessions is reported as
#: a setup issue: something to fix once, rather than a run of bad luck.
RECURRING_PROBLEM_SHARE = 0.5


@dataclass
class MetricTrend:
    """Where one metric stands in the latest session, against its history."""

    metric: str
    name: str
    latest: float
    value_text: str
    reference: Optional[float] = None  # baseline centre, or the previous value
    reference_kind: Optional[Literal["baseline", "previous"]] = None
    movement: Optional[Movement] = None
    change_text: Optional[str] = None
    n_values: int = 0


@dataclass
class Concern:
    """A metric flagged in this person's history, in plain terms."""

    metric: str
    name: str
    severity: Literal["attention", "watch"]
    message: str
    in_latest: bool
    sessions_flagged: int
    first_seen: str
    last_seen: str
    confirmed: bool = False
    triggers: tuple[str, ...] = ()


@dataclass
class HistorySummary:
    n_sessions: int
    first_date: Optional[str]
    last_date: Optional[str]
    n_low_confidence: int
    headline: str
    sub_headline: str
    tone: Tone
    latest: list[MetricTrend] = field(default_factory=list)
    concerns: list[Concern] = field(default_factory=list)
    recording_issues: list[tuple[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def active_concerns(self) -> list[Concern]:
        return [c for c in self.concerns if c.in_latest]

    @property
    def earlier_concerns(self) -> list[Concern]:
        return [c for c in self.concerns if not c.in_latest]


def summarise_history(
    history: pd.DataFrame,
    flags_by_session: dict[str, Sequence[Flag]],
    cfg: Config,
) -> HistorySummary:
    """Summarise a person's stored sessions, oldest first in ``history``."""
    if history.empty:
        return HistorySummary(
            n_sessions=0, first_date=None, last_date=None, n_low_confidence=0,
            headline="No walks saved yet",
            sub_headline="Analyse a walk and save it to start this person's trend.",
            tone="insufficient",
        )

    history = history.sort_values("session_date").reset_index(drop=True)
    dates = pd.to_datetime(history["session_date"])
    latest_row = history.iloc[-1]
    earlier = history.iloc[:-1]

    summary = HistorySummary(
        n_sessions=len(history),
        first_date=str(dates.iloc[0].date()),
        last_date=str(dates.iloc[-1].date()),
        n_low_confidence=int(pd.to_numeric(history.get("low_confidence", 0)).sum())
        if "low_confidence" in history else 0,
        headline="", sub_headline="", tone="good",
    )
    summary.latest = _latest_trends(latest_row, earlier, cfg)
    summary.concerns = _concerns(history, flags_by_session)
    summary.recording_issues = _recurring_problems(history)
    summary.notes = _notes(history, latest_row)
    summary.headline, summary.sub_headline, summary.tone = _verdict(summary, latest_row)
    return summary


# --------------------------------------------------------------------------
# latest values against history
# --------------------------------------------------------------------------
def _latest_trends(latest: pd.Series, earlier: pd.DataFrame, cfg: Config) -> list[MetricTrend]:
    mdc = cfg.get("flagging.trend.minimum_detectable_change", {}) or {}
    out: list[MetricTrend] = []
    for metric in CORE_METRICS:
        value = _number(latest.get(metric))
        if value is None or metric not in PLAIN:
            continue
        spec = PLAIN[metric]
        trend = MetricTrend(
            metric=metric, name=spec["name"], latest=value,
            value_text=f"{spec['fmt'].format(value)} {spec['unit']}".strip(),
            n_values=int(earlier[metric].notna().sum()) + 1 if metric in earlier else 1,
        )

        reference, kind = None, None
        if not earlier.empty and metric in earlier:
            baseline = compute_baseline(earlier, metric, cfg)
            if baseline.available:
                reference, kind = baseline.centre, "baseline"
            else:
                previous = earlier[metric].dropna()
                if not previous.empty:
                    reference, kind = float(previous.iloc[-1]), "previous"

        if reference is not None:
            trend.reference, trend.reference_kind = reference, kind
            trend.movement = _movement(metric, value, reference, mdc.get(metric))
            trend.change_text = _change_text(metric, value, reference, kind)
        out.append(trend)
    return out


def _movement(metric: str, value: float, reference: float,
              minimum_change: Optional[float]) -> Movement:
    """Better, worse or steady -- with "steady" meaning "too small to measure"."""
    delta = value - reference
    if minimum_change is not None and abs(delta) < float(minimum_change):
        return "steady"
    if delta == 0:
        return "steady"
    worse_when = DETERIORATION_DIRECTION.get(metric)
    if worse_when is None:
        return "steady"
    return "worse" if np.sign(delta) == worse_when else "better"


def _change_text(metric: str, value: float, reference: float, kind: str) -> str:
    spec = PLAIN[metric]
    delta = value - reference
    sign = "+" if delta >= 0 else "−"
    amount = spec["fmt"].format(abs(delta))
    against = "their usual" if kind == "baseline" else "the last walk"
    return f"{sign}{amount} {spec['unit']} vs {against}".replace("  ", " ")


# --------------------------------------------------------------------------
# concerns
# --------------------------------------------------------------------------
def _concerns(history: pd.DataFrame, flags_by_session: dict[str, Sequence[Flag]]) -> list[Concern]:
    latest_id = history.iloc[-1]["session_id"]
    gathered: dict[str, dict] = {}

    for _, row in history.iterrows():
        session_flags = flags_by_session.get(row["session_id"], ())
        date = str(pd.to_datetime(row["session_date"]).date())
        seen_here: set[str] = set()
        for flag in session_flags:
            # Recording-quality flags describe the video, not the person, and
            # are reported separately so the two are never confused.
            if flag.trigger == "quality" or flag.metric not in PLAIN:
                continue
            entry = gathered.setdefault(flag.metric, {
                "severity": "watch", "message": flag.message, "sessions": set(),
                "first": date, "last": date, "in_latest": False,
                "confirmed": False, "triggers": set(),
            })
            if flag.severity == "high":
                entry["severity"] = "attention"
            if row["session_id"] == latest_id:
                entry["in_latest"] = True
                entry["message"] = flag.message  # the most recent wording leads
            entry["confirmed"] |= bool(flag.confirmed)
            entry["triggers"].add(flag.trigger)
            entry["last"] = date
            seen_here.add(flag.metric)
        for metric in seen_here:
            gathered[metric]["sessions"].add(row["session_id"])

    concerns = [
        Concern(
            metric=metric, name=PLAIN[metric]["name"], severity=e["severity"],
            message=e["message"], in_latest=e["in_latest"],
            sessions_flagged=len(e["sessions"]), first_seen=e["first"],
            last_seen=e["last"], confirmed=e["confirmed"],
            triggers=tuple(sorted(e["triggers"])),
        )
        for metric, e in gathered.items()
    ]
    # Active before earlier, the more serious first, the more persistent first.
    concerns.sort(key=lambda c: (not c.in_latest, c.severity != "attention",
                                 -c.sessions_flagged))
    return concerns


def _recurring_problems(history: pd.DataFrame) -> list[tuple[str, int]]:
    if "recording_diagnostics_json" not in history:
        return []
    counts: Counter = Counter()
    titles: dict[str, str] = {}
    for raw in history["recording_diagnostics_json"]:
        if not raw or not isinstance(raw, str):
            continue
        try:
            entries = json.loads(raw)
        except ValueError:
            continue
        for entry in {e["code"]: e for e in entries}.values():
            counts[entry["code"]] += 1
            titles[entry["code"]] = entry.get("title", entry["code"])
    threshold = max(2, int(np.ceil(RECURRING_PROBLEM_SHARE * len(history))))
    return [(titles[c], n) for c, n in counts.most_common() if n >= threshold]


def _notes(history: pd.DataFrame, latest: pd.Series) -> list[str]:
    notes = []
    if "algo_version" in history and history["algo_version"].nunique() > 1:
        versions = ", ".join(sorted(history["algo_version"].dropna().unique()))
        notes.append(
            f"These walks were measured by different versions of the tool ({versions}). "
            "Part of any change between them may be the tool rather than the person."
        )
    if bool(_number(latest.get("low_confidence")) or 0):
        notes.append(
            "The latest walk was a low-quality recording, so it is left out of this "
            "person's usual range and its numbers are less certain."
        )
    return notes


def _verdict(summary: HistorySummary, latest: pd.Series) -> tuple[str, str, Tone]:
    active = summary.active_concerns
    attention = [c for c in active if c.severity == "attention"]
    if not summary.latest:
        return (
            "The latest walk could not be measured",
            "Check the recording feedback for that session, then record again.",
            "insufficient",
        )
    if attention:
        return (
            f"Worth discussing: {_join([c.name.lower() for c in attention])}",
            "The latest walk has measures outside the range this tool uses as a "
            "prompt to look more closely. That is not a diagnosis -- it is a "
            "suggestion to raise it with a clinician.",
            "attention",
        )
    if active:
        return (
            f"Keep an eye on {_join([c.name.lower() for c in active])}",
            "Nothing stands out strongly in the latest walk, but these are near the "
            "edge of the usual range and worth watching over the next few sessions.",
            "watch",
        )
    worse = [t for t in summary.latest if t.movement == "worse"]
    if worse and summary.n_sessions > 1:
        return (
            "No concerns raised, but some measures have moved",
            f"{_join([t.name.lower() for t in worse]).capitalize()} changed in the "
            "less favourable direction compared with earlier walks, though not by "
            "enough to raise a flag.",
            "good",
        )
    if summary.n_sessions == 1:
        return (
            "No concerns in the first walk",
            "Trends need several walks: a personal baseline forms after a few more "
            "sessions, and changes are judged against it.",
            "good",
        )
    return (
        "No concerns in the latest walk",
        "Every measure taken is within the usual range for this person.",
        "good",
    )


# --------------------------------------------------------------------------
def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]
