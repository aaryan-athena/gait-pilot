"""Plain-language presentation of a session's results.

The metrics layer speaks in the vocabulary of gait analysis -- "stride-time
coefficient of variation", "double-support percentage", "trunk AP sway
normalised by leg length". That is the right vocabulary for the pipeline and the
wrong one for the person reading the result, who is usually a caregiver or a
family member rather than a gait lab.

This module is the translation layer, and it is only translation: it derives
everything from the metrics and flags already computed, and adds no analysis of
its own. Keeping it separate means the wording can be revised for
comprehensibility without any risk of changing what the tool measures.

Three rules on the wording, all of which follow from this being a screening tool:

* **No diagnostic language.** A metric outside its threshold is "worth
  discussing", never "abnormal". The tool cannot diagnose, so its vocabulary
  should not imply that it has.
* **Absence is a distinct state, not a bad result.** "Not measured" gets its own
  status and its own plain reason. Showing it as zero, or blank, or lumped in
  with a normal reading, would each be a different kind of lie.
* **Direction is stated, because it is not guessable.** For speed, higher is
  better; for variability and asymmetry, lower is. A reader cannot infer that
  from the number, so each metric says which way is good.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from ..config import Config
from ..types import CORE_METRICS, SessionMetrics

Status = Literal["good", "watch", "attention", "unmeasured"]

STATUS_LABEL: dict[Status, str] = {
    "good": "Typical",
    "watch": "Keep an eye on",
    "attention": "Worth discussing",
    "unmeasured": "Not measured",
}

#: Ordering, naming and explanation for each metric, aimed at a non-specialist.
#: ``higher_is_better`` is shown to the reader because it cannot be inferred.
PLAIN: dict[str, dict] = {
    "gait_speed_mps": {
        "name": "Walking speed",
        "unit": "m/s",
        "fmt": "{:.2f}",
        "higher_is_better": True,
        "what": "How fast they walk. The single best summary of overall mobility, "
                "and the measure most strongly tied to independence.",
        "everyday": "About {kmh:.1f} km/h.",
    },
    "cadence_spm": {
        "name": "Steps per minute",
        "unit": "steps/min",
        "fmt": "{:.0f}",
        # Cadence has a comfortable band rather than a good direction. Saying
        # "higher is better" would endorse a fast shuffle, which is the opposite
        # of what it usually indicates.
        "direction_text": "usually 100-120 in comfortable walking",
        "what": "How often they take a step. Taken together with step length, "
                "this is what makes up walking speed.",
    },
    "stride_time_cv_pct": {
        "name": "Step-to-step consistency",
        "unit": "% variation",
        "fmt": "{:.1f}",
        "higher_is_better": False,
        "what": "How much the timing of each step varies. Steady, even timing is "
                "a sign of confident walking; irregular timing is the measure "
                "most closely linked to falls in older adults.",
    },
    "step_length_asymmetry_pct": {
        "name": "Left/right evenness",
        "unit": "% difference",
        "fmt": "{:.1f}",
        "higher_is_better": False,
        "what": "How closely the two legs match each other in step length. A "
                "large difference can point to weakness or pain on one side.",
    },
    "double_support_pct": {
        "name": "Time on both feet",
        "unit": "% of each step",
        "fmt": "{:.1f}",
        "higher_is_better": False,
        "what": "How much of each walking cycle is spent with both feet on the "
                "ground. People who feel unsteady tend to spend longer on both "
                "feet, because it is the stable part of the cycle.",
        "known_bias": (
            "This is the least reliable of the measures here: the moment a toe "
            "leaves the ground is genuinely hard to see in a video. Judge it by "
            "how it changes across sessions rather than against the number "
            "itself."
        ),
    },
    "trunk_ap_sway_norm": {
        "name": "Upper-body steadiness",
        "unit": "",
        "fmt": "{:.3f}",
        "higher_is_better": False,
        "what": "How much the upper body rocks forwards and backwards with each "
                "step, relative to their height. More rocking can accompany a "
                "guarded or unsteady gait.",
    },
}

#: Short, non-technical versions of the reasons a metric could not be measured.
#: The full technical reason is still available and still shown on request; this
#: is what leads.
UNMEASURED_PLAIN: list[tuple[str, str]] = [
    ("no calibration", "Needs a one-off setup step to convert pixels into metres. "
                       "Everything else on this page works without it."),
    ("forward travel", "The person does not cross the frame in this video, so "
                       "there is no distance to measure speed over."),
    ("camera itself moved", "The camera moved during the recording, so distance "
                            "cannot be measured reliably."),
    ("camera has probably been moved", "The camera looks like it has moved since "
                                       "setup, so distances are not comparable."),
    ("at least", "Not enough steps in this video. A longer walk, or two to three "
                 "passes back and forth, will produce this."),
    ("no usable walking pass", "No usable walking was found in this video -- see "
                               "the recording feedback above."),
    ("does not transfer between resolutions", "The setup step was done at a "
                                              "different video size, so it does "
                                              "not apply here."),
    ("not enough", "Not enough clean steps in this video to work this out."),
    ("both feet", "The two feet could not be tracked separately well enough for "
                  "this measure."),
    ("two per side", "Too few steps on each side to compare left with right."),
]


@dataclass
class MetricCard:
    """One metric, ready to display."""

    key: str
    name: str
    status: Status
    value_text: str  # formatted value with unit, or ""
    what: str  # what this measures, in plain words
    direction: str  # "higher is better" / "lower is better"
    note: Optional[str] = None  # caveat or plain unmeasured reason
    technical_note: Optional[str] = None  # the full original wording
    everyday: Optional[str] = None  # an everyday-units restatement

    @property
    def measured(self) -> bool:
        return self.status != "unmeasured"


@dataclass
class PlainSummary:
    """A whole session, in the terms a non-specialist reads it in."""

    headline: str
    sub_headline: str
    tone: Literal["good", "watch", "attention", "insufficient"]
    cards: list[MetricCard] = field(default_factory=list)
    n_measured: int = 0
    n_total: int = 0
    walk_description: Optional[str] = None

    @property
    def measured_cards(self) -> list[MetricCard]:
        return [c for c in self.cards if c.measured]

    @property
    def unmeasured_cards(self) -> list[MetricCard]:
        return [c for c in self.cards if not c.measured]


def summarise(result, cfg: Config) -> PlainSummary:
    """Build the plain-language view of a completed session."""
    metrics = result.metrics
    statuses = _statuses_from_flags(result)
    cards = [_card(key, metrics, statuses, cfg) for key in CORE_METRICS]

    n_measured = sum(1 for c in cards if c.measured)
    headline, sub, tone = _verdict(result, cards, n_measured)

    return PlainSummary(
        headline=headline,
        sub_headline=sub,
        tone=tone,
        cards=cards,
        n_measured=n_measured,
        n_total=len(cards),
        walk_description=_describe_walk(result),
    )


# --------------------------------------------------------------------------
def _statuses_from_flags(result) -> dict[str, Status]:
    """Map each flagged metric to a status.

    Derived from the flags rather than re-tested here, so the plain view and the
    clinical view can never disagree about whether something was flagged.
    """
    statuses: dict[str, Status] = {}
    for flag in result.flags.flags:
        if flag.trigger == "quality":
            continue
        current = statuses.get(flag.metric)
        new: Status = "attention" if flag.severity == "high" else "watch"
        if current != "attention":
            statuses[flag.metric] = new
    return statuses


def _leads_verdict(card: MetricCard, result) -> bool:
    """Whether a flagged metric is fit to headline the plain summary.

    A metric with a known measurement bias in this pipeline must not drive the
    headline on the strength of an *absolute* threshold: the absolute number is
    the part the bias corrupts. Double support reads roughly 8-10 points high
    here, so on a healthy adult it crosses the illustrative "high concern" line
    and would tell a caregiver to seek advice about an artefact of 2D video.

    A *trend* flag on the same metric does lead, because a consistent bias
    cancels when a person is compared with their own earlier sessions -- which
    is what this tool is actually for. The flag itself is never suppressed; it
    stays visible, with its caveat, in the cards and in the clinical view.
    """
    if not PLAIN[card.key].get("known_bias"):
        return True
    return any(
        flag.metric == card.key and flag.trigger == "trend"
        for flag in result.flags.flags
    )


def _card(key: str, metrics: SessionMetrics, statuses: dict[str, Status],
          cfg: Config) -> MetricCard:
    spec = PLAIN[key]
    value = metrics.value(key)
    direction = spec.get("direction_text") or (
        "higher is better" if spec.get("higher_is_better") else "lower is better"
    )

    if value is None:
        technical = metrics.unavailable.get(key)
        return MetricCard(
            key=key, name=spec["name"], status="unmeasured", value_text="",
            what=spec["what"], direction=direction,
            note=_plain_unmeasured(technical), technical_note=technical,
        )

    unit = spec["unit"]
    value_text = f"{spec['fmt'].format(value)} {unit}".strip()
    everyday = None
    if key == "gait_speed_mps":
        everyday = spec["everyday"].format(kmh=value * 3.6)

    note = _plain_caveat(metrics.low_confidence_metrics.get(key))
    bias = spec.get("known_bias")
    if bias:
        note = f"{bias} {note}" if note else bias

    return MetricCard(
        key=key, name=spec["name"], status=statuses.get(key, "good"),
        value_text=value_text, what=spec["what"], direction=direction,
        note=note, technical_note=metrics.low_confidence_metrics.get(key),
        everyday=everyday,
    )


def _plain_unmeasured(technical: Optional[str]) -> str:
    if not technical:
        return "This could not be worked out from this video."
    lowered = technical.lower()
    for needle, plain in UNMEASURED_PLAIN:
        if needle.lower() in lowered:
            return plain
    return "This could not be worked out from this video."


def _plain_caveat(technical: Optional[str]) -> Optional[str]:
    """Shorten the low-confidence caveats to a single readable sentence."""
    if not technical:
        return None
    lowered = technical.lower()
    if "fps" in lowered and "quantised" in lowered:
        return ("Treat as a rough figure: the video frame rate is too low to "
                "time steps precisely. Record at 60 fps for a reliable value.")
    if "lateral trunk sway" in lowered:
        return ("Measured front-to-back from the side view, not side-to-side, "
                "which would need a second camera facing the walker.")
    if "single view" in lowered or "occluded" in lowered:
        return ("Measured from one side, so the far leg is partly hidden. Record "
                "a pass in each direction for a fairer left/right comparison.")
    if "inconsistent" in lowered or "independent count" in lowered:
        return ("Treat with caution: the step detection did not agree with itself "
                "on this video.")
    if "not being tracked separately" in lowered:
        return "Treat with caution: the two legs were hard to tell apart here."
    if "single-scalar pixel scale" in lowered:
        return ("Approximate: the distance setup assumes the person stays the same "
                "distance from the camera.")
    if "only" in lowered and "strides" in lowered:
        return "Based on only a few steps, so it is less reliable than usual."
    return None


def _verdict(result, cards: list[MetricCard], n_measured: int) -> tuple[str, str, str]:
    """The single line that leads the results page."""
    if n_measured == 0:
        return (
            "This video could not be measured",
            "Nothing was wrong with the walking -- the recording itself did not "
            "give the tool enough to work with. The feedback below says what to "
            "change.",
            "insufficient",
        )

    attention = [c for c in cards
                 if c.status == "attention" and _leads_verdict(c, result)]
    watch = [c for c in cards
             if c.status == "watch" and _leads_verdict(c, result)]
    trend_flags = [f for f in result.flags.flags if f.trigger == "trend"]

    caveat = _confidence_caveat(result)

    if attention:
        names = _join([c.name.lower() for c in attention])
        return (
            f"Worth discussing: {names}",
            "One or more measures fall outside the range this tool uses as a "
            "prompt to look more closely. That is not a diagnosis -- it is a "
            "suggestion to raise it with a clinician." + caveat,
            "watch" if caveat else "attention",
        )
    if watch:
        names = _join([c.name.lower() for c in watch])
        return (
            f"Mostly typical, keep an eye on {names}",
            "Nothing stands out strongly. The measures noted are near the edge of "
            "the usual range and are worth watching over the next few sessions."
            + caveat,
            "watch",
        )
    if trend_flags:
        return (
            "A change from this person's usual pattern",
            "The individual measures look typical, but one has shifted compared "
            "with this person's own previous sessions. Changes matter more than "
            "single readings.",
            "watch",
        )
    held_back = [c for c in cards
                 if c.status in ("attention", "watch")
                 and not _leads_verdict(c, result)]
    if held_back:
        names = _join([c.name.lower() for c in held_back])
        return (
            "Nothing clearly stands out in this walk",
            f"The measures sit in their typical ranges, except {names}, which "
            "this tool is known to read high from video and so should be judged "
            "on how it changes between sessions rather than on today's number. "
            "Repeat sessions are what make any of this meaningful.",
            "watch",
        )
    return (
        "Nothing stands out in this walk",
        f"All {n_measured} measures the tool could take sit in their typical "
        "ranges. Repeat sessions are what make this meaningful -- the tool is "
        "built to spot change over time.",
        "good",
    )


def _confidence_caveat(result) -> str:
    """Temper a finding the recording cannot actually support.

    A verdict of "worth discussing" invites someone to act -- to raise it with a
    clinician, or to worry. When the recording had blocking problems, or scored
    below the quality threshold, or rested on a handful of steps, that invitation
    is premature: the most likely explanation is the video, not the person. The
    finding is still named and still shown; what changes is that the reader is
    told to fix the recording and repeat before drawing anything from it.
    """
    reasons: list[str] = []

    quality = getattr(result, "quality", None)
    if quality is not None and getattr(quality, "low_confidence", False):
        reasons.append("the recording scored below the quality threshold")

    diagnostics = getattr(result, "diagnostics", None)
    blockers = getattr(diagnostics, "blockers", []) if diagnostics else []
    if blockers:
        reasons.append(f"the recording has a blocking problem ({blockers[0].title.lower()})")

    n_valid = (getattr(result.analysis, "cycle_summary", {}) or {}).get("n_valid", 0)
    if 0 < n_valid < 4:
        reasons.append(f"only {n_valid} complete step cycles were usable")

    if not reasons:
        return ""
    return (
        " Read this cautiously, though: " + _join(reasons) + ". Improve the "
        "recording using the feedback below and repeat before drawing anything "
        "from it."
    )


def _describe_walk(result) -> Optional[str]:
    """One sentence on what was actually analysed, in plain terms."""
    summary = result.analysis.cycle_summary or {}
    n_valid = summary.get("n_valid", 0)
    if not n_valid:
        return None
    passes = len(result.analysis.passes)
    duration = result.extraction.info.duration_s
    pass_text = "one walk" if passes == 1 else f"{passes} walks"
    return (
        f"Measured from {n_valid} complete step cycles across {pass_text} in "
        f"{duration:.0f} seconds of video."
    )


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"
