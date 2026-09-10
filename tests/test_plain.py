"""The plain-language results layer.

These tests are about wording as much as logic, because the wording is the
feature: this layer exists so that someone with no background in gait analysis
reads the right thing, and "the right thing" for a screening tool includes not
sounding like a diagnosis.
"""
import pandas as pd
import pytest

from gaitscreen.features.session import analyse
from gaitscreen.flagging.engine import evaluate
from gaitscreen.reporting.plain import PLAIN, summarise
from gaitscreen.types import CORE_METRICS, Flag, QualityReport, SessionMetrics
from fixtures.extraction import build_extraction


class _FakeResult:
    """A SessionResult stand-in, so wording can be tested without running video."""

    def __init__(self, metrics, flags, analysis=None, extraction=None):
        self.metrics = metrics
        self.flags = flags
        self.analysis = analysis or _EmptyAnalysis()
        self.extraction = extraction or _EmptyExtraction()


class _EmptyAnalysis:
    cycle_summary: dict = {}
    passes: list = []


class _EmptyExtraction:
    class info:
        duration_s = 8.0


def _flags(*flag_list):
    class Result:
        flags = list(flag_list)
        baselines: dict = {}
        baseline_notes: list = []
    return Result()


def _flag(metric, severity="high", trigger="absolute"):
    return Flag(code=f"{metric}_x", metric=metric, severity=severity,
                trigger=trigger, message="m", detail={})


# --------------------------------------------------------------------------
# every metric is describable
# --------------------------------------------------------------------------
def test_every_core_metric_has_plain_wording():
    assert set(PLAIN) == set(CORE_METRICS)
    for key, spec in PLAIN.items():
        assert spec["name"] and not spec["name"].endswith("."), key
        assert len(spec["what"]) > 40, key
        assert spec.get("direction_text") or "higher_is_better" in spec, key


def test_no_metric_name_uses_gait_lab_vocabulary():
    """The point of this layer is that the names need no glossary."""
    jargon = ("coefficient", "variability", "asymmetry", "cadence", "sagittal",
              "normalised", "cv", "trunk ap")
    for key, spec in PLAIN.items():
        lowered = spec["name"].lower()
        assert not any(word in lowered for word in jargon), (
            f"{key} is still named in technical terms: {spec['name']}"
        )


def test_cadence_states_a_band_not_a_direction():
    """"Higher is better" would endorse a fast shuffle."""
    assert "higher_is_better" not in PLAIN["cadence_spm"]
    assert "100-120" in PLAIN["cadence_spm"]["direction_text"]


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------
def test_all_typical_reads_as_nothing_standing_out(cfg):
    metrics = SessionMetrics(gait_speed_mps=1.25, cadence_spm=110)
    summary = summarise(_FakeResult(metrics, _flags()), cfg)

    assert summary.tone == "good"
    assert "Nothing stands out" in summary.headline
    assert summary.n_measured == 2


def test_high_flag_reads_as_worth_discussing(cfg):
    metrics = SessionMetrics(gait_speed_mps=0.45)
    summary = summarise(
        _FakeResult(metrics, _flags(_flag("gait_speed_mps"))), cfg
    )

    assert summary.tone == "attention"
    assert "Worth discussing" in summary.headline
    assert "walking speed" in summary.headline


def test_moderate_flag_reads_as_keep_an_eye_on(cfg):
    metrics = SessionMetrics(gait_speed_mps=0.85)
    summary = summarise(
        _FakeResult(metrics, _flags(_flag("gait_speed_mps", "moderate"))), cfg
    )
    assert summary.tone == "watch"
    assert "keep an eye on" in summary.headline.lower()


def test_nothing_measurable_blames_the_recording_not_the_person(cfg):
    """The commonest pilot outcome, and the easiest one to word alarmingly."""
    metrics = SessionMetrics()
    for metric in CORE_METRICS:
        metrics.mark_unavailable(metric, "no usable walking pass was found")

    summary = summarise(_FakeResult(metrics, _flags()), cfg)
    assert summary.tone == "insufficient"
    assert "could not be measured" in summary.headline
    assert "Nothing was wrong with the walking" in summary.sub_headline


def test_verdict_never_uses_diagnostic_language(cfg):
    """A screening tool must not sound like it has diagnosed something."""
    banned = ("abnormal", "pathological", "impaired", "disorder", "diagnosis of",
              "you have", "confirms")
    cases = [
        (SessionMetrics(gait_speed_mps=0.4), _flags(_flag("gait_speed_mps"))),
        (SessionMetrics(gait_speed_mps=0.85),
         _flags(_flag("gait_speed_mps", "moderate"))),
        (SessionMetrics(gait_speed_mps=1.3), _flags()),
    ]
    for metrics, flags in cases:
        summary = summarise(_FakeResult(metrics, flags), cfg)
        text = f"{summary.headline} {summary.sub_headline}".lower()
        for word in banned:
            assert word not in text, f"{word!r} appears in: {text}"


# --------------------------------------------------------------------------
# unmeasured metrics
# --------------------------------------------------------------------------
def test_unmeasured_metrics_get_a_plain_reason(cfg):
    metrics = SessionMetrics(cadence_spm=100)
    metrics.mark_unavailable(
        "gait_speed_mps",
        "no calibration on file for this user, so distances cannot be converted "
        "to metres; scale-free metrics are unaffected",
    )
    summary = summarise(_FakeResult(metrics, _flags()), cfg)
    card = next(c for c in summary.cards if c.key == "gait_speed_mps")

    assert card.status == "unmeasured"
    assert not card.value_text
    assert "one-off setup step" in card.note
    # The original wording is kept for the technical view.
    assert "scale-free" in card.technical_note


def test_unmeasured_reason_falls_back_rather_than_showing_jargon(cfg):
    metrics = SessionMetrics()
    metrics.mark_unavailable("cadence_spm", "some unmapped internal condition")
    summary = summarise(_FakeResult(metrics, _flags()), cfg)
    card = next(c for c in summary.cards if c.key == "cadence_spm")

    assert card.note == "This could not be worked out from this video."


def test_measured_and_unmeasured_are_separable(cfg):
    metrics = SessionMetrics(cadence_spm=100, double_support_pct=22.0)
    for metric in ("gait_speed_mps", "stride_time_cv_pct"):
        metrics.mark_unavailable(metric, "at least 10 are needed")

    summary = summarise(_FakeResult(metrics, _flags()), cfg)
    assert len(summary.measured_cards) == 2
    assert len(summary.unmeasured_cards) == 4


# --------------------------------------------------------------------------
# the double-support bias
# --------------------------------------------------------------------------
def test_double_support_always_carries_its_known_bias(cfg):
    metrics = SessionMetrics(double_support_pct=22.0)
    summary = summarise(_FakeResult(metrics, _flags()), cfg)
    card = next(c for c in summary.cards if c.key == "double_support_pct")

    assert "reads this measure about 8-10 points high" in card.note


def test_biased_metric_does_not_headline_on_an_absolute_threshold(cfg):
    """A caregiver must not be told to seek advice about a known artefact.

    Double support reads high from 2D video, so on a healthy adult it crosses
    the illustrative threshold. The flag stays visible on the card and in the
    clinical view; it just does not drive the headline.
    """
    metrics = SessionMetrics(double_support_pct=39.0, cadence_spm=100)
    summary = summarise(
        _FakeResult(metrics, _flags(_flag("double_support_pct"))), cfg
    )

    assert "Worth discussing" not in summary.headline
    assert "read high from video" in summary.sub_headline
    # The card itself still shows the flag.
    card = next(c for c in summary.cards if c.key == "double_support_pct")
    assert card.status == "attention"


def test_biased_metric_does_headline_on_a_trend_change(cfg):
    """A consistent bias cancels when comparing a person with themselves."""
    metrics = SessionMetrics(double_support_pct=39.0)
    summary = summarise(
        _FakeResult(metrics, _flags(_flag("double_support_pct", trigger="trend"))),
        cfg,
    )
    assert summary.tone == "attention"
    assert "time on both feet" in summary.headline


def test_unbiased_metric_still_headlines_normally(cfg):
    metrics = SessionMetrics(stride_time_cv_pct=9.0)
    summary = summarise(
        _FakeResult(metrics, _flags(_flag("stride_time_cv_pct"))), cfg
    )
    assert summary.tone == "attention"
    assert "step-to-step consistency" in summary.headline


# --------------------------------------------------------------------------
# against a real analysed session
# --------------------------------------------------------------------------
def test_summarises_a_real_pipeline_result(cfg):
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16, in_place=True,
                                     leg_length_px=340, width=1280, height=720)
    analysis = analyse(extraction, cfg)
    quality = QualityReport(score=0.9, low_confidence=False)
    flags = evaluate(analysis.metrics, quality, pd.DataFrame(), cfg)

    class Result:
        pass
    result = Result()
    result.metrics = analysis.metrics
    result.analysis = analysis
    result.extraction = extraction
    result.flags = flags

    summary = summarise(result, cfg)
    assert summary.headline
    assert summary.n_total == len(CORE_METRICS)
    assert summary.measured_cards, "a clean synthetic walk should measure something"
    assert summary.walk_description and "step cycles" in summary.walk_description
    for card in summary.cards:
        assert card.name and card.what and card.direction


# --------------------------------------------------------------------------
# confidence tempering
# --------------------------------------------------------------------------
class _Diagnostics:
    def __init__(self, blockers=()):
        self.blockers = list(blockers)


class _Blocker:
    def __init__(self, title):
        self.title = title


class _Analysis(_EmptyAnalysis):
    def __init__(self, n_valid):
        self.cycle_summary = {"n_valid": n_valid}
        self.passes = [object()]


def test_finding_is_tempered_when_the_recording_had_a_blocker(cfg):
    """A verdict that invites action must not rest on a recording that failed."""
    metrics = SessionMetrics(step_length_asymmetry_pct=17.0)
    result = _FakeResult(metrics, _flags(_flag("step_length_asymmetry_pct")),
                         analysis=_Analysis(8))
    result.quality = QualityReport(score=0.9, low_confidence=False)
    result.diagnostics = _Diagnostics([_Blocker("Subject is too small in the frame")])

    summary = summarise(result, cfg)
    assert "left/right evenness" in summary.headline
    assert "Read this cautiously" in summary.sub_headline
    assert "subject is too small" in summary.sub_headline
    assert summary.tone == "watch", "a blocked recording must not read as settled"


def test_finding_is_tempered_by_a_low_quality_score(cfg):
    metrics = SessionMetrics(step_length_asymmetry_pct=17.0)
    result = _FakeResult(metrics, _flags(_flag("step_length_asymmetry_pct")),
                         analysis=_Analysis(8))
    result.quality = QualityReport(score=0.3, low_confidence=True)
    result.diagnostics = _Diagnostics()

    summary = summarise(result, cfg)
    assert "below the quality threshold" in summary.sub_headline


def test_finding_is_tempered_by_too_few_step_cycles(cfg):
    metrics = SessionMetrics(step_length_asymmetry_pct=17.0)
    result = _FakeResult(metrics, _flags(_flag("step_length_asymmetry_pct")),
                         analysis=_Analysis(2))
    result.quality = QualityReport(score=0.9, low_confidence=False)
    result.diagnostics = _Diagnostics()

    summary = summarise(result, cfg)
    assert "2 complete step cycles" in summary.sub_headline


def test_a_sound_recording_states_the_finding_without_hedging(cfg):
    """The tempering must not fire on a good recording, or it is just noise."""
    metrics = SessionMetrics(step_length_asymmetry_pct=17.0)
    result = _FakeResult(metrics, _flags(_flag("step_length_asymmetry_pct")),
                         analysis=_Analysis(12))
    result.quality = QualityReport(score=0.9, low_confidence=False)
    result.diagnostics = _Diagnostics()

    summary = summarise(result, cfg)
    assert summary.tone == "attention"
    assert "Read this cautiously" not in summary.sub_headline
