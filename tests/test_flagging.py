"""Flagging: absolute thresholds, personal baselines and trend deviations."""
import numpy as np
import pandas as pd
import pytest

from gaitscreen.flagging import absolute, baseline as baseline_module, trend
from gaitscreen.flagging.engine import evaluate
from gaitscreen.types import QualityReport, SessionMetrics

GOOD_QUALITY = QualityReport(score=0.9, low_confidence=False)


def _history(values, metric="gait_speed_mps", start="2026-01-01", freq="30D",
             low_confidence=None):
    dates = pd.date_range(start=start, periods=len(values), freq=freq)
    frame = pd.DataFrame({
        "session_date": dates,
        metric: values,
        "low_confidence": low_confidence or [0] * len(values),
    })
    return frame


# --------------------------------------------------------------------------
# absolute thresholds
# --------------------------------------------------------------------------
def test_slow_walking_flags_high_risk(cfg):
    metrics = SessionMetrics(gait_speed_mps=0.45)
    flags = absolute.evaluate(metrics, cfg)

    assert len(flags) == 1
    assert flags[0].severity == "high"
    assert flags[0].trigger == "absolute"
    assert "0.45" in flags[0].message and "0.60" in flags[0].message


def test_moderate_band_flags_moderate(cfg):
    flags = absolute.evaluate(SessionMetrics(gait_speed_mps=0.85), cfg)
    assert [f.severity for f in flags] == ["moderate"]


def test_normal_walking_raises_nothing(cfg):
    assert absolute.evaluate(SessionMetrics(gait_speed_mps=1.25), cfg) == []


def test_high_supersedes_moderate_for_one_metric(cfg):
    """A single metric must not produce two flags for the same problem."""
    flags = absolute.evaluate(SessionMetrics(gait_speed_mps=0.3), cfg)
    assert len(flags) == 1


def test_unmeasured_metrics_are_not_flagged(cfg):
    """A missing measurement is not evidence of a normal one, or an abnormal one."""
    metrics = SessionMetrics()
    metrics.mark_unavailable("gait_speed_mps", "no calibration")
    assert absolute.evaluate(metrics, cfg) == []


def test_thresholds_are_configurable(cfg):
    strict = cfg.with_overrides(
        {"flagging": {"absolute": {"gait_speed_mps": {"high_risk_below": 1.4}}}}
    )
    assert absolute.evaluate(SessionMetrics(gait_speed_mps=1.25), strict)


def test_flags_carry_the_illustrative_caveat(cfg):
    flags = absolute.evaluate(SessionMetrics(gait_speed_mps=0.45), cfg)
    assert "illustrative" in flags[0].detail["note"]


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def test_baseline_requires_a_minimum_number_of_sessions(cfg):
    """An SD from three observations is not a reference distribution."""
    result = baseline_module.compute(_history([1.1, 1.2, 1.15]), "gait_speed_mps", cfg)
    assert not result.available
    assert "at least" in result.unavailable_reason


def test_baseline_computed_once_enough_sessions_exist(cfg):
    result = baseline_module.compute(
        _history([1.10, 1.15, 1.12, 1.18, 1.14]), "gait_speed_mps", cfg
    )
    assert result.available
    assert result.method == "robust"
    assert abs(result.centre - 1.14) < 0.02


def test_robust_baseline_resists_a_single_bad_session(cfg):
    """One poor recording must not redefine what is normal for a person."""
    clean = [1.10, 1.15, 1.12, 1.18, 1.14]
    contaminated = clean + [0.20]

    robust = baseline_module.compute(_history(contaminated), "gait_speed_mps", cfg)
    classical = baseline_module.compute(
        _history(contaminated), "gait_speed_mps",
        cfg.with_overrides({"flagging": {"baseline": {"robust": False}}}),
    )

    assert abs(robust.centre - 1.14) < 0.03
    assert classical.centre < robust.centre - 0.1, "mean should be dragged down"
    assert robust.spread < classical.spread


def test_low_confidence_sessions_excluded_from_the_baseline(cfg):
    values = [1.10, 1.15, 1.12, 1.18, 1.14, 0.30]
    frame = _history(values, low_confidence=[0, 0, 0, 0, 0, 1])

    result = baseline_module.compute(frame, "gait_speed_mps", cfg)
    assert result.n_sessions == 5
    assert 0.30 not in result.values


def test_window_falls_back_to_session_count_at_slow_cadence(cfg):
    """At monthly screening, 90 days is three sessions -- too few on its own."""
    frame = _history([1.1, 1.12, 1.14, 1.16, 1.18, 1.20], freq="30D")
    result = baseline_module.compute(frame, "gait_speed_mps", cfg)

    assert result.available
    assert result.n_sessions >= int(cfg["flagging.baseline.window_min_sessions"])


# --------------------------------------------------------------------------
# trend deviations
# --------------------------------------------------------------------------
def _baseline(cfg, values=(1.10, 1.15, 1.12, 1.18, 1.14)):
    return baseline_module.compute(_history(list(values)), "gait_speed_mps", cfg)


def test_deterioration_triggers(cfg):
    base = _baseline(cfg)
    deviation = trend.assess(0.70, base, cfg)

    assert deviation.deteriorating
    assert deviation.triggered
    flag = trend.to_flag(deviation, base, cfg)
    assert flag.trigger == "trend"
    assert "decreased" in flag.message


def test_improvement_does_not_trigger(cfg):
    """Walking faster than usual is not a fall risk."""
    base = _baseline(cfg)
    deviation = trend.assess(1.60, base, cfg)

    assert not deviation.deteriorating
    assert not deviation.triggered


def test_minimum_detectable_change_floors_the_test(cfg):
    """A very consistent baseline must not make measurement noise significant."""
    tight = baseline_module.compute(
        _history([1.100, 1.101, 1.099, 1.100, 1.101]), "gait_speed_mps", cfg
    )
    deviation = trend.assess(1.09, tight, cfg)

    assert deviation.exceeds_sds, "sanity: this is many SDs on a tight baseline"
    assert not deviation.exceeds_mdc
    assert not deviation.triggered, (
        "a 0.01 m/s change is below the tool's own repeatability and must not flag"
    )


def test_change_beyond_mdc_and_sds_triggers(cfg):
    tight = baseline_module.compute(
        _history([1.100, 1.101, 1.099, 1.100, 1.101]), "gait_speed_mps", cfg
    )
    deviation = trend.assess(0.95, tight, cfg)
    assert deviation.triggered


def test_direction_is_metric_specific(cfg):
    """Rising variability is deterioration; rising speed is not."""
    frame = _history([2.0, 2.1, 2.05, 2.15, 2.08], metric="stride_time_cv_pct")
    base = baseline_module.compute(frame, "stride_time_cv_pct", cfg)

    assert trend.assess(6.0, base, cfg).deteriorating
    assert not trend.assess(1.0, base, cfg).deteriorating


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
def test_engine_combines_absolute_and_trend(cfg):
    history = _history([1.10, 1.15, 1.12, 1.18, 1.14])
    metrics = SessionMetrics(gait_speed_mps=0.50)

    result = evaluate(metrics, GOOD_QUALITY, history, cfg)
    triggers = {f.trigger for f in result.flags}
    assert triggers == {"absolute", "trend"}, (
        "a person both slow in general and slower than their own normal should "
        "raise both kinds of flag"
    )


def test_engine_flags_trend_without_absolute(cfg):
    """The case a trend tool exists for: a real decline still inside population norms."""
    history = _history([1.40, 1.45, 1.42, 1.48, 1.44])
    result = evaluate(SessionMetrics(gait_speed_mps=1.15), GOOD_QUALITY, history, cfg)

    assert [f.trigger for f in result.flags] == ["trend"]


def test_engine_without_history_uses_absolute_only(cfg):
    result = evaluate(SessionMetrics(gait_speed_mps=0.5), GOOD_QUALITY,
                      pd.DataFrame(), cfg)
    assert all(f.trigger == "absolute" for f in result.flags)
    assert result.baseline_notes


def test_low_quality_raises_its_own_flag(cfg):
    quality = QualityReport(score=0.35, low_confidence=True)
    result = evaluate(SessionMetrics(cadence_spm=100), quality, pd.DataFrame(), cfg)

    quality_flags = [f for f in result.flags if f.trigger == "quality"]
    assert len(quality_flags) == 1
    assert "excluded from this person's baseline" in quality_flags[0].message


def test_repeated_deviation_is_marked_confirmed(cfg):
    """Confirmation aids triage; it must never suppress a flag."""
    history = _history([1.10, 1.15, 1.12, 1.18, 1.14, 0.72, 0.70])
    result = evaluate(SessionMetrics(gait_speed_mps=0.71), GOOD_QUALITY, history, cfg)

    trend_flags = [f for f in result.flags if f.trigger == "trend"]
    assert trend_flags and trend_flags[0].confirmed


def test_single_deviation_flags_but_is_not_confirmed(cfg):
    history = _history([1.10, 1.15, 1.12, 1.18, 1.14])
    result = evaluate(SessionMetrics(gait_speed_mps=0.72), GOOD_QUALITY, history, cfg)

    trend_flags = [f for f in result.flags if f.trigger == "trend"]
    assert trend_flags, "sensitivity: one deviating session must still flag"
    assert not trend_flags[0].confirmed


def test_flags_are_ordered_most_severe_first(cfg):
    metrics = SessionMetrics(gait_speed_mps=0.85, double_support_pct=35.0)
    result = evaluate(metrics, GOOD_QUALITY, pd.DataFrame(), cfg)

    severities = [f.severity for f in result.flags]
    assert severities == sorted(severities, key=lambda s: {"high": 0, "moderate": 1}[s])


def test_every_flag_has_a_plain_language_message(cfg):
    metrics = SessionMetrics(gait_speed_mps=0.45, stride_time_cv_pct=7.0,
                             double_support_pct=33.0)
    result = evaluate(metrics, QualityReport(score=0.4, low_confidence=True),
                      pd.DataFrame(), cfg)

    assert result.flags
    for flag in result.flags:
        assert len(flag.message) > 40
        assert flag.message[0].isupper() or flag.message.split()[0].islower()
        assert flag.detail
