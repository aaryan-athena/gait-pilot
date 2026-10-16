"""Camera setup (coverage length) and the trends-page history summary."""
import json

import numpy as np
import pandas as pd
import pytest

from gaitscreen.calibration.model import CalibrationError
from gaitscreen.calibration.setup import (CAMERA_SETUP_ID, is_reserved,
                                          load_setup, save_setup,
                                          validate_coverage)
from gaitscreen.reporting.history import summarise_history
from gaitscreen.storage.repository import SessionRepository
from gaitscreen.types import Flag


@pytest.fixture
def repository(tmp_path):
    repo = SessionRepository(tmp_path / "test.db")
    yield repo
    repo.close()


# --------------------------------------------------------------------------
# camera setup
# --------------------------------------------------------------------------
def test_no_setup_until_one_is_saved(repository):
    assert load_setup(repository) is None


def test_setup_round_trips(repository):
    save_setup(repository, 4.5)
    setup = load_setup(repository)
    assert setup is not None
    assert setup.coverage_m == pytest.approx(4.5)


def test_a_new_setup_replaces_the_old_one(repository):
    save_setup(repository, 4.5)
    save_setup(repository, 6.0)
    assert load_setup(repository).coverage_m == pytest.approx(6.0)


@pytest.mark.parametrize("width,height", [(1920, 1080), (832, 464), (464, 832)])
def test_the_setup_applies_at_any_resolution(repository, width, height):
    """A pixel scale breaks across video sizes; a coverage length does not.

    The coverage is a physical distance across the whole frame, so whatever the
    video's size, the full frame width must map back onto it.
    """
    setup = save_setup(repository, 5.0)
    calibration = setup.calibration_for(width, height)

    assert calibration.applies_to(width, height)
    assert calibration.scale_m_per_px * width == pytest.approx(5.0)


def test_a_session_links_to_the_setup_that_measured_it(repository):
    """The per-video calibration keeps the stored id, so the session's
    reference to its calibration points at a row that actually exists."""
    setup = save_setup(repository, 5.0)
    calibration = setup.calibration_for(1920, 1080)
    assert repository.get_calibration(calibration.calibration_id) is not None


@pytest.mark.parametrize("coverage", [0.2, 450.0, 30.0])
def test_implausible_coverage_is_refused(coverage):
    """Centimetres or feet typed as metres would silently scale every speed."""
    with pytest.raises(CalibrationError):
        validate_coverage(coverage)


def test_the_setup_is_not_a_person():
    assert is_reserved(CAMERA_SETUP_ID)
    assert not is_reserved("Bob")


def test_speed_is_measured_with_the_setup(cfg):
    """End to end: a walk across a frame whose coverage is known gives m/s.

    The fixture walks at 180 px/s across a 1280 px frame. With the frame
    covering 6.4 m, one pixel is 5 mm, so the true speed is 0.9 m/s.
    """
    from dataclasses import replace

    from fixtures.synthetic import synthetic_walk
    from gaitscreen.calibration.setup import CameraSetup
    from gaitscreen.features.session import analyse
    from gaitscreen.pipeline import extraction_from_raw

    raw, _ = synthetic_walk(fps=60.0, n_strides=12, speed_px_s=180.0, width=1280)
    setup = CameraSetup(coverage_m=6.4, calibration_id="x", created_at="now")
    calibration = setup.calibration_for(raw.video.width, raw.video.height)
    extraction = extraction_from_raw(raw, cfg, calibration=calibration)
    # The synthetic clip has no background to track, so camera motion is
    # undeterminable; speed then rests on the subject's own travel.
    speed = analyse(extraction, cfg).metrics.gait_speed_mps

    assert speed is not None
    assert speed == pytest.approx(0.9, rel=0.1)


# --------------------------------------------------------------------------
# history summary
# --------------------------------------------------------------------------
def _history(values: dict[str, list], **extra) -> pd.DataFrame:
    n = len(next(iter(values.values())))
    frame = pd.DataFrame({
        "session_id": [f"s{i}" for i in range(n)],
        "session_date": pd.date_range("2026-01-01", periods=n, freq="14D"),
        "low_confidence": [0] * n,
        "algo_version": ["0.5.0"] * n,
        **values,
    })
    for key, column in extra.items():
        frame[key] = column
    return frame


def _flag(metric, severity="moderate", trigger="absolute", confirmed=False):
    return Flag(code=f"{metric}_{trigger}", metric=metric, severity=severity,
                trigger=trigger, message=f"{metric} flagged", confirmed=confirmed)


def test_empty_history(cfg):
    summary = summarise_history(pd.DataFrame(), {}, cfg)
    assert summary.n_sessions == 0
    assert summary.tone == "insufficient"


def test_counts_and_dates(cfg):
    summary = summarise_history(_history({"gait_speed_mps": [1.1, 1.0, 1.05]}), {}, cfg)
    assert summary.n_sessions == 3
    assert summary.first_date == "2026-01-01"
    assert summary.last_date == "2026-01-29"


def test_a_concern_in_the_latest_walk_leads(cfg):
    history = _history({"gait_speed_mps": [1.1, 1.0, 0.5]})
    flags = {"s2": [_flag("gait_speed_mps", "high")]}
    summary = summarise_history(history, flags, cfg)

    assert summary.tone == "attention"
    assert "walking speed" in summary.headline.lower()
    assert [c.metric for c in summary.active_concerns] == ["gait_speed_mps"]


def test_an_old_concern_is_kept_but_not_active(cfg):
    """A flag that has cleared is still history worth seeing, not a current alarm."""
    history = _history({"gait_speed_mps": [0.5, 1.0, 1.05]})
    flags = {"s0": [_flag("gait_speed_mps", "high")]}
    summary = summarise_history(history, flags, cfg)

    assert summary.active_concerns == []
    assert [c.metric for c in summary.earlier_concerns] == ["gait_speed_mps"]
    assert summary.tone == "good"


def test_persistence_is_counted(cfg):
    history = _history({"stride_time_cv_pct": [4.0, 4.2, 4.1]})
    flags = {sid: [_flag("stride_time_cv_pct")] for sid in ("s0", "s1", "s2")}
    concern = summarise_history(history, flags, cfg).concerns[0]

    assert concern.sessions_flagged == 3
    assert concern.first_seen == "2026-01-01"
    assert concern.in_latest


def test_recording_quality_flags_are_not_concerns_about_the_person(cfg):
    history = _history({"gait_speed_mps": [1.0]})
    flags = {"s0": [_flag("quality_score", trigger="quality")]}
    assert summarise_history(history, flags, cfg).concerns == []


def test_movement_respects_the_direction_of_concern(cfg):
    """Slower walking is worse; more variable timing is worse."""
    history = _history({
        "gait_speed_mps": [1.20, 1.20, 1.00],
        "stride_time_cv_pct": [3.0, 3.0, 1.5],
    })
    trends = {t.metric: t for t in summarise_history(history, {}, cfg).latest}

    assert trends["gait_speed_mps"].movement == "worse"
    assert trends["stride_time_cv_pct"].movement == "better"


def test_a_change_below_the_detectable_minimum_is_steady(cfg):
    """A wobble the tool cannot resolve must not be shown as a rise or fall."""
    history = _history({"gait_speed_mps": [1.20, 1.21, 1.18]})
    trend = summarise_history(history, {}, cfg).latest[0]
    assert trend.movement == "steady"


def test_the_baseline_is_used_once_there_is_one(cfg):
    speeds = [1.10, 1.12, 1.09, 1.11, 1.10, 1.11, 0.95]
    trend = summarise_history(_history({"gait_speed_mps": speeds}), {}, cfg).latest[0]
    assert trend.reference_kind == "baseline"
    assert trend.reference == pytest.approx(np.median(speeds[:-1]), abs=0.01)


def test_recurring_recording_problems_are_reported(cfg):
    problem = json.dumps([{"code": "subject_too_small", "title": "Subject is too small"}])
    history = _history({"gait_speed_mps": [1.0, 1.0, 1.0, 1.0]},
                       recording_diagnostics_json=[problem, problem, problem, None])
    issues = summarise_history(history, {}, cfg).recording_issues
    assert issues == [("Subject is too small", 3)]


def test_mixed_tool_versions_are_mentioned(cfg):
    history = _history({"gait_speed_mps": [1.0, 1.0]},
                       algo_version=["0.4.0", "0.5.0"])
    assert any("different versions" in n for n in summarise_history(history, {}, cfg).notes)
