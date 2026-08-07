"""Storage: round-trips, reprocessing idempotency, and baseline hygiene."""
import numpy as np
import pytest

from gaitscreen.calibration.model import Calibration
from gaitscreen.storage.artifacts import SessionArtifacts
from gaitscreen.storage.repository import (SessionRecord, SessionRepository,
                                           make_session_id)
from gaitscreen.storage.schema import SCHEMA_VERSION, schema_version
from gaitscreen.types import Flag, GaitEvent, QualityReport, SessionMetrics
from fixtures.synthetic import synthetic_walk


@pytest.fixture()
def repository(tmp_path):
    with SessionRepository(tmp_path / "gait.db") as repo:
        yield repo


def _record(user="u1", date="2026-01-15", video="walk1.mp4", algo="0.1.0", **kwargs):
    metrics = SessionMetrics(
        stride_time_cv_pct=3.2, cadence_spm=104.0, n_strides_valid=11, n_strides_total=12,
    )
    metrics.mark_unavailable("gait_speed_mps", "no forward travel in the recording")
    quality = QualityReport(score=0.72, low_confidence=False, components={"frame_rate": 0.5})
    return SessionRecord(
        user_id=user, session_date=date, video_source=video, algo_version=algo,
        metrics=metrics, quality=quality, **kwargs,
    )


def test_schema_version_recorded(repository):
    assert schema_version(repository.conn) == SCHEMA_VERSION


def test_session_roundtrip_preserves_nulls_and_reasons(repository):
    session_id = repository.upsert_session(_record())
    row = repository.get_session(session_id)

    assert row["gait_speed_mps"] is None, "an unmeasurable metric must stay NULL"
    assert "no forward travel" in row["metrics_unavailable_json"]
    assert row["stride_time_cv_pct"] == 3.2


def test_reprocessing_updates_rather_than_duplicates(repository):
    """Re-deriving a session under a new algorithm version must not add a row.

    A duplicate would show up in the trend as a second visit on the same day.
    """
    first = repository.upsert_session(_record(algo="0.1.0"))

    updated = _record(algo="0.2.0")
    updated.metrics.stride_time_cv_pct = 4.4
    second = repository.upsert_session(updated)

    assert first == second == make_session_id("u1", "2026-01-15", "walk1.mp4")
    frame = repository.sessions_for_user("u1")
    assert len(frame) == 1
    assert frame.iloc[0]["algo_version"] == "0.2.0"
    assert frame.iloc[0]["stride_time_cv_pct"] == 4.4


def test_mixed_algorithm_versions_are_visible(repository):
    repository.upsert_session(_record(date="2026-01-01", video="a.mp4", algo="0.1.0"))
    repository.upsert_session(_record(date="2026-02-01", video="b.mp4", algo="0.2.0"))
    assert repository.algo_versions_for_user("u1") == ["0.1.0", "0.2.0"]


def test_flags_are_replaced_not_accumulated_on_reprocess(repository):
    session_id = repository.upsert_session(_record())
    flag = Flag(
        code="speed_below_absolute", metric="gait_speed_mps", severity="high",
        trigger="absolute", message="walking speed is below the high-risk threshold",
        detail={"value": 0.55, "threshold": 0.6},
    )
    repository.save_flags(session_id, [flag])
    repository.save_flags(session_id, [flag])

    stored = repository.flags_for_session(session_id)
    assert len(stored) == 1
    assert stored[0].detail["value"] == 0.55


def test_events_are_stored_for_auditability(repository):
    session_id = repository.upsert_session(_record())
    events = [
        GaitEvent(t=0.52, kind="heel_strike", side="left", frame=13),
        GaitEvent(t=1.06, kind="toe_off", side="right", frame=26, confidence=0.8),
    ]
    repository.save_events(session_id, events)
    rows = repository.conn.execute(
        "SELECT * FROM session_events WHERE session_id=? ORDER BY t", (session_id,)
    ).fetchall()
    assert [r["kind"] for r in rows] == ["heel_strike", "toe_off"]


def test_baseline_query_excludes_the_session_being_evaluated(repository):
    """``before_date`` is exclusive, so a session cannot enter its own baseline."""
    for day in range(1, 6):
        repository.upsert_session(
            _record(date=f"2026-01-0{day}", video=f"walk{day}.mp4")
        )
    history = repository.sessions_for_user("u1", before_date="2026-01-04")
    assert len(history) == 3


def test_low_confidence_sessions_can_be_excluded(repository):
    good = _record(date="2026-01-01", video="good.mp4")
    bad = _record(date="2026-01-02", video="bad.mp4")
    bad.quality = QualityReport(score=0.3, low_confidence=True)
    repository.upsert_session(good)
    repository.upsert_session(bad)

    assert len(repository.sessions_for_user("u1")) == 2
    assert len(repository.sessions_for_user("u1", exclude_low_confidence=True)) == 1


def test_calibration_persistence_and_activation(repository):
    first = Calibration.from_two_points("u1", (0, 400), (500, 400), 2.0, 1280, 720)
    repository.save_calibration(first)
    second = Calibration.from_two_points("u1", (0, 400), (400, 400), 2.0, 1280, 720)
    repository.save_calibration(second)

    active = repository.active_calibration("u1")
    assert active.calibration_id == second.calibration_id, "newest must become active"
    assert repository.get_calibration(first.calibration_id).is_active is False


def test_raw_landmark_artifact_roundtrip(tmp_path, cfg):
    """Reprocessing depends on this being lossless."""
    raw, _ = synthetic_walk(n_strides=6, fps=60.0)
    artifacts = SessionArtifacts(tmp_path, "u1", "sess1")
    artifacts.save_meta({
        "video": {
            "path": "synthetic.mp4", "width": raw.video.width,
            "height": raw.video.height, "fps": raw.video.fps, "warnings": [],
        }
    })
    artifacts.save_raw_landmarks(raw)

    restored = artifacts.load_raw_landmarks()
    np.testing.assert_allclose(restored.xy, raw.xy)
    np.testing.assert_allclose(restored.visibility, raw.visibility)
    np.testing.assert_allclose(restored.t, raw.t)
    assert restored.detected.tolist() == raw.detected.tolist()
    assert restored.video.fps == raw.video.fps


def test_angle_curve_artifact_roundtrip(tmp_path):
    from gaitscreen.types import AngleCurves

    curves = AngleCurves(
        percent=np.linspace(0, 100, 101),
        curves={"left_knee": np.random.default_rng(1).normal(size=(7, 101))},
    )
    artifacts = SessionArtifacts(tmp_path, "u1", "sess1")
    artifacts.save_angle_curves(curves)

    restored = artifacts.load_angle_curves()
    np.testing.assert_allclose(restored.curves["left_knee"], curves.curves["left_knee"])
    assert restored.mean_curve("left_knee").shape == (101,)
