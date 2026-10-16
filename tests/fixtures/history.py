"""A small, believable stored history, for exercising the app with data in it.

Most app tests run against an empty database, which only ever reaches the
"nothing saved yet" states. These seed one person whose walking slows over a
few months with a flag on the latest walk, one person with just two walks, and
a camera setup -- enough for every section of the trends page to have
something to show.
"""
from __future__ import annotations

import json

from gaitscreen.calibration.setup import save_setup
from gaitscreen.storage.repository import SessionRecord
from gaitscreen.types import Flag, QualityReport, SessionMetrics

DECLINING = "Ada Example"
NEW_PERSON = "Ravi Example"

_SMALL_SUBJECT = json.dumps([{
    "code": "subject_too_small", "severity": "major",
    "title": "Subject is too small in the frame",
}])


def _session(user, date, speed, cv, asym, cadence, ds, *, low=False, diagnostics=None):
    metrics = SessionMetrics(
        gait_speed_mps=speed, stride_time_cv_pct=cv, step_length_asymmetry_pct=asym,
        cadence_spm=cadence, double_support_pct=ds, trunk_ap_sway_norm=0.06,
        stride_time_mean_s=120.0 / cadence, n_strides_total=14, n_strides_valid=11,
        n_passes=3,
    )
    return SessionRecord(
        user_id=user, session_date=date, video_source=f"{user}_{date}.mp4",
        algo_version="0.5.0", metrics=metrics,
        quality=QualityReport(score=0.55 if low else 0.86, low_confidence=low),
        recording_diagnostics=json.loads(diagnostics) if diagnostics else None,
        view_kind="sagittal",
    )


def seed_history(repository, *, with_setup: bool = True) -> None:
    if with_setup:
        save_setup(repository, 4.5, notes="Garage, tripod by the door")

    declining = [
        ("2026-06-02", 1.12, 2.1, 3.0, 108, 22.0, False),
        ("2026-06-30", 1.10, 2.3, 3.4, 107, 22.5, False),
        ("2026-07-28", 1.11, 2.2, 2.9, 108, 22.1, False),
        ("2026-08-25", 1.08, 2.6, 3.8, 106, 23.0, True),
        ("2026-09-22", 1.05, 2.9, 4.2, 105, 23.4, False),
        ("2026-10-06", 0.92, 3.6, 4.6, 101, 25.1, False),
    ]
    for i, (date, speed, cv, asym, cad, ds, low) in enumerate(declining):
        record = _session(DECLINING, date, speed, cv, asym, cad, ds, low=low,
                          diagnostics=_SMALL_SUBJECT if i % 2 == 0 else None)
        sid = repository.upsert_session(record)
        if date == "2026-10-06":
            repository.save_flags(sid, [
                Flag(code="gait_speed_mps_trend", metric="gait_speed_mps",
                     severity="high", trigger="trend", confirmed=True,
                     message="Walking speed has dropped well below this person's "
                             "usual range."),
                Flag(code="stride_time_cv_pct_absolute", metric="stride_time_cv_pct",
                     severity="moderate", trigger="absolute",
                     message="Step timing is more irregular than the screening "
                             "threshold."),
            ])
        if date == "2026-08-25":
            repository.save_flags(sid, [
                Flag(code="step_length_asymmetry_pct_absolute",
                     metric="step_length_asymmetry_pct", severity="moderate",
                     trigger="absolute",
                     message="The two legs' step lengths differed more than usual."),
            ])

    for date, speed in (("2026-09-29", 1.21), ("2026-10-07", 1.19)):
        repository.upsert_session(_session(NEW_PERSON, date, speed, 1.8, 2.0, 112, 21.0))
