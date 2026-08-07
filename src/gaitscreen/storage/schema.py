"""SQLite schema for per-user session history.

Built first, because everything else writes to it.

Design split, and why:

* **Scalars live in SQL.** The six core metrics get real, indexed columns so
  trend queries are ordinary SQL rather than JSON unpacking.
* **Arrays live on disk.** Joint-angle curves (6 joints x 101 points x N cycles)
  and raw landmark trajectories go to a per-session artifact directory as
  Parquet/NPZ, referenced by ``artifacts_dir``. Stuffing them into a row makes
  both reprocessing and trend queries awkward, and bloats the DB that the
  reporting layer scans on every run.
* **Raw landmarks are retained.** A longitudinal tool must be able to re-derive
  old sessions when the algorithm improves; otherwise an upgrade introduces a
  step change in the numbers that is indistinguishable from real decline. Hence
  ``algo_version`` on every row and a reprocess path.
* **Nullable metrics.** Every metric column is nullable and paired with
  ``metrics_unavailable`` (JSON: metric -> reason). A metric that could not be
  measured stays NULL; it is never defaulted to a plausible-looking number.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT,
    created_at   TEXT NOT NULL,
    notes        TEXT
);

-- One row per (user, camera setup). Reused across sessions rather than
-- recalibrated each time, but validated per session for drift: a moved tripod
-- silently rescales every distance-based metric.
CREATE TABLE IF NOT EXISTS calibrations (
    calibration_id   TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at       TEXT NOT NULL,
    method           TEXT NOT NULL,          -- 'two_point' | 'homography'
    frame_width      INTEGER NOT NULL,
    frame_height     INTEGER NOT NULL,
    reference_distance_m REAL,
    scale_m_per_px   REAL,                   -- two_point only
    homography_json  TEXT,                   -- homography only, 3x3 row-major
    reference_points_json TEXT NOT NULL,
    -- Drift check baseline: subject pixel height at calibration time.
    expected_subject_px_height REAL,
    is_active        INTEGER NOT NULL DEFAULT 1,
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS idx_calibrations_user
    ON calibrations(user_id, is_active);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_date   TEXT NOT NULL,            -- ISO date of the recording
    created_at     TEXT NOT NULL,            -- ISO timestamp of processing
    algo_version   TEXT NOT NULL,
    calibration_id TEXT REFERENCES calibrations(calibration_id),

    video_source         TEXT NOT NULL,
    frontal_video_source TEXT,
    video_fps      REAL,
    video_width    INTEGER,
    video_height   INTEGER,
    camera_side    TEXT,                     -- body side facing the camera
    assistive_device TEXT,                   -- operator-entered, authoritative

    -- Core metrics (all nullable; see metrics_unavailable).
    gait_speed_mps            REAL,
    stride_time_cv_pct        REAL,
    step_length_asymmetry_pct REAL,
    step_time_asymmetry_pct   REAL,
    cadence_spm               REAL,
    double_support_pct        REAL,
    trunk_ap_sway_norm        REAL,

    stride_time_mean_s REAL,
    stride_time_sd_s   REAL,
    n_strides_total    INTEGER NOT NULL DEFAULT 0,
    n_strides_valid    INTEGER NOT NULL DEFAULT 0,
    n_passes           INTEGER NOT NULL DEFAULT 0,

    quality_score   REAL,
    low_confidence  INTEGER NOT NULL DEFAULT 0,
    quality_components_json TEXT,
    quality_notes_json      TEXT,

    metrics_unavailable_json      TEXT,
    low_confidence_metrics_json   TEXT,
    speed_feasibility_json        TEXT,

    artifacts_dir TEXT,
    notes         TEXT,

    UNIQUE(user_id, session_date, video_source)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_date
    ON sessions(user_id, session_date);
CREATE INDEX IF NOT EXISTS idx_sessions_algo
    ON sessions(algo_version);

CREATE TABLE IF NOT EXISTS session_flags (
    flag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    code        TEXT NOT NULL,
    metric      TEXT NOT NULL,
    severity    TEXT NOT NULL,               -- 'moderate' | 'high'
    trigger     TEXT NOT NULL,               -- 'absolute' | 'trend' | 'quality'
    message     TEXT NOT NULL,               -- plain language, for caregivers
    detail_json TEXT,                        -- raw numbers, for clinicians
    confirmed   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_flags_session ON session_flags(session_id);

-- Gait events, kept for auditability: a caregiver questioning a flag should be
-- able to see which strides it was computed from.
CREATE TABLE IF NOT EXISTS session_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    t          REAL NOT NULL,
    kind       TEXT NOT NULL,
    side       TEXT NOT NULL,
    frame      INTEGER NOT NULL,
    confidence REAL,
    method     TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_session ON session_events(session_id);
"""


def connect(db_path: str | Path, *, create: bool = True) -> sqlite3.Connection:
    """Open the session database, creating and migrating the schema as needed."""
    db_path = Path(db_path)
    if create:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    elif not db_path.exists():
        raise FileNotFoundError(f"no gaitscreen database at {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    if create:
        initialise(conn)
    return conn


def initialise(conn: sqlite3.Connection) -> None:
    """Apply the DDL and record the schema version."""
    with conn:
        conn.executescript(_DDL)
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    return int(row["value"]) if row else 0
