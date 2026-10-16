"""Read/write access to the session database.

Session IDs are derived deterministically from ``(user_id, session_date,
video basename)``. That makes reprocessing idempotent: re-running a session under
a new algorithm version updates the existing row rather than adding a duplicate
that would appear in the trend as a second visit on the same day.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from ..calibration.model import Calibration
from ..types import CORE_METRICS, Flag, GaitEvent, QualityReport, SessionMetrics
from . import schema

_METRIC_COLUMNS = (
    "gait_speed_mps",
    "stride_time_cv_pct",
    "step_length_asymmetry_pct",
    "step_time_asymmetry_pct",
    "cadence_spm",
    "double_support_pct",
    "trunk_ap_sway_norm",
    "step_width_norm",
    "trunk_lateral_sway_norm",
    "stride_time_mean_s",
    "stride_time_sd_s",
)


def make_session_id(user_id: str, session_date: str, video_source: str) -> str:
    key = f"{user_id}|{session_date}|{Path(video_source).name}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class SessionRecord:
    """Everything written to the ``sessions`` table for one session."""

    user_id: str
    session_date: str
    video_source: str
    algo_version: str
    metrics: SessionMetrics
    quality: QualityReport
    session_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    calibration_id: Optional[str] = None
    frontal_video_source: Optional[str] = None
    video_fps: Optional[float] = None
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    camera_side: Optional[str] = None
    view_kind: Optional[str] = None
    assistive_device: Optional[str] = None
    speed_feasibility: Optional[dict] = None
    recording_diagnostics: Optional[list] = None
    artifacts_dir: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = make_session_id(
                self.user_id, self.session_date, self.video_source
            )


#: Errors that mean the server forgot this connection's stream, not that the
#: query was wrong. Turso closes idle Hrana streams, and a cached connection
#: outlives them.
_STREAM_GONE = ("stream not found", "stream expired", "stream is closed")


class _ReconnectingConnection:
    """A connection that reopens itself when its remote stream has expired.

    Transparent for local SQLite, which never raises these. Only ``execute``,
    ``executemany`` and ``commit`` are proxied, which is everything the
    repository calls.
    """

    def __init__(self, factory):
        self._factory = factory
        self._conn, self.backend, self.note = factory()

    def _call(self, name, *args, **kwargs):
        try:
            return getattr(self._conn, name)(*args, **kwargs)
        except Exception as exc:
            if not any(s in str(exc).lower() for s in _STREAM_GONE):
                raise
            # The stream is gone, not the database. Reopen and try once more;
            # a second failure is a real error and propagates.
            try:
                self._conn.close()
            except Exception:
                pass
            conn, backend, note = self._factory()
            if backend != self.backend:
                # get_connection() falls back to local SQLite when Turso is
                # unreachable. Accepting that here would move the data to a
                # different database without anyone noticing.
                try:
                    conn.close()
                except Exception:
                    pass
                raise RuntimeError(
                    f"storage backend changed from {self.backend!r} to "
                    f"{backend!r} while reconnecting ({note}); refusing to "
                    f"read or write a different database"
                ) from exc
            self._conn, self.note = conn, note
            return getattr(self._conn, name)(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self._call("execute", *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._call("executemany", *args, **kwargs)

    def commit(self, *args, **kwargs):
        return self._call("commit", *args, **kwargs)

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


class SessionRepository:
    """Thin data-access layer over the SQLite schema."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        # Turso if configured (persistent), else local SQLite (ephemeral on Cloud).
        # get_connection returns (conn, backend, note).
        self.conn = _ReconnectingConnection(
            lambda: schema.get_connection(self.db_path)
        )
        self.backend = self.conn.backend
        self.backend_note = self.conn.note

    def _one(self, query, params=()):
        """Fetch one row as a dict (works for sqlite3.Row and libsql tuple)."""
        cur = self.conn.execute(query, tuple(params))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], tuple(row)))

    def _frame(self, query, params=()):
        """Fetch all rows as a DataFrame (works for both backends)."""
        import pandas as pd
        cur = self.conn.execute(query, tuple(params))
        cols = [d[0] for d in cur.description]
        return pd.DataFrame([tuple(r) for r in cur.fetchall()], columns=cols)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SessionRepository":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- users -----------------------------------------------------------
    def ensure_user(
        self, user_id: str, display_name: Optional[str] = None, notes: Optional[str] = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO users(user_id, display_name, created_at, notes) "
            "VALUES(?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
            "display_name=COALESCE(excluded.display_name, users.display_name)",
            (
                user_id,
                display_name,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                notes,
            ),
        )

        self.conn.commit()
    def list_users(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT user_id FROM users ORDER BY user_id", ()
        ).fetchall()
        return [r[0] for r in rows]

    # -- calibration -----------------------------------------------------
    def save_calibration(self, calibration: Calibration, *, make_active: bool = True) -> str:
        self.ensure_user(calibration.user_id)
        row = calibration.to_row()
        if make_active:
            self.conn.execute(
                "UPDATE calibrations SET is_active=0 WHERE user_id=?",
                (calibration.user_id,),
            )
            row["is_active"] = 1
        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        self.conn.execute(
            f"INSERT INTO calibrations({columns}) VALUES({placeholders})",
            tuple(row.values()),
        )
        self.conn.commit()
        return calibration.calibration_id

    def active_calibration(self, user_id: str) -> Optional[Calibration]:
        row = self._one(
            "SELECT * FROM calibrations WHERE user_id=? AND is_active=1 "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        )
        return Calibration.from_row(row) if row else None

    def get_calibration(self, calibration_id: str) -> Optional[Calibration]:
        row = self._one(
            "SELECT * FROM calibrations WHERE calibration_id=?", (calibration_id,)
        )
        return Calibration.from_row(row) if row else None

    # -- sessions --------------------------------------------------------
    def upsert_session(self, record: SessionRecord) -> str:
        self.ensure_user(record.user_id)
        metrics, quality = record.metrics, record.quality

        row: dict = {
            "session_id": record.session_id,
            "user_id": record.user_id,
            "session_date": record.session_date,
            "created_at": record.created_at,
            "algo_version": record.algo_version,
            "calibration_id": record.calibration_id,
            "video_source": str(record.video_source),
            "frontal_video_source": (
                str(record.frontal_video_source) if record.frontal_video_source else None
            ),
            "video_fps": record.video_fps,
            "video_width": record.video_width,
            "video_height": record.video_height,
            "camera_side": record.camera_side,
            "view_kind": record.view_kind,
            "assistive_device": record.assistive_device,
            "n_strides_total": metrics.n_strides_total,
            "n_strides_valid": metrics.n_strides_valid,
            "n_passes": metrics.n_passes,
            "quality_score": quality.score,
            "low_confidence": int(quality.low_confidence),
            "quality_components_json": json.dumps(quality.components),
            "quality_notes_json": json.dumps(quality.notes),
            "metrics_unavailable_json": json.dumps(metrics.unavailable),
            "low_confidence_metrics_json": json.dumps(metrics.low_confidence_metrics),
            "speed_feasibility_json": (
                json.dumps(record.speed_feasibility) if record.speed_feasibility else None
            ),
            "recording_diagnostics_json": (
                json.dumps(record.recording_diagnostics)
                if record.recording_diagnostics else None
            ),
            "artifacts_dir": str(record.artifacts_dir) if record.artifacts_dir else None,
            "notes": record.notes,
        }
        for column in _METRIC_COLUMNS:
            row[column] = getattr(metrics, column, None)

        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "session_id")
        self.conn.execute(
            f"INSERT INTO sessions({columns}) VALUES({placeholders}) "
            f"ON CONFLICT(session_id) DO UPDATE SET {updates}",
            tuple(row.values()),
        )
        self.conn.commit()
        return record.session_id

    def save_flags(self, session_id: str, flags: Sequence[Flag]) -> None:
        """Replace this session's flags (reprocessing must not accumulate them)."""
        self.conn.execute("DELETE FROM session_flags WHERE session_id=?", (session_id,))
        self.conn.executemany(
            "INSERT INTO session_flags(session_id, code, metric, severity, "
            "trigger, message, detail_json, confirmed) VALUES(?,?,?,?,?,?,?,?)",
            tuple(
                (
                    session_id, f.code, f.metric, f.severity, f.trigger, f.message,
                    json.dumps(f.detail), int(f.confirmed),
                )
                for f in flags
            ),
        )

        self.conn.commit()
    def save_events(self, session_id: str, events: Iterable[GaitEvent]) -> None:
        self.conn.execute("DELETE FROM session_events WHERE session_id=?", (session_id,))
        self.conn.executemany(
            "INSERT INTO session_events(session_id, t, kind, side, frame, "
            "confidence, method) VALUES(?,?,?,?,?,?,?)",
            tuple(
                (session_id, e.t, e.kind, e.side, e.frame, e.confidence, e.method)
                for e in events
            ),
        )

        self.conn.commit()
    def flags_for_session(self, session_id: str) -> list[Flag]:
        rows = self.conn.execute(
            "SELECT * FROM session_flags WHERE session_id=? ORDER BY severity DESC, code",
            (session_id,),
        ).fetchall()
        return [
            Flag(
                code=r[2], metric=r[3], severity=r[4],
                trigger=r[5], message=r[6],
                detail=json.loads(r[7] or "{}"),
                confirmed=bool(r[8]),
            )
            for r in rows
        ]

    def get_session(self, session_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()

    def sessions_for_user(
        self,
        user_id: str,
        *,
        before_date: Optional[str] = None,
        exclude_low_confidence: bool = False,
        algo_version: Optional[str] = None,
    ) -> pd.DataFrame:
        """Session history as a DataFrame, oldest first.

        ``before_date`` is exclusive, so a baseline can be built from strictly
        prior sessions without leaking the session being evaluated into its own
        reference distribution.
        """
        query = "SELECT * FROM sessions WHERE user_id=?"
        params: list = [user_id]
        if before_date is not None:
            query += " AND session_date < ?"
            params.append(before_date)
        if exclude_low_confidence:
            query += " AND low_confidence=0"
        if algo_version is not None:
            query += " AND algo_version=?"
            params.append(algo_version)
        query += " ORDER BY session_date ASC, created_at ASC"

        frame = self._frame(query, params)
        if not frame.empty:
            frame["session_date"] = pd.to_datetime(frame["session_date"])
        return frame

    def algo_versions_for_user(self, user_id: str) -> list[str]:
        """Distinct algorithm versions present in a user's history.

        More than one means the trend mixes yardsticks and should be reprocessed.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT algo_version FROM sessions WHERE user_id=? "
            "ORDER BY algo_version",
            (user_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def delete_session(self, session_id: str) -> None:
        self.conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))

        self.conn.commit()
    # -- convenience -----------------------------------------------------
    @staticmethod
    def metric_columns() -> tuple[str, ...]:
        return CORE_METRICS

    def summary(self, user_id: str) -> dict:
        frame = self.sessions_for_user(user_id)
        return {
            "user_id": user_id,
            "n_sessions": len(frame),
            "date_range": (
                (str(frame["session_date"].min().date()), str(frame["session_date"].max().date()))
                if not frame.empty else None
            ),
            "algo_versions": self.algo_versions_for_user(user_id),
            "n_low_confidence": int(frame["low_confidence"].sum()) if not frame.empty else 0,
        }


def record_to_dict(record: SessionRecord) -> dict:  # pragma: no cover - debugging aid
    return asdict(record)
