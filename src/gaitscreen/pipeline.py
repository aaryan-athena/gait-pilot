"""Pipeline stages, composed end to end.

Stage 1 -- :func:`extract_session`: ingestion, calibration validation, landmark
extraction, smoothing, and an explicit account of what the resulting
trajectories can and cannot support.

Stage 2 -- :func:`analyse_video`: segmentation, feature extraction, quality
scoring and flagging, producing a complete :class:`SessionResult`.

Stage 3 -- :func:`persist_session`: writes the result to the database and the
per-session artifact directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path
from typing import Optional

import pandas as pd

from .calibration.model import Calibration
from .calibration.verify import CalibrationCheck
from .calibration import verify as calibration_verify
from .config import Config
from .features.session import SessionAnalysis, analyse
from .flagging.engine import FlagResult, evaluate as evaluate_flags
from .io import video as video_io
from .io.video import CameraMotion
from .pose import landmarker
from .pose.to_pixels import subject_pixel_height, to_pixels
from .signal import filters, resample
from .quality import feasibility
from .quality.score import score_session
from .storage.artifacts import SessionArtifacts
from .storage.repository import SessionRecord, SessionRepository, make_session_id
from .types import (PixelSeries, QualityReport, RawLandmarks, SpeedFeasibility,
                    VideoInfo)
from .version import ALGO_VERSION


@dataclass
class Extraction:
    """Everything stage 1 produces for one video."""

    info: VideoInfo
    raw: RawLandmarks
    series: PixelSeries  # filtered, gap-filled, isotropic pixels, y-up
    segments: list[tuple[int, int]]
    camera_motion: CameraMotion
    speed: SpeedFeasibility
    calibration: Optional[Calibration]
    calibration_check: CalibrationCheck
    gap_summary: dict[str, int]
    subject_px_height: float

    @property
    def warnings(self) -> list[str]:
        """All ingestion-level caveats, in the order a user should read them."""
        out = list(self.info.warnings)
        if self.speed.reason:
            out.append(self.speed.reason)
        if self.calibration_check.reason:
            out.append(self.calibration_check.reason)
        if self.camera_motion.note:
            out.append(self.camera_motion.note)
        if not self.segments:
            out.append(
                "no analysis segment long enough to contain a full gait cycle was "
                "found; landmark tracking may be failing"
            )
        return out

    @property
    def usable_frames(self) -> int:
        return sum(stop - start for start, stop in self.segments)


def extract_session(
    video_path: str | Path,
    cfg: Config,
    *,
    calibration: Optional[Calibration] = None,
    project_root: str | Path = ".",
    progress: Optional[callable] = None,
    check_camera_motion: bool = True,
) -> Extraction:
    """Run stage 1 over one video file.

    Order matters here. Pose extraction comes before camera-motion estimation
    because the subject's bounding box is needed to mask the subject out of the
    background feature tracking -- otherwise the subject's own limbs dominate the
    flow field and every recording looks like a moving camera.
    """
    info = video_io.probe(video_path, cfg)
    raw = landmarker.extract(info, cfg, project_root=project_root, progress=progress)

    if check_camera_motion:
        motion = video_io.estimate_camera_motion(
            info.path, cfg, landmarker.subject_boxes(raw),
            max_frames=int(cfg.get("video.max_frames", 0) or 0),
        )
    else:
        motion = CameraMotion(
            net_px=0.0, path_px=0.0, determinate=False, n_samples=0,
            note="camera-motion check skipped",
        )

    series = to_pixels(raw, cfg)
    series, gap_summary = resample.fill_short_gaps(series, cfg)
    series = filters.smooth_series(series, cfg)
    segments = resample.analysis_segments(series, cfg)

    speed = feasibility.assess_speed(series, motion, cfg)
    check = calibration_verify.check(calibration, series)

    return Extraction(
        info=info,
        raw=raw,
        series=series,
        segments=segments,
        camera_motion=motion,
        speed=speed,
        calibration=calibration,
        calibration_check=check,
        gap_summary=gap_summary,
        subject_px_height=subject_pixel_height(series),
    )


# --------------------------------------------------------------------------
# Stage 2: full session analysis
# --------------------------------------------------------------------------
@dataclass
class SessionResult:
    """A completely analysed session, ready to display or persist."""

    extraction: Extraction
    analysis: SessionAnalysis
    quality: QualityReport
    flags: FlagResult
    user_id: str
    session_date: str
    session_id: str
    algo_version: str = ALGO_VERSION
    assistive_device: Optional[str] = None
    artifacts_dir: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @property
    def metrics(self):
        return self.analysis.metrics

    @property
    def all_notes(self) -> list[str]:
        """Ingestion, analysis and quality caveats, de-duplicated in order."""
        seen: set[str] = set()
        out: list[str] = []
        for note in [*self.extraction.warnings, *self.analysis.notes,
                     *self.quality.notes, *self.notes]:
            if note and note not in seen:
                seen.add(note)
                out.append(note)
        return out


def analyse_video(
    video_path: str | Path,
    cfg: Config,
    *,
    user_id: str,
    session_date: Optional[str] = None,
    calibration: Optional[Calibration] = None,
    declared_device: Optional[str] = None,
    history: Optional[pd.DataFrame] = None,
    project_root: str | Path = ".",
    progress: Optional[callable] = None,
    check_camera_motion: bool = True,
) -> SessionResult:
    """Run the whole pipeline over one video.

    ``history`` must contain only sessions strictly *before* ``session_date``;
    a session that contributes to its own baseline can never deviate from it.
    Pass ``None`` to skip the personal-trend checks entirely (absolute
    thresholds still apply).
    """
    session_date = session_date or date_type.today().isoformat()

    extraction = extract_session(
        video_path, cfg, calibration=calibration, project_root=project_root,
        progress=progress, check_camera_motion=check_camera_motion,
    )
    analysis = analyse(extraction, cfg)
    quality = score_session(
        extraction.series, extraction.raw, cfg,
        cycles=analysis.cycles, declared_device=declared_device,
    )
    flags = evaluate_flags(
        analysis.metrics, quality,
        history if history is not None else pd.DataFrame(), cfg,
    )

    return SessionResult(
        extraction=extraction,
        analysis=analysis,
        quality=quality,
        flags=flags,
        user_id=user_id,
        session_date=session_date,
        session_id=make_session_id(user_id, session_date, str(extraction.info.path)),
        assistive_device=quality.assistive_device,
        notes=flags.baseline_notes,
    )


# --------------------------------------------------------------------------
# Stage 3: persistence
# --------------------------------------------------------------------------
def persist_session(
    result: SessionResult,
    cfg: Config,
    repository: SessionRepository,
    *,
    project_root: str | Path = ".",
    calibration_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Write metrics, flags, events and array artifacts for one session."""
    artifacts = SessionArtifacts(
        cfg.resolve_path("storage.artifacts_root", project_root),
        result.user_id, result.session_id,
    )
    info = result.extraction.info

    artifacts.save_meta({
        "session_id": result.session_id,
        "user_id": result.user_id,
        "session_date": result.session_date,
        "algo_version": result.algo_version,
        "video": {
            "path": str(info.path), "width": info.width, "height": info.height,
            "fps": info.fps, "n_frames": info.n_frames, "warnings": info.warnings,
        },
        "cycle_summary": result.analysis.cycle_summary,
        "range_of_motion": result.analysis.range_of_motion,
        "camera_side": result.analysis.camera_side,
        "direction_method": result.analysis.direction_method,
        "event_agreement_ms": result.analysis.event_agreement_ms,
        "notes": result.all_notes,
    })
    if cfg["storage.keep_raw_landmarks"]:
        artifacts.save_raw_landmarks(result.extraction.raw)
    artifacts.save_angle_curves(result.analysis.angles)
    result.artifacts_dir = str(artifacts.dir)

    speed = result.extraction.speed
    record = SessionRecord(
        user_id=result.user_id,
        session_date=result.session_date,
        video_source=str(info.path),
        algo_version=result.algo_version,
        metrics=result.analysis.metrics,
        quality=result.quality,
        session_id=result.session_id,
        calibration_id=calibration_id or (
            result.extraction.calibration.calibration_id
            if result.extraction.calibration else None
        ),
        video_fps=info.fps,
        video_width=info.width,
        video_height=info.height,
        camera_side=result.analysis.camera_side,
        assistive_device=result.assistive_device,
        speed_feasibility={
            "feasible": speed.feasible,
            "subject_translation_frac": speed.subject_translation_frac,
            "camera_motion_frac": speed.camera_motion_frac,
            "reason": speed.reason,
        },
        artifacts_dir=result.artifacts_dir,
        notes=notes,
    )
    session_id = repository.upsert_session(record)
    repository.save_flags(session_id, result.flags.flags)
    repository.save_events(session_id, result.analysis.events)
    return session_id
