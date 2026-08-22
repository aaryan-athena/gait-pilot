"""Command-line interface.

Commands::

    gaitscreen probe     VIDEO...            container metadata + quality warnings
    gaitscreen extract   VIDEO...            landmarks, smoothing, feasibility
    gaitscreen analyze   VIDEO... --user U   full pipeline: metrics, quality, flags
    gaitscreen calibrate --user U --video V  one-time pixel-to-metre calibration
    gaitscreen users                         list users
    gaitscreen sessions  --user U            list a user's session history

For pilot testing, ``streamlit run app/main.py`` gives the same pipeline with
upload, charts and trend views.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import gaitscreen
from .calibration import interactive
from .calibration.model import Calibration
from .config import Config
from .pipeline import Extraction, extract_session
from .pose.to_pixels import subject_pixel_height
from .storage.artifacts import SessionArtifacts
from .storage.repository import SessionRepository, make_session_id
from .version import ALGO_VERSION

PROJECT_ROOT = Path(gaitscreen.__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _load_config(args) -> Config:
    return Config.load(args.config)


def _repository(cfg: Config) -> SessionRepository:
    return SessionRepository(cfg.resolve_path("storage.db_path", PROJECT_ROOT))


def _print_warnings(warnings: Sequence[str], prefix: str = "  ! ") -> None:
    for warning in warnings:
        # Wrap long explanatory warnings so they stay readable in a terminal.
        text = " ".join(warning.split())
        first = True
        while text:
            width = 96 if first else 92
            if len(text) <= width:
                chunk, text = text, ""
            else:
                cut = text.rfind(" ", 0, width)
                cut = cut if cut > 0 else width
                chunk, text = text[:cut], text[cut + 1:]
            print(f"{prefix if first else '    '}{chunk}")
            first = False


def _fmt(value: Optional[float], spec: str = ".3f", dash: str = "n/a") -> str:
    return dash if value is None or value != value else format(value, spec)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_probe(args) -> int:
    cfg = _load_config(args)
    from .io import video as video_io

    for path in args.videos:
        info = video_io.probe(path, cfg)
        print(f"\n{info.path.name}")
        print(f"  {info.width}x{info.height} @ {info.fps:g} fps, "
              f"{info.n_frames} frames, {info.duration_s:.1f}s")
        print(f"  frame interval: {info.frame_interval_s * 1000:.1f} ms")
        _print_warnings(info.warnings)
    return 0


def cmd_extract(args) -> int:
    cfg = _load_config(args)
    calibration = None
    repository = None

    if args.user:
        repository = _repository(cfg)
        calibration = repository.active_calibration(args.user)

    try:
        for path in args.videos:
            print(f"\n=== {Path(path).name}")
            extraction = extract_session(
                path, cfg, calibration=calibration, project_root=PROJECT_ROOT,
                check_camera_motion=not args.skip_camera_check,
                progress=(lambda i: print(f"\r  frame {i}", end="", flush=True))
                if args.verbose else None,
            )
            if args.verbose:
                print()
            _report_extraction(extraction)

            if args.save:
                if not args.user:
                    print("  ! --save requires --user", file=sys.stderr)
                    return 2
                _save_extraction(extraction, cfg, repository, args)
    finally:
        if repository is not None:
            repository.close()
    return 0


def _report_extraction(extraction: Extraction) -> None:
    info, raw, series = extraction.info, extraction.raw, extraction.series
    print(f"  video      : {info.width}x{info.height} @ {info.fps:g} fps, "
          f"{info.n_frames} frames")
    print(f"  detection  : {raw.detection_rate:.1%} of frames")
    print(f"  gaps       : {extraction.gap_summary['samples_interpolated']} samples "
          f"interpolated, {extraction.gap_summary['samples_left_missing']} left missing")
    print(f"  segments   : {len(extraction.segments)} usable "
          f"({extraction.usable_frames}/{series.n_frames} frames)")
    print(f"  subject    : {extraction.subject_px_height:.0f} px tall (scale proxy)")

    motion = extraction.camera_motion
    if motion.determinate:
        print(f"  camera     : net background shift {motion.net_px:.1f} px "
              f"({motion.net_px / info.width:.1%} of width)")
    else:
        print("  camera     : motion indeterminate (untrackable background)")

    speed = extraction.speed
    print(f"  subject travel: {speed.subject_translation_px:.0f} px "
          f"({speed.subject_translation_frac:.0%} of frame width)")
    print(f"  gait speed : {'MEASURABLE' if speed.feasible else 'NOT MEASURABLE'}")
    _print_warnings(extraction.warnings)


def _save_extraction(
    extraction: Extraction, cfg: Config, repository: SessionRepository, args
) -> None:
    session_date = args.date or date.today().isoformat()
    video_source = str(extraction.info.path)
    session_id = make_session_id(args.user, session_date, video_source)

    artifacts = SessionArtifacts(
        cfg.resolve_path("storage.artifacts_root", PROJECT_ROOT), args.user, session_id
    )
    artifacts.save_meta({
        "session_id": session_id,
        "user_id": args.user,
        "session_date": session_date,
        "algo_version": ALGO_VERSION,
        "video": {
            "path": video_source,
            "width": extraction.info.width,
            "height": extraction.info.height,
            "fps": extraction.info.fps,
            "n_frames": extraction.info.n_frames,
            "warnings": extraction.info.warnings,
        },
        "speed_feasibility": {
            "feasible": extraction.speed.feasible,
            "subject_translation_frac": extraction.speed.subject_translation_frac,
            "camera_motion_frac": extraction.speed.camera_motion_frac,
            "reason": extraction.speed.reason,
        },
        "gap_summary": extraction.gap_summary,
        "subject_px_height": extraction.subject_px_height,
    })
    if cfg["storage.keep_raw_landmarks"]:
        artifacts.save_raw_landmarks(extraction.raw)
    repository.ensure_user(args.user)
    print(f"  saved      : {artifacts.dir}")


def cmd_calibrate(args) -> int:
    cfg = _load_config(args)
    from .io import video as video_io

    info = video_io.probe(args.video, cfg)

    if args.method == "two_point":
        if args.distance_m is None:
            print("! --distance-m is required for two_point calibration", file=sys.stderr)
            return 2
        points = args.points or interactive.pick_points(
            args.video, 2, frame_index=args.frame,
            instructions=(
                f"Click the two ends of a known {args.distance_m} m reference. "
                "Mark it ALONG the walking path, not across it."
            ),
        )
        calibration = Calibration.from_two_points(
            args.user, points[0], points[1], args.distance_m, info.width, info.height,
        )
    else:
        if not args.floor_points_m:
            print("! --floor-points-m is required for homography calibration",
                  file=sys.stderr)
            return 2
        floor = json.loads(args.floor_points_m)
        points = args.points or interactive.pick_points(
            args.video, 4, frame_index=args.frame,
            instructions="Click the 4 floor points, in the same order as their "
                         "given real-world coordinates.",
        )
        calibration = Calibration.from_floor_points(
            args.user, points, floor, info.width, info.height,
        )

    # Store the subject's pixel height now, so a later session can detect that
    # the camera was moved -- which would otherwise silently rescale every
    # distance-based metric.
    try:
        extraction = extract_session(
            args.video, cfg, project_root=PROJECT_ROOT, check_camera_motion=False,
        )
        calibration.expected_subject_px_height = subject_pixel_height(extraction.series)
        if not extraction.speed.feasible:
            # A metric scale is only useful if this camera setup can actually
            # capture forward travel, so say so at setup time rather than
            # letting every future session fail the same way.
            _print_warnings([
                "the reference video from this setup cannot yield gait speed: "
                + (extraction.speed.reason or "")
                + " The calibration is stored, but check the camera is fixed and "
                "framed so the subject walks across the frame."
            ])
    except Exception as exc:  # noqa: BLE001 - calibration is still usable without it
        print(f"  ! could not measure subject height for the drift check: {exc}")

    with _repository(cfg) as repository:
        repository.save_calibration(calibration)

    print(f"\ncalibration {calibration.calibration_id} saved for user {args.user!r}")
    print(f"  method     : {calibration.method}")
    if calibration.scale_m_per_px:
        print(f"  scale      : {calibration.scale_m_per_px * 1000:.3f} mm/px")
    if calibration.expected_subject_px_height:
        print(f"  subject    : {calibration.expected_subject_px_height:.0f} px tall "
              "(reference for camera-movement checks)")
    if calibration.notes:
        _print_warnings([calibration.notes])
    return 0


def cmd_analyze(args) -> int:
    """Full pipeline: segmentation, features, quality and flagging."""
    cfg = _load_config(args)
    from .pipeline import analyse_video, persist_session

    session_date = args.date or date.today().isoformat()
    repository = _repository(cfg)
    try:
        calibration = repository.active_calibration(args.user)
        for path in args.videos:
            print(f"\n=== {Path(path).name}")
            history = repository.sessions_for_user(args.user, before_date=session_date)
            result = analyse_video(
                path, cfg,
                user_id=args.user,
                session_date=session_date,
                calibration=calibration,
                declared_device=args.device,
                history=history,
                project_root=PROJECT_ROOT,
                check_camera_motion=not args.skip_camera_check,
            )
            _report_session(result)
            if args.save:
                session_id = persist_session(
                    result, cfg, repository, project_root=PROJECT_ROOT
                )
                print(f"  saved     : session {session_id}")
    finally:
        repository.close()
    return 0


def _report_session(result) -> None:
    from .types import CORE_METRICS

    metrics = result.metrics
    analysis = result.analysis
    units = {
        "gait_speed_mps": "m/s", "stride_time_cv_pct": "%",
        "step_length_asymmetry_pct": "%", "cadence_spm": "steps/min",
        "double_support_pct": "% of cycle", "trunk_ap_sway_norm": "x leg length",
    }

    print(f"  quality   : {result.quality.score:.2f}"
          f"{' (LOW CONFIDENCE)' if result.quality.low_confidence else ''}")
    print(f"  strides   : {analysis.cycle_summary.get('n_valid', 0)} valid of "
          f"{analysis.cycle_summary.get('n_total', 0)} across "
          f"{len(analysis.passes)} pass(es)")
    print("  metrics   :")
    for metric in CORE_METRICS:
        value = metrics.value(metric)
        if value is None:
            print(f"    {metric:28s} not measurable")
            _print_warnings([metrics.unavailable.get(metric, "")], prefix="      - ")
        else:
            marker = " *" if metric in metrics.low_confidence_metrics else ""
            print(f"    {metric:28s} {value:8.3f} {units.get(metric, '')}{marker}")

    if result.flags.flags:
        print("  flags     :")
        for flag in result.flags.flags:
            print(f"    [{flag.severity.upper():8s}] {flag.metric} ({flag.trigger})")
            _print_warnings([flag.message], prefix="      - ")
    else:
        print("  flags     : none")

    report = result.diagnostics
    if report.is_clean:
        print("  recording : no problems found with the setup")
    else:
        counts = {}
        for diagnostic in report.diagnostics:
            counts[diagnostic.severity] = counts.get(diagnostic.severity, 0) + 1
        tally = ", ".join(f"{n} {s}" for s, n in counts.items())
        print(f"  recording : {tally}")
        for diagnostic in report.diagnostics:
            print(f"    [{diagnostic.severity.upper():8s}] {diagnostic.title}")
            _print_warnings([diagnostic.detail], prefix="      - ")
            _print_warnings([f"FIX: {diagnostic.fix}"], prefix="      > ")

    if result.all_notes:
        print("  notes     :")
        _print_warnings(result.all_notes, prefix="    ! ")


def cmd_users(args) -> int:
    cfg = _load_config(args)
    with _repository(cfg) as repository:
        users = repository.list_users()
        if not users:
            print("no users yet")
            return 0
        for user_id in users:
            summary = repository.summary(user_id)
            print(f"{user_id}: {summary['n_sessions']} sessions, "
                  f"algo versions {summary['algo_versions'] or ['-']}, "
                  f"{summary['n_low_confidence']} low-confidence")
    return 0


def cmd_sessions(args) -> int:
    cfg = _load_config(args)
    with _repository(cfg) as repository:
        frame = repository.sessions_for_user(args.user)
        if frame.empty:
            print(f"no sessions for user {args.user!r}")
            return 0
        columns = [
            "session_date", "algo_version", "gait_speed_mps", "stride_time_cv_pct",
            "cadence_spm", "n_strides_valid", "quality_score", "low_confidence",
        ]
        print(frame[[c for c in columns if c in frame]].to_string(index=False))

        versions = repository.algo_versions_for_user(args.user)
        if len(versions) > 1:
            _print_warnings([
                f"this history spans {len(versions)} algorithm versions "
                f"({', '.join(versions)}); a change in the algorithm shifts the "
                "numbers and can look like a change in the person. Reprocess the "
                "history under one version before reading the trend."
            ])
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaitscreen",
        description=(
            "Video-based gait screening and trend monitoring. Screening tool, "
            "not a diagnostic instrument."
        ),
    )
    parser.add_argument("--config", help="YAML file merged over config/default.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="show video metadata and warnings")
    probe.add_argument("videos", nargs="+")
    probe.set_defaults(func=cmd_probe)

    extract = subparsers.add_parser(
        "extract", help="extract and smooth landmarks; assess what is measurable"
    )
    extract.add_argument("videos", nargs="+")
    extract.add_argument("--user", help="user id (enables calibration lookup and --save)")
    extract.add_argument("--date", help="session date (ISO); defaults to today")
    extract.add_argument("--save", action="store_true",
                         help="write raw landmarks and metadata to the artifact store")
    extract.add_argument("--skip-camera-check", action="store_true",
                         help="skip background optical-flow camera-motion estimation")
    extract.set_defaults(func=cmd_extract)

    calibrate = subparsers.add_parser("calibrate", help="one-time pixel-to-metre setup")
    calibrate.add_argument("--user", required=True)
    calibrate.add_argument("--video", required=True,
                           help="a video from this camera setup")
    calibrate.add_argument("--method", choices=("two_point", "homography"),
                           default="two_point")
    calibrate.add_argument("--distance-m", type=float,
                           help="real-world distance between the two marked points")
    calibrate.add_argument("--floor-points-m",
                           help='JSON list of 4 [x, y] floor coordinates in metres')
    calibrate.add_argument("--frame", type=int, default=0,
                           help="frame index to click on")
    calibrate.add_argument("--points", type=json.loads,
                           help="JSON list of image points, to skip interactive picking")
    calibrate.set_defaults(func=cmd_calibrate)

    analyze = subparsers.add_parser(
        "analyze", help="full analysis: segmentation, metrics, quality and flags"
    )
    analyze.add_argument("videos", nargs="+")
    analyze.add_argument("--user", required=True)
    analyze.add_argument("--date", help="session date (ISO); defaults to today")
    analyze.add_argument("--device", help="assistive device used, if any")
    analyze.add_argument("--save", action="store_true",
                         help="persist the session to the trend history")
    analyze.add_argument("--skip-camera-check", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

    users = subparsers.add_parser("users", help="list users and session counts")
    users.set_defaults(func=cmd_users)

    sessions = subparsers.add_parser("sessions", help="list a user's sessions")
    sessions.add_argument("--user", required=True)
    sessions.set_defaults(func=cmd_sessions)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
