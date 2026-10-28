"""Analysing a recorded walk on the device.

Runs exactly the analysis the server and the Streamlit app run -- it is the
same ``gaitscreen`` package -- and writes the outcome as plain JSON, so the
server can store it without needing Python objects from this device.

Two things the device cannot know shape what it reports:

* **Who walked past.** A camera on a corridor wall sees whoever comes by. The
  device reports the walk; deciding which resident it was, and so which
  personal history it belongs to, happens on the server. Only the absolute
  checks (e.g. walking speed under the illustrative threshold) can run here;
  the personal-trend checks run on the server once the walk has a person.
* **Whether it was a walk at all.** The motion trigger records doors opening
  and people pausing in shot. A clip in which no gait could be measured is
  marked ``no_walk`` and, by default, deleted rather than sent: footage of
  people that measured nothing should not leave the device.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from .settings import Settings

#: Person id used for walks the device has not been told the owner of.
UNASSIGNED = "unassigned"


def analyse_file(settings: Settings, video: Path, out_dir: Path, *,
                 make_overlay: bool = True, progress=None) -> dict:
    """Analyse ``video`` and write ``result.json`` (and ``overlay.mp4``)."""
    from gaitscreen.calibration.setup import CameraSetup
    from gaitscreen.io import video as video_io
    from gaitscreen.pipeline import analyse_video

    cfg = settings.analysis_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = None
    coverage = settings["processing.coverage_m"]
    if coverage:
        info = video_io.probe(video, cfg)
        setup = CameraSetup(float(coverage), f"device-{settings.device_id}", "")
        calibration = setup.calibration_for(info.width, info.height)

    started = time.time()
    result = analyse_video(
        video, cfg, user_id=UNASSIGNED, calibration=calibration,
        project_root=settings.root, progress=progress,
        check_camera_motion=True,
    )
    analysis_s = time.time() - started

    payload = result_to_dict(result, cfg)
    payload["device"] = settings.describe()
    payload["processing"] = {
        "where": "device",
        "seconds": round(analysis_s, 1),
        "coverage_m": coverage,
    }

    if make_overlay and payload["summary"]["is_walk"]:
        payload["overlay"] = _overlay(result, cfg, out_dir / "overlay.mp4")

    (out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _overlay(result, cfg, path: Path) -> Optional[dict]:
    from gaitscreen.reporting.overlay import render_overlay_video

    try:
        overlay = render_overlay_video(
            result.extraction, result.analysis, path, cfg,
            max_width=int(cfg["reporting.overlay.max_width"]),
        )
        return {"file": path.name, "browser_playable": overlay.browser_playable,
                "note": overlay.note}
    except Exception as exc:  # noqa: BLE001 - the replay is not the measurement
        return {"file": None, "error": str(exc)}


# --------------------------------------------------------------------------
def result_to_dict(result, cfg) -> dict:
    """A SessionResult as JSON-safe data, in the shape the server stores."""
    from gaitscreen.reporting.plain import summarise
    from gaitscreen.version import ALGO_VERSION

    analysis, extraction, metrics = result.analysis, result.extraction, result.metrics
    info = extraction.info
    plain = summarise(result, cfg)
    n_valid = int(analysis.cycle_summary.get("n_valid", 0) or 0)
    measured = [c for c in plain.cards if c.measured]

    return _clean({
        "schema": "gaitedge.result/1",
        "algo_version": ALGO_VERSION,
        "recorded": {
            "file": Path(str(info.path)).name,
            "width": info.width, "height": info.height,
            "fps": info.fps, "duration_s": info.duration_s,
            "dropped_frames": info.dropped_frames,
            "drop_rate_pct": info.drop_rate_pct,
        },
        "summary": {
            "is_walk": bool(n_valid > 0 and measured),
            "headline": plain.headline,
            "detail": plain.sub_headline,
            "tone": plain.tone,
            "view": analysis.view.kind if analysis.view else None,
            "strides_valid": n_valid,
            "quality_score": result.quality.score,
            "low_confidence": result.quality.low_confidence,
        },
        "metrics": {k: v for k, v in _asdict(metrics).items()
                    if k not in ("unavailable", "low_confidence_metrics", "lower_bounds")},
        "unavailable": dict(metrics.unavailable),
        "caveats": dict(metrics.low_confidence_metrics),
        "lower_bounds": dict(metrics.lower_bounds),
        "cards": [
            {"key": c.key, "name": c.name, "status": c.status, "value": c.value_text,
             "note": c.note}
            for c in plain.cards
        ],
        "flags": [_asdict(f) for f in result.flags.flags],
        "recording_feedback": [
            {"code": d.code, "severity": d.severity, "title": d.title,
             "detail": d.detail, "fix": d.fix}
            for d in result.diagnostics.diagnostics
        ],
        "events": [
            {"t": e.t, "kind": e.kind, "side": e.side} for e in analysis.events
        ],
        "notes": result.all_notes,
    })


def _asdict(obj) -> dict:
    return asdict(obj) if is_dataclass(obj) else dict(obj)


def _clean(value: Any) -> Any:
    """Replace NaN/inf with None and numpy scalars with Python ones."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
