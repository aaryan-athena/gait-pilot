"""Session data-quality scoring.

A single score in [0, 1] combining the things that actually determine whether a
session's numbers can be trusted: how well the gait-critical landmarks were seen,
how many complete strides were recovered, the frame rate, the pose detection
rate, and whether an assistive device was in use.

Sessions scoring below the configured threshold are marked ``low_confidence``.
They are still stored and still reported -- hiding a session would hide the fact
that data collection is degrading -- but they are excluded from personal baseline
computation, so a bad recording cannot silently redefine what "normal" means for
that user.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..config import Config
from ..pose.schema import GAIT_CRITICAL
from ..types import GaitCycle, PixelSeries, QualityReport, RawLandmarks
from . import assistive_device as device_module


def score_session(
    series: PixelSeries,
    raw: RawLandmarks,
    cfg: Config,
    *,
    cycles: Optional[Sequence[GaitCycle]] = None,
    declared_device: Optional[str] = None,
    extra_notes: Optional[Sequence[str]] = None,
) -> QualityReport:
    """Compute the session quality score and its component breakdown."""
    weights = cfg.section("quality.weights")
    notes: list[str] = list(extra_notes or [])
    components: dict[str, float] = {}

    # --- landmark visibility on the joints segmentation depends on --------
    critical = list(GAIT_CRITICAL)
    visibility = raw.visibility[:, critical]
    seen = visibility[raw.detected] if raw.detected.any() else visibility
    mean_visibility = float(np.mean(seen)) if seen.size else 0.0
    components["landmark_visibility"] = _clip01(mean_visibility)
    # Mean visibility only just above the per-sample rejection threshold means a
    # large share of samples are being discarded and interpolated over.
    if mean_visibility < float(cfg["landmarks.visibility_threshold"]) + 0.1:
        notes.append(
            f"mean visibility of hip/knee/ankle/foot landmarks is only "
            f"{mean_visibility:.2f}; loose clothing, poor lighting or a partially "
            "occluded walking path all cause this"
        )

    # --- how many usable strides ----------------------------------------
    target = float(cfg["quality.target_cycle_count"])
    n_valid = sum(1 for c in (cycles or []) if c.valid)
    components["valid_cycle_count"] = _clip01(n_valid / target) if target > 0 else 0.0
    if cycles is not None and n_valid < cfg["features.min_cycles_for_session"]:
        notes.append(
            f"only {n_valid} valid gait cycles were recovered; too few for stable "
            "session metrics"
        )

    # --- frame rate ------------------------------------------------------
    fps = series.fps
    variability_min = float(cfg["video.fps_variability_min"])
    components["frame_rate"] = _clip01(fps / variability_min)
    if fps < variability_min:
        notes.append(
            f"{fps:g} fps limits timing precision; stride-time variability from this "
            "recording is indicative only"
        )

    # --- pose detection rate ---------------------------------------------
    components["detection_rate"] = _clip01(raw.detection_rate)
    if raw.detection_rate < 0.95:
        notes.append(
            f"a pose was found in only {raw.detection_rate:.0%} of frames; the "
            "subject may leave the frame or be partly out of view"
        )

    # --- assistive device -------------------------------------------------
    device, source, device_note = device_module.resolve(
        series, cfg, declared_device=declared_device
    )
    if device is None:
        components["assistive_device"] = 1.0
    elif source == "metadata":
        components["assistive_device"] = 0.0
        notes.append(
            f"recorded as using a {device}; this version does not model "
            "device-assisted gait, so the session is low-confidence by design"
        )
    else:
        # Heuristic suspicion is softer than a declared device: it may lower
        # confidence but should not condemn the session outright.
        components["assistive_device"] = 0.5
        if device_note:
            notes.append(device_note + " (heuristic only -- not confirmed)")

    total_weight = sum(float(weights[key]) for key in components)
    score = sum(components[key] * float(weights[key]) for key in components)
    score = score / total_weight if total_weight else 0.0

    threshold = float(cfg["quality.low_confidence_below"])
    low_confidence = score < threshold
    if low_confidence:
        notes.append(
            f"overall quality {score:.2f} is below the {threshold:.2f} threshold, so "
            "this session is excluded from personal baseline calculations"
        )

    return QualityReport(
        score=float(score),
        low_confidence=bool(low_confidence),
        components=components,
        notes=notes,
        assistive_device=device,
        assistive_device_source=source,
    )


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))
