"""Session-level feature extraction: segmentation through to the metric set.

The guiding rule throughout is that a metric which cannot be measured is reported
as missing, with a reason, and never defaulted. In a screening tool a plausible
default reads as a normal result, and a normal result is precisely what a missed
decline looks like.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..segmentation import cycles as cycles_module
from ..segmentation import direction as direction_module
from ..segmentation.events import (EventSet, detect_events,
                                   independent_cadence_spm)
from ..types import (AngleCurves, GaitCycle, GaitEvent, SessionMetrics,
                     WalkPass)
from . import angles as angles_module
from . import asymmetry as asymmetry_module
from . import spatiotemporal, trunk as trunk_module
from . import variability as variability_module


@dataclass
class SessionAnalysis:
    """Everything the feature stage produces, including its own diagnostics."""

    metrics: SessionMetrics
    angles: AngleCurves
    passes: list[WalkPass] = field(default_factory=list)
    events: list[GaitEvent] = field(default_factory=list)
    cycles: list[GaitCycle] = field(default_factory=list)
    cycle_summary: dict = field(default_factory=dict)
    range_of_motion: dict[str, float] = field(default_factory=dict)
    direction_method: str = ""
    camera_side: str | None = None
    event_agreement_ms: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def valid_cycles(self) -> list[GaitCycle]:
        return [c for c in self.cycles if c.valid]


def analyse(extraction, cfg: Config) -> SessionAnalysis:
    """Run segmentation and feature extraction over a completed extraction."""
    series = extraction.series
    metrics = SessionMetrics()
    notes: list[str] = []

    sign, method, agreement = direction_module.anterior_sign(series)
    if method == "assumed":
        notes.append(
            "the direction the subject is facing could not be determined from the "
            "landmarks; left-to-right was assumed, which may reverse heel strike "
            "and toe-off"
        )
    elif agreement < 0.85:
        notes.append(
            f"direction of travel was resolved from {method} with only "
            f"{agreement:.0%} frame agreement"
        )

    passes = direction_module.find_passes(series, cfg, extraction.segments)
    if not passes:
        metrics.n_passes = 0
        for metric in ("gait_speed_mps", "stride_time_cv_pct",
                       "step_length_asymmetry_pct", "cadence_spm",
                       "double_support_pct", "trunk_ap_sway_norm"):
            metrics.mark_unavailable(
                metric, "no usable walking pass was found in this recording"
            )
        return SessionAnalysis(
            metrics=metrics, angles=AngleCurves(percent=np.linspace(0, 100, 101)),
            direction_method=method, notes=notes + [
                "no walking pass could be segmented; the subject may not be fully "
                "visible, or the walk may be too short"
            ],
        )

    all_events: list[GaitEvent] = []
    all_cycles: list[GaitCycle] = []
    agreements: list[float] = []

    for index, walk_pass in enumerate(passes):
        event_set: EventSet = detect_events(series, walk_pass, cfg)
        all_events.extend(event_set.events)
        all_cycles.extend(
            cycles_module.build_cycles(series, walk_pass, index, event_set, cfg)
        )
        if event_set.agreement_ms is not None:
            agreements.append(event_set.agreement_ms)
        if event_set.note:
            notes.append(event_set.note)

    all_cycles.sort(key=lambda c: c.t_start)
    summary = cycles_module.summarise(all_cycles)
    camera_side = next((p.camera_side for p in passes if p.camera_side), None)

    metrics.n_passes = len(passes)
    metrics.n_strides_total = summary["n_total"]
    metrics.n_strides_valid = summary["n_valid"]

    _fill_temporal(metrics, all_events, all_cycles, cfg, series.fps, notes)
    _fill_spatial(metrics, extraction, series, all_events, all_cycles, passes,
                  cfg, camera_side, notes)
    _fill_trunk(metrics, series, all_cycles, passes, notes)

    _check_internal_consistency(metrics, notes)
    _check_limb_interleaving(all_cycles, metrics, notes)
    reference_cadence = independent_cadence_spm(series, cfg)
    _check_against_independent_cadence(metrics, reference_cadence, notes)

    if summary["n_valid"] < int(cfg["features.min_cycles_for_session"]):
        notes.append(
            f"only {summary['n_valid']} valid strides were recovered from "
            f"{summary['n_total']} detected; session metrics are weakly supported"
        )

    curves = angles_module.cycle_curves(series, all_cycles, passes, cfg)

    return SessionAnalysis(
        metrics=metrics,
        angles=curves,
        passes=passes,
        events=all_events,
        cycles=all_cycles,
        cycle_summary=summary,
        range_of_motion=angles_module.range_of_motion(curves),
        direction_method=method,
        camera_side=camera_side,
        event_agreement_ms=float(np.mean(agreements)) if agreements else None,
        notes=notes,
    )


def _check_internal_consistency(metrics: SessionMetrics, notes: list[str]) -> None:
    """Cross-check two measures that are derived independently but must agree.

    Cadence comes from intervals between successive heel strikes on alternating
    feet; stride time comes from intervals between heel strikes on the *same*
    foot. In any real walk one stride is two steps, so cadence must be close to
    ``120 / stride_time``. When they disagree, heel strikes are being
    double-counted or missed, and the timing metrics are unreliable even though
    each one looks plausible in isolation. Without this check that failure is
    invisible.
    """
    cadence = metrics.cadence_spm
    stride = metrics.stride_time_mean_s
    if cadence is None or not stride:
        return

    expected = 120.0 / stride
    error = abs(cadence - expected) / expected
    if error <= 0.15:
        return

    explanation = (
        f"cadence ({cadence:.0f} steps/min) and mean stride time ({stride:.2f}s) "
        f"are inconsistent -- one stride should equal two steps, implying "
        f"{expected:.0f} steps/min. Heel strikes are probably being "
        f"{'missed' if cadence < expected else 'double-counted'}, so the timing "
        "metrics from this recording are unreliable."
    )
    notes.append(explanation)
    for metric in ("cadence_spm", "stride_time_cv_pct", "double_support_pct"):
        if metrics.value(metric) is not None:
            metrics.low_confidence_metrics[metric] = explanation


def _check_against_independent_cadence(
    metrics: SessionMetrics, reference: float | None, notes: list[str]
) -> None:
    """Compare pipeline cadence against the label-free ankle-separation count.

    The two share no code path: one comes from per-limb Zeni events on a
    pelvis-relative signal, the other from counting peaks in the absolute
    distance between the ankles. Agreement is strong evidence the segmentation
    locked onto the real rhythm; disagreement is the clearest available signal
    that it did not.
    """
    cadence = metrics.cadence_spm
    if cadence is None or reference is None or reference <= 0:
        return

    error = abs(cadence - reference) / reference
    if error <= 0.15:
        return

    explanation = (
        f"the pipeline measured {cadence:.0f} steps/min, but an independent count "
        f"of leg-split events gives {reference:.0f} steps/min ({error:.0%} apart). "
        "Gait events are probably not being detected reliably in this recording, "
        "so all timing metrics should be treated as unreliable."
    )
    notes.append(explanation)
    for metric in ("cadence_spm", "stride_time_cv_pct", "double_support_pct",
                   "step_time_asymmetry_pct"):
        if metrics.value(metric) is not None:
            metrics.low_confidence_metrics[metric] = explanation


def _check_limb_interleaving(
    cycles: list[GaitCycle], metrics: SessionMetrics, notes: list[str]
) -> None:
    """Verify that the two limbs alternate, as walking requires.

    In any real gait the opposite foot strikes near the middle of the stride, so
    the contralateral heel strike should sit around 50% of the cycle. A value
    near 0% or 100% means the two limbs are not being distinguished -- MediaPipe
    assigns left and right anatomically, and it can swap them when the subject
    has little visual texture to separate the legs (a plain silhouette is the
    worst case). Undetected, that inflates asymmetry and halves stride time while
    every metric still looks individually plausible.
    """
    valid = [c for c in cycles if c.valid and c.duration_s > 0]
    phases = [
        (c.contra_heel_strike_t - c.t_start) / c.duration_s
        for c in valid if c.contra_heel_strike_t is not None
    ]

    if len(phases) < 3:
        # Silence here would be the wrong answer. If strides were found but the
        # opposite foot was never seen to strike inside any of them, the two
        # limbs are not being separated at all -- the most likely cause of which
        # is both "legs" being tracked onto the same one.
        if len(valid) >= 3:
            explanation = (
                f"{len(valid)} strides were found, but the opposite foot was not "
                "seen to strike within them, so the two legs are not being tracked "
                "separately. Left/right measurements and double-support time "
                "cannot be trusted from this recording."
            )
            notes.append(explanation)
            for metric in ("step_length_asymmetry_pct", "step_time_asymmetry_pct",
                           "double_support_pct"):
                if metrics.value(metric) is not None:
                    metrics.low_confidence_metrics[metric] = explanation
        return

    median_phase = float(np.median(phases))
    if 0.25 <= median_phase <= 0.75:
        return

    explanation = (
        f"the opposite foot strikes at {median_phase:.0%} of the stride rather "
        "than near the middle, so the two legs are probably not being told apart "
        "reliably. Left/right measurements from this recording, including "
        "asymmetry, should not be trusted."
    )
    notes.append(explanation)
    for metric in ("step_length_asymmetry_pct", "step_time_asymmetry_pct",
                   "double_support_pct"):
        if metrics.value(metric) is not None:
            metrics.low_confidence_metrics[metric] = explanation


# --------------------------------------------------------------------------
# metric groups
# --------------------------------------------------------------------------
def _fill_temporal(
    metrics: SessionMetrics, events, cycles, cfg: Config, fps: float, notes: list[str]
) -> None:
    result = variability_module.stride_time_variability(cycles, cfg, fps)
    metrics.stride_time_mean_s = result.mean_s
    metrics.stride_time_sd_s = result.sd_s
    if result.cv_pct is None:
        metrics.mark_unavailable("stride_time_cv_pct", result.unavailable_reason or "")
    else:
        metrics.stride_time_cv_pct = result.cv_pct
        if result.low_confidence_reason:
            metrics.low_confidence_metrics["stride_time_cv_pct"] = (
                result.low_confidence_reason
            )

    cadence = spatiotemporal.cadence_spm(events, cycles)
    if cadence is None:
        metrics.mark_unavailable(
            "cadence_spm",
            "fewer than three heel strikes fell inside valid strides, so step "
            "intervals could not be measured",
        )
    else:
        metrics.cadence_spm = cadence

    double_support, n_used = spatiotemporal.double_support_pct(cycles)
    if double_support is None:
        metrics.mark_unavailable(
            "double_support_pct",
            "no stride had a complete set of heel-strike and toe-off events on "
            "both feet, which double-support time requires",
        )
    else:
        metrics.double_support_pct = double_support
        if n_used < 3:
            metrics.low_confidence_metrics["double_support_pct"] = (
                f"based on only {n_used} strides"
            )


def _fill_spatial(
    metrics: SessionMetrics, extraction, series, events, cycles, passes,
    cfg: Config, camera_side, notes: list[str],
) -> None:
    measures = spatiotemporal.step_measures(series, events, cycles, passes)
    result = asymmetry_module.compute(measures, camera_side=camera_side)

    if result.step_length_pct is None:
        metrics.mark_unavailable(
            "step_length_asymmetry_pct", result.unavailable_reason or ""
        )
    else:
        metrics.step_length_asymmetry_pct = result.step_length_pct
        metrics.step_time_asymmetry_pct = result.step_time_pct
        if result.note:
            metrics.low_confidence_metrics["step_length_asymmetry_pct"] = result.note

    # --- gait speed: three independent preconditions ---------------------
    speed = extraction.speed
    check = extraction.calibration_check
    if not speed.feasible:
        metrics.mark_unavailable("gait_speed_mps", speed.reason or "")
        return
    if not check.usable:
        reason = check.reason or "calibration unavailable"
        # Some calibration reasons already say that scale-free metrics survive;
        # appending the same reassurance again just reads as noise.
        if "unaffected" not in reason:
            reason += " -- the timing metrics are unaffected by this"
        metrics.mark_unavailable("gait_speed_mps", reason)
        return

    value, reason = spatiotemporal.gait_speed_mps(
        series, passes, cfg, extraction.calibration
    )
    if value is None:
        metrics.mark_unavailable("gait_speed_mps", reason or "")
    else:
        metrics.gait_speed_mps = value
        if extraction.calibration.method == "two_point":
            metrics.low_confidence_metrics["gait_speed_mps"] = (
                "derived from a single-scalar pixel scale, which is exact only in "
                "the plane of the calibration reference; a 4-point floor "
                "homography removes the remaining perspective error"
            )


def _fill_trunk(metrics: SessionMetrics, series, cycles, passes, notes: list[str]) -> None:
    result = trunk_module.compute(series, cycles, passes)
    if result.ap_sway_norm is None:
        metrics.mark_unavailable("trunk_ap_sway_norm", result.unavailable_reason or "")
    else:
        metrics.trunk_ap_sway_norm = result.ap_sway_norm
        metrics.low_confidence_metrics["trunk_ap_sway_norm"] = result.note or ""
    if result.note and result.note not in notes:
        notes.append(result.note)
