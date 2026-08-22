"""Session feature extraction against synthetic ground truth."""
import numpy as np
import pytest

from gaitscreen.features.session import analyse
from fixtures.extraction import build_extraction
from fixtures.synthetic import EXPECTED_DOUBLE_SUPPORT_PCT

# --------------------------------------------------------------------------
# temporal metrics
# --------------------------------------------------------------------------
def test_cadence_matches_stride_time(cfg):
    """One stride is two steps -- these are derived independently and must agree."""
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16,
                                     stride_time_s=1.10, stride_time_cv=0.02)
    metrics = analyse(extraction, cfg).metrics

    assert metrics.cadence_spm is not None
    expected = 120.0 / metrics.stride_time_mean_s
    assert abs(metrics.cadence_spm - expected) / expected < 0.05


def test_cadence_recovered_from_ground_truth(cfg):
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16,
                                     stride_time_s=1.20, stride_time_cv=0.0)
    metrics = analyse(extraction, cfg).metrics
    assert abs(metrics.cadence_spm - 100.0) < 6.0  # 120 / 1.2


def test_double_support_recovered(cfg):
    """The fixture places toe-off at 60% of cycle, implying 20% double support."""
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16,
                                     stride_time_cv=0.0)
    metrics = analyse(extraction, cfg).metrics

    assert metrics.double_support_pct is not None
    assert abs(metrics.double_support_pct - EXPECTED_DOUBLE_SUPPORT_PCT) < 5.0


def test_variability_withheld_when_too_few_strides(cfg):
    """A CV from a handful of strides is sampling noise, not a measurement."""
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=4)
    metrics = analyse(extraction, cfg).metrics

    assert metrics.stride_time_cv_pct is None
    assert "at least" in metrics.unavailable["stride_time_cv_pct"]
    # The underlying numbers are still reported, just not the derived CV.
    assert metrics.stride_time_mean_s is not None


def test_variability_marked_low_confidence_at_low_frame_rate(cfg):
    extraction, _ = build_extraction(cfg, fps=25.0, n_strides=20)
    metrics = analyse(extraction, cfg).metrics

    assert metrics.stride_time_cv_pct is not None
    assert "25 fps" in metrics.low_confidence_metrics["stride_time_cv_pct"]


def test_variability_not_flagged_at_high_frame_rate(cfg):
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=20)
    metrics = analyse(extraction, cfg).metrics
    assert "stride_time_cv_pct" not in metrics.low_confidence_metrics


# --------------------------------------------------------------------------
# spatial metrics
# --------------------------------------------------------------------------
def test_symmetric_walker_reads_as_symmetric(cfg):
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16,
                                     stride_time_cv=0.0)
    metrics = analyse(extraction, cfg).metrics

    assert metrics.step_length_asymmetry_pct is not None
    assert metrics.step_length_asymmetry_pct < 10.0
    assert metrics.step_time_asymmetry_pct < 10.0


def test_gait_speed_recovered_with_calibration(cfg):
    extraction, truth = build_extraction(
        cfg, calibrate=True, fps=60.0, n_strides=16, speed_px_s=180.0,
        scale_m_per_px=0.005, width=1280,
    )
    metrics = analyse(extraction, cfg).metrics

    assert metrics.gait_speed_mps is not None
    assert abs(metrics.gait_speed_mps - truth.speed_m_s) < 0.1 * truth.speed_m_s


def test_gait_speed_refused_without_calibration(cfg):
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16, speed_px_s=180.0)
    metrics = analyse(extraction, cfg).metrics

    assert metrics.gait_speed_mps is None
    assert "no calibration" in metrics.unavailable["gait_speed_mps"]
    # Timing metrics must be unaffected by a missing calibration.
    assert metrics.cadence_spm is not None


def test_gait_speed_refused_for_in_place_walk(cfg):
    extraction, _ = build_extraction(cfg, calibrate=True, fps=60.0, n_strides=16,
                                     in_place=True)
    metrics = analyse(extraction, cfg).metrics

    assert metrics.gait_speed_mps is None
    assert "forward travel" in metrics.unavailable["gait_speed_mps"]
    assert metrics.stride_time_cv_pct is not None, (
        "an unmeasurable speed must not suppress the calibration-free metrics"
    )


def test_trunk_sway_reports_the_sagittal_substitute(cfg):
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16)
    result = analyse(extraction, cfg)

    assert result.metrics.trunk_ap_sway_norm is not None
    note = result.metrics.low_confidence_metrics["trunk_ap_sway_norm"]
    assert "lateral trunk sway cannot be measured" in note


# --------------------------------------------------------------------------
# joint angles
# --------------------------------------------------------------------------
def test_angle_curves_produced_for_every_joint(cfg):
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16)
    angles = analyse(extraction, cfg).angles

    assert angles.percent.shape == (101,)
    for side in ("left", "right"):
        for joint in ("hip", "knee", "ankle"):
            rows = angles.curves[f"{side}_{joint}"]
            assert rows.shape[1] == 101
            assert rows.shape[0] >= 5


def test_knee_flexes_during_swing(cfg):
    """Sign convention check: knee flexion must be positive in swing."""
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16)
    angles = analyse(extraction, cfg).angles

    knee = angles.mean_curve("left_knee")
    swing_peak = float(np.nanmax(knee[60:90]))
    stance_min = float(np.nanmin(knee[0:40]))
    assert swing_peak > stance_min
    assert swing_peak > 20.0, "swing-phase knee flexion should be clearly positive"


# --------------------------------------------------------------------------
# self-consistency guards
# --------------------------------------------------------------------------
def test_clean_walk_raises_no_consistency_warnings(cfg):
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16,
                                     stride_time_cv=0.02)
    notes = " ".join(analyse(extraction, cfg).notes)

    assert "inconsistent" not in notes
    assert "told apart" not in notes
    assert "independent count" not in notes


def test_swapped_limb_labels_are_detected(cfg):
    """MediaPipe can swap left and right on low-texture subjects.

    Undetected, this halves stride time and inflates asymmetry while every
    number still looks individually plausible.
    """
    from gaitscreen.pose.schema import SIDE_LANDMARKS

    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16,
                                     stride_time_cv=0.0)
    series = extraction.series
    # Make the "right" limb a copy of the left: the legs no longer alternate.
    for joint in ("hip", "knee", "ankle", "heel", "foot_index"):
        series.xy[:, int(SIDE_LANDMARKS["right"][joint])] = series.xy[
            :, int(SIDE_LANDMARKS["left"][joint])
        ]

    result = analyse(extraction, cfg)
    notes = " ".join(result.notes)
    assert "not being tracked separately" in notes, (
        f"limb duplication went undetected; notes were: {result.notes}"
    )
    # The affected metrics must not be reported as if they were sound: either
    # withheld outright, or carried with the caveat attached.
    metrics = result.metrics
    for metric in ("double_support_pct", "step_length_asymmetry_pct"):
        assert (
            metrics.value(metric) is None
            or metric in metrics.low_confidence_metrics
        ), f"{metric} was reported without qualification"


def test_no_passes_marks_everything_unavailable(cfg):
    extraction, _ = build_extraction(cfg, fps=60.0, n_strides=16)
    extraction.segments = []

    result = analyse(extraction, cfg)
    assert result.metrics.n_passes == 0
    assert result.metrics.cadence_spm is None
    assert all(
        reason for reason in result.metrics.unavailable.values()
    ), "every unavailable metric needs a stated reason"
