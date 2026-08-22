"""Recording diagnostics.

Each test degrades one property of an otherwise-clean synthetic recording and
checks that the diagnostic layer names *that* problem — not merely that
something went wrong. A diagnostic that fires on everything is as useless as one
that never fires, so the clean-recording case is checked just as carefully.
"""
import numpy as np
import pytest

from gaitscreen.features.session import analyse
from gaitscreen.io.video import CameraMotion
from gaitscreen.pose.schema import PL, SIDE_LANDMARKS
from gaitscreen.quality.diagnostics import diagnose
from fixtures.extraction import build_extraction


def _report(cfg, **kwargs):
    extraction, _ = build_extraction(cfg, **kwargs)
    return diagnose(extraction, analyse(extraction, cfg), cfg), extraction


def _clean(cfg, **overrides):
    """A recording with nothing wrong with it: big subject, 60 fps, many strides.

    ``in_place=True`` is not a shortcut. At a good framing -- subject filling
    half the frame, so a leg is ~340 px -- one stride covers roughly 480 px, and
    sixteen of them is 7600 px. That cannot fit inside a 1280 px frame, so a
    single overground pass at good framing physically cannot contain enough
    strides for a variability measurement. Real recordings solve this with
    several passes back and forth; the fixture stands in for that by keeping the
    subject in place. The diagnostics here are about framing, angle, clothing and
    tracking, none of which depend on net translation.
    """
    params = dict(fps=60.0, n_strides=16, leg_length_px=340, width=1280,
                  height=720, stride_time_cv=0.02, in_place=True)
    params.update(overrides)
    return params


# --------------------------------------------------------------------------
# the clean case
# --------------------------------------------------------------------------
def test_good_recording_reports_no_problems(cfg):
    report, _ = _report(cfg, **_clean(cfg))
    assert report.is_clean, (
        "a well-framed 60 fps recording of a long walk must produce no "
        f"diagnostics, got: {report.codes()}"
    )
    assert report.headline is None


def test_measurements_are_always_reported(cfg):
    """Even a clean recording exposes its numbers, for tuning a setup."""
    report, _ = _report(cfg, **_clean(cfg))
    for key in ("subject_height_frac", "leg_length_px", "fully_visible_frac",
                "camera_angle_deg", "foot_jitter_norm", "strides_valid", "fps"):
        assert key in report.measurements, f"{key} missing from measurements"


# --------------------------------------------------------------------------
# framing and subject size
# --------------------------------------------------------------------------
def test_small_subject_is_flagged_with_a_distance_factor(cfg):
    report, _ = _report(cfg, **_clean(cfg, leg_length_px=60, height=720))
    codes = report.codes()

    assert "subject_too_small" in codes
    diagnostic = next(d for d in report.diagnostics if d.code == "subject_too_small")
    assert "x closer" in diagnostic.fix, "the fix must say how much closer"
    assert diagnostic.measured["subject_height_frac"] < 0.45


def test_subject_leaving_frame_is_flagged(cfg):
    extraction, _ = build_extraction(cfg, **_clean(cfg))
    # Blank the lower body for the first half: the subject is only partly in shot.
    half = extraction.series.n_frames // 2
    for side in ("left", "right"):
        for joint in ("ankle", "heel", "foot_index"):
            extraction.series.xy[:half, int(SIDE_LANDMARKS[side][joint])] = np.nan

    report = diagnose(extraction, analyse(extraction, cfg), cfg)
    assert "subject_not_in_frame" in report.codes()
    diagnostic = next(d for d in report.diagnostics
                      if d.code == "subject_not_in_frame")
    assert "usable_seconds" in diagnostic.measured


def test_feet_below_frame_edge_are_flagged(cfg):
    extraction, _ = build_extraction(cfg, **_clean(cfg))
    feet = [int(SIDE_LANDMARKS[s][j]) for s in ("left", "right")
            for j in ("heel", "foot_index")]
    # Push the feet to the very bottom of the frame (y-up, so near zero).
    extraction.series.xy[:, feet, 1] = 1.0

    report = diagnose(extraction, analyse(extraction, cfg), cfg)
    assert "feet_clipped" in report.codes()


# --------------------------------------------------------------------------
# camera angle
# --------------------------------------------------------------------------
def test_side_on_recording_reads_near_zero_degrees(cfg):
    report, _ = _report(cfg, **_clean(cfg))
    assert report.measurements["camera_angle_deg"] < 10.0


def test_oblique_camera_angle_is_flagged(cfg):
    extraction, _ = build_extraction(cfg, **_clean(cfg))
    series = extraction.series
    # Separate the shoulders horizontally, as a front-quarter view would.
    trunk = float(np.nanmedian(
        series.point(PL.LEFT_SHOULDER)[:, 1] - series.point(PL.LEFT_HIP)[:, 1]
    ))
    # Both shoulders move, so each takes half the intended separation.
    # A separation of 0.7 x the anatomical width implies asin(0.7) = 44 degrees.
    offset = 0.5 * 0.7 * 0.79 * trunk
    series.xy[:, int(PL.LEFT_SHOULDER), 0] += offset
    series.xy[:, int(PL.RIGHT_SHOULDER), 0] -= offset

    report = diagnose(extraction, analyse(extraction, cfg), cfg)
    assert "oblique_camera_angle" in report.codes()
    diagnostic = next(d for d in report.diagnostics
                      if d.code == "oblique_camera_angle")
    assert 38 < diagnostic.measured["camera_angle_deg"] < 52
    assert diagnostic.measured["foreshortening"] > 0.1


# --------------------------------------------------------------------------
# clothing and tracking quality
# --------------------------------------------------------------------------
def test_loose_clothing_signature_is_flagged(cfg):
    """Low leg visibility against a confidently-tracked torso."""
    extraction, _ = build_extraction(cfg, **_clean(cfg))
    lower = [int(SIDE_LANDMARKS[s][j]) for s in ("left", "right")
             for j in ("knee", "ankle", "heel", "foot_index")]
    extraction.raw.visibility[:, lower] = 0.55

    report = diagnose(extraction, analyse(extraction, cfg), cfg)
    assert "lower_body_occluded" in report.codes()
    diagnostic = next(d for d in report.diagnostics
                      if d.code == "lower_body_occluded")
    assert "clothing" in diagnostic.detail
    assert "trousers" in diagnostic.fix


def test_clothing_check_does_not_fire_on_uniform_visibility(cfg):
    """A dim recording that is *equally* dim everywhere is not a clothing problem."""
    extraction, _ = build_extraction(cfg, **_clean(cfg))
    extraction.raw.visibility[:, :] = 0.6

    report = diagnose(extraction, analyse(extraction, cfg), cfg)
    assert "lower_body_occluded" not in report.codes()


def test_jittery_foot_tracking_is_flagged(cfg):
    extraction, _ = build_extraction(cfg, **_clean(cfg))
    rng = np.random.default_rng(3)
    for side in ("left", "right"):
        index = int(SIDE_LANDMARKS[side]["heel"])
        extraction.series.xy[:, index, 0] += rng.normal(
            0, 0.05 * 340, extraction.series.n_frames
        )

    report = diagnose(extraction, analyse(extraction, cfg), cfg)
    assert "noisy_foot_tracking" in report.codes()


def test_person_switch_is_flagged(cfg):
    extraction, _ = build_extraction(cfg, **_clean(cfg))
    # Teleport the whole body mid-recording, as a tracker swapping subjects would.
    half = extraction.series.n_frames // 2
    extraction.series.xy[half:, :, 0] += 400.0

    report = diagnose(extraction, analyse(extraction, cfg), cfg)
    assert "tracking_switched_person" in report.codes()
    diagnostic = next(d for d in report.diagnostics
                      if d.code == "tracking_switched_person")
    assert diagnostic.measured["identity_jumps"] >= 1


def test_camera_shake_is_flagged_but_a_steady_pan_is_not(cfg):
    shaky = CameraMotion(net_px=2.0, path_px=400.0, determinate=True, n_samples=50)
    report, _ = _report(cfg, camera=shaky, **_clean(cfg))
    assert "camera_shake" in report.codes()

    # A steady pan has path ~= net: it is a different problem, handled by the
    # speed feasibility check, and must not be reported as shake.
    panning = CameraMotion(net_px=400.0, path_px=402.0, determinate=True, n_samples=50)
    report, _ = _report(cfg, camera=panning, **_clean(cfg))
    assert "camera_shake" not in report.codes()


# --------------------------------------------------------------------------
# walk length and frame rate
# --------------------------------------------------------------------------
def test_short_walk_says_how_much_more_is_needed(cfg):
    report, _ = _report(cfg, **_clean(cfg, n_strides=4))
    assert "walk_too_short" in report.codes()
    diagnostic = next(d for d in report.diagnostics if d.code == "walk_too_short")
    assert "more walking" in diagnostic.fix or "longer walk" in diagnostic.fix
    assert diagnostic.measured["strides_needed"] == 10


def test_low_frame_rate_is_minor_not_blocking(cfg):
    report, _ = _report(cfg, **_clean(cfg, fps=30.0))
    codes = report.codes()
    assert "frame_rate_too_low" in codes
    diagnostic = next(d for d in report.diagnostics if d.code == "frame_rate_too_low")
    assert diagnostic.severity == "minor", (
        "a low frame rate degrades one metric; it does not stop the measurement"
    )


# --------------------------------------------------------------------------
# report structure
# --------------------------------------------------------------------------
def test_diagnostics_are_ordered_most_severe_first(cfg):
    report, _ = _report(cfg, **_clean(cfg, leg_length_px=50, n_strides=3, fps=25.0))
    ranks = [d.rank for d in report.diagnostics]
    assert ranks == sorted(ranks)
    assert report.headline is report.diagnostics[0]


def test_every_diagnostic_is_actionable(cfg):
    """A diagnostic without a concrete fix is just a restated failure."""
    report, _ = _report(cfg, **_clean(cfg, leg_length_px=50, n_strides=3, fps=25.0))
    assert report.diagnostics
    for diagnostic in report.diagnostics:
        assert diagnostic.title and not diagnostic.title.endswith("."), diagnostic.code
        assert len(diagnostic.detail) > 40, diagnostic.code
        assert len(diagnostic.fix) > 25, diagnostic.code
        assert diagnostic.measured, diagnostic.code
        # The fix must tell someone what to do, not restate the measurement.
        assert any(
            verb in diagnostic.fix.lower()
            for verb in ("record", "set ", "place", "stand", "tilt", "put",
                         "capture", "fill", "start", "make sure")
        ), f"{diagnostic.code} fix is not an instruction: {diagnostic.fix}"


def test_blockers_are_separable_from_advice(cfg):
    report, _ = _report(cfg, **_clean(cfg, leg_length_px=50, n_strides=3))
    assert report.blockers
    assert all(d.severity == "blocker" for d in report.blockers)
