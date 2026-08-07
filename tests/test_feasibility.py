"""Speed feasibility: the guard that stops the tool inventing a gait speed.

The four sample clips in this project are all treadmill/in-place or
camera-tracked, so they are the real-world case these tests mirror: a naive
pipeline reports a very low speed for them, and a very low speed is exactly the
tool's high-risk signal.
"""
import numpy as np

from gaitscreen.io.video import CameraMotion
from gaitscreen.pose.to_pixels import to_pixels
from gaitscreen.quality import feasibility
from fixtures.synthetic import synthetic_walk

STATIC_CAMERA = CameraMotion(net_px=2.0, path_px=8.0, determinate=True, n_samples=20)
UNTRACKABLE = CameraMotion(
    net_px=float("nan"), path_px=float("nan"), determinate=False, n_samples=0,
    note="chroma-key backdrop",
)


def test_overground_walk_is_measurable(cfg):
    raw, _ = synthetic_walk(n_strides=12, speed_px_s=180.0, width=1280, fps=60.0)
    series = to_pixels(raw, cfg)

    result = feasibility.assess_speed(series, STATIC_CAMERA, cfg)
    assert result.feasible
    assert result.subject_translation_frac > 0.25


def test_walking_in_place_is_refused(cfg):
    """Treadmill or looped footage: no forward travel to measure."""
    raw, _ = synthetic_walk(n_strides=12, in_place=True, width=1280, fps=60.0)
    series = to_pixels(raw, cfg)

    result = feasibility.assess_speed(series, UNTRACKABLE, cfg)
    assert not result.feasible
    assert "no measurable forward travel" in result.reason
    assert "Timing metrics are unaffected" in result.reason


def test_tracking_camera_is_refused_even_when_subject_translates(cfg):
    """The street clip's failure mode: subject moves, but the camera follows."""
    raw, _ = synthetic_walk(n_strides=12, speed_px_s=180.0, width=1280, fps=60.0)
    series = to_pixels(raw, cfg)
    panning = CameraMotion(
        net_px=0.5 * series.video.width, path_px=0.5 * series.video.width,
        determinate=True, n_samples=40,
    )

    result = feasibility.assess_speed(series, panning, cfg)
    assert not result.feasible
    assert "the camera itself moved" in result.reason


def test_untrackable_background_is_flagged_but_not_fatal(cfg):
    """A green screen means camera motion is unknown, not zero."""
    raw, _ = synthetic_walk(n_strides=12, speed_px_s=180.0, width=1280, fps=60.0)
    series = to_pixels(raw, cfg)

    result = feasibility.assess_speed(series, UNTRACKABLE, cfg)
    assert result.feasible
    assert "could not be verified" in result.reason
    assert np.isnan(result.camera_motion_frac)


def test_translation_threshold_is_configurable(cfg):
    """A short pass crossing ~31% of the frame sits either side of the threshold."""
    raw, _ = synthetic_walk(n_strides=2, speed_px_s=180.0, width=1280, fps=60.0)
    series = to_pixels(raw, cfg)

    result = feasibility.assess_speed(series, STATIC_CAMERA, cfg)
    assert result.feasible, f"only travelled {result.subject_translation_frac:.0%}"

    strict = cfg.with_overrides({"speed": {"min_subject_translation_frac": 0.5}})
    assert not feasibility.assess_speed(series, STATIC_CAMERA, strict).feasible
