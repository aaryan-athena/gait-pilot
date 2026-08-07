"""Calibration maths and the camera-movement check."""
import numpy as np
import pytest

from gaitscreen.calibration import verify
from gaitscreen.calibration.model import Calibration, CalibrationError
from gaitscreen.pose.to_pixels import subject_pixel_height, to_pixels
from fixtures.synthetic import synthetic_walk


def test_two_point_scale():
    calibration = Calibration.from_two_points(
        "u1", (100, 400), (600, 400), distance_m=2.0,
        frame_width=1280, frame_height=720,
    )
    assert np.isclose(calibration.scale_m_per_px, 2.0 / 500)
    assert np.isclose(calibration.px_to_m(250), 1.0)
    assert np.isclose(calibration.distance_m((0, 0), (500, 0)), 2.0)


def test_two_point_rejects_degenerate_and_negative_input():
    with pytest.raises(CalibrationError):
        Calibration.from_two_points("u1", (10, 10), (10, 10), 2.0, 1280, 720)
    with pytest.raises(CalibrationError):
        Calibration.from_two_points("u1", (0, 0), (500, 0), -1.0, 1280, 720)


def test_short_reference_span_is_flagged():
    """A short reference amplifies clicking error into every speed measurement."""
    calibration = Calibration.from_two_points(
        "u1", (600, 400), (700, 400), distance_m=0.5,
        frame_width=1280, frame_height=720,
    )
    assert calibration.notes and "spans only" in calibration.notes


def test_homography_recovers_floor_distances_under_perspective():
    """The point of the 4-point method: distance stays right at varying depth."""
    # A 4x2 m floor rectangle imaged with perspective (nearer edge wider).
    image_points = [(300, 600), (900, 600), (760, 380), (440, 380)]
    floor_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)]
    calibration = Calibration.from_floor_points(
        "u1", image_points, floor_points, 1280, 720
    )

    near = calibration.distance_m(image_points[0], image_points[1])
    far = calibration.distance_m(image_points[3], image_points[2])
    assert np.isclose(near, 2.0, atol=1e-6)
    assert np.isclose(far, 2.0, atol=1e-6), (
        "a single pixel scale would report the far edge as shorter; a homography "
        "must not"
    )


def test_homography_rejects_collinear_points():
    with pytest.raises(CalibrationError):
        Calibration.from_floor_points(
            "u1", [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 0), (1, 0), (2, 0), (3, 0)], 1280, 720,
        )
    with pytest.raises(CalibrationError):
        Calibration.from_floor_points("u1", [(0, 0)], [(0, 0)], 1280, 720)


def test_px_to_m_refused_for_homography():
    calibration = Calibration.from_floor_points(
        "u1", [(300, 600), (900, 600), (760, 380), (440, 380)],
        [(0, 0), (2, 0), (2, 4), (0, 4)], 1280, 720,
    )
    with pytest.raises(CalibrationError):
        calibration.px_to_m(100)


def test_row_roundtrip():
    original = Calibration.from_floor_points(
        "u1", [(300, 600), (900, 600), (760, 380), (440, 380)],
        [(0, 0), (2, 0), (2, 4), (0, 4)], 1280, 720,
    )
    original.expected_subject_px_height = 312.0
    restored = Calibration.from_row(original.to_row())
    np.testing.assert_allclose(restored.homography, original.homography)
    assert restored.expected_subject_px_height == 312.0
    assert restored.method == "homography"


# --------------------------------------------------------------------------
# camera-movement check
# --------------------------------------------------------------------------
def _series_with_subject_height(cfg, leg_length_px: float):
    raw, _ = synthetic_walk(n_strides=8, leg_length_px=leg_length_px)
    return to_pixels(raw, cfg)


def test_unchanged_geometry_passes(cfg):
    series = _series_with_subject_height(cfg, 160.0)
    calibration = Calibration.from_two_points(
        "u1", (0, 400), (500, 400), 2.0, series.video.width, series.video.height
    )
    calibration.expected_subject_px_height = subject_pixel_height(series)

    check = verify.check(calibration, series)
    assert check.usable
    assert abs(check.deviation_frac) < 1e-6


def test_moved_camera_is_detected(cfg):
    """A zoom or move rescales distances while leaving timing untouched -- which
    otherwise mimics a genuine gait-speed decline."""
    series = _series_with_subject_height(cfg, 160.0)
    calibration = Calibration.from_two_points(
        "u1", (0, 400), (500, 400), 2.0, series.video.width, series.video.height
    )
    # Calibrated when the subject filled 25% more of the frame.
    calibration.expected_subject_px_height = subject_pixel_height(series) * 1.25

    check = verify.check(calibration, series)
    assert not check.usable
    assert "camera has probably been moved" in check.reason


def test_resolution_mismatch_is_rejected(cfg):
    series = _series_with_subject_height(cfg, 160.0)
    calibration = Calibration.from_two_points("u1", (0, 400), (500, 400), 2.0, 640, 360)
    check = verify.check(calibration, series)
    assert not check.usable
    assert "does not transfer between resolutions" in check.reason


def test_missing_calibration_is_reported_not_raised(cfg):
    series = _series_with_subject_height(cfg, 160.0)
    check = verify.check(None, series)
    assert not check.usable
    assert "no calibration on file" in check.reason


def test_calibration_without_height_reference_still_usable(cfg):
    series = _series_with_subject_height(cfg, 160.0)
    calibration = Calibration.from_two_points(
        "u1", (0, 400), (500, 400), 2.0, series.video.width, series.video.height
    )
    check = verify.check(calibration, series)
    assert check.usable
    assert "cannot be detected" in check.reason
