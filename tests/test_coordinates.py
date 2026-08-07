"""Coordinate-frame conversion.

The aspect-ratio bug these guard against is silent: everything still runs, joint
angles are simply wrong by a factor that depends on the frame shape.
"""
import numpy as np

from gaitscreen.pose.schema import PL
from gaitscreen.pose.to_pixels import leg_length_px, subject_pixel_height, to_pixels
from fixtures.synthetic import occlude, synthetic_walk


def test_axes_are_isotropic_after_conversion(cfg):
    """A square displacement in pixels must stay square through the conversion."""
    raw, _ = synthetic_walk(width=1280, height=720, n_strides=4)
    # Place two landmarks 100px apart horizontally, then 100px apart vertically.
    raw.xy[0, int(PL.NOSE)] = [0.5, 0.5]
    raw.xy[0, int(PL.LEFT_WRIST)] = [0.5 + 100 / 1280, 0.5]
    raw.xy[0, int(PL.RIGHT_WRIST)] = [0.5, 0.5 + 100 / 720]

    series = to_pixels(raw, cfg)
    horizontal = np.linalg.norm(
        series.xy[0, int(PL.LEFT_WRIST)] - series.xy[0, int(PL.NOSE)]
    )
    vertical = np.linalg.norm(
        series.xy[0, int(PL.RIGHT_WRIST)] - series.xy[0, int(PL.NOSE)]
    )
    np.testing.assert_allclose(horizontal, 100.0, atol=1e-6)
    np.testing.assert_allclose(vertical, 100.0, atol=1e-6)


def test_vertical_axis_is_flipped_to_y_up(cfg):
    raw, _ = synthetic_walk(n_strides=4)
    series = to_pixels(raw, cfg)
    # The nose is anatomically above the ankles, so it must have a larger y.
    assert series.xy[0, int(PL.NOSE), 1] > series.xy[0, int(PL.LEFT_ANKLE), 1]


def test_low_visibility_samples_become_nan_not_dropped(cfg):
    raw, _ = synthetic_walk(n_strides=6)
    n_before = raw.n_frames
    occlude(raw, PL.LEFT_ANKLE, 10, 20, visibility=0.1)

    series = to_pixels(raw, cfg)
    assert series.n_frames == n_before, "frames must never be dropped"
    assert np.isnan(series.xy[10:20, int(PL.LEFT_ANKLE), 0]).all()
    assert np.isfinite(series.xy[10:20, int(PL.RIGHT_ANKLE), 0]).all()


def test_scale_proxies_are_stable_across_the_walk(cfg):
    raw, truth = synthetic_walk(n_strides=10, leg_length_px=160.0)
    series = to_pixels(raw, cfg)
    # Leg length recovered from the 95th percentile of hip-ankle distance.
    assert 0.85 * truth.leg_length_px < leg_length_px(series) < 1.15 * truth.leg_length_px
    assert np.isfinite(subject_pixel_height(series))
