"""Conversion from raw MediaPipe output to the analysis coordinate frame.

This module exists because of one easily-missed property of the raw output:
``x`` is normalised by image width and ``y`` by image height *independently*.
On a 768x432 frame, one unit of normalised ``x`` is 768 px while one unit of
normalised ``y`` is 432 px, so any distance, velocity or joint angle computed
from raw normalised coordinates is wrong by the aspect ratio -- wrong in a way
that still looks entirely reasonable. Everything downstream therefore works in
pixels.

The vertical axis is also flipped to y-up here, so "higher" means a larger
``y`` and the sign conventions match the biomechanics literature.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import PixelSeries, RawLandmarks


def to_pixels(raw: RawLandmarks, cfg: Config) -> PixelSeries:
    """Scale to isotropic pixels, flip to y-up, and build the validity mask.

    A sample is valid when a pose was detected in that frame, the landmark's
    visibility meets the threshold, and the coordinates are finite. Invalid
    samples are set to NaN here and handled by :mod:`gaitscreen.signal.resample`
    -- never dropped, because dropping frames breaks the uniform time base.
    """
    width, height = raw.video.width, raw.video.height
    threshold = float(cfg["landmarks.visibility_threshold"])

    xy = np.empty_like(raw.xy)
    xy[..., 0] = raw.xy[..., 0] * width
    xy[..., 1] = height - raw.xy[..., 1] * height  # flip to y-up

    valid = (
        raw.detected[:, None]
        & (raw.visibility >= threshold)
        & np.isfinite(xy).all(axis=2)
    )
    xy[~valid] = np.nan

    return PixelSeries(
        t=raw.t.copy(),
        xy=xy,
        visibility=raw.visibility.copy(),
        valid=valid,
        video=raw.video,
    )


def subject_pixel_height(series: PixelSeries) -> float:
    """Median nose-to-lowest-ankle distance in pixels.

    A stable, calibration-free scale proxy. Used for two things: normalising
    length metrics when no metric calibration is available, and detecting that
    the camera has been moved between sessions (see
    :mod:`gaitscreen.calibration.verify`).
    """
    from ..pose.schema import PL

    nose_y = series.xy[:, PL.NOSE, 1]
    ankles = series.xy[:, [PL.LEFT_ANKLE, PL.RIGHT_ANKLE], 1]
    # Frames where neither ankle was tracked are expected in poor recordings;
    # nanmin warns on an all-NaN slice, and NaN is already the right answer.
    both_missing = np.isnan(ankles).all(axis=1)
    ankle_y = np.full(ankles.shape[0], np.nan)
    if (~both_missing).any():
        ankle_y[~both_missing] = np.nanmin(ankles[~both_missing], axis=1)
    heights = nose_y - ankle_y
    heights = heights[np.isfinite(heights)]
    return float(np.median(heights)) if heights.size else float("nan")


def leg_length_px(series: PixelSeries) -> float:
    """Median hip-to-ankle distance in pixels, averaged over both sides.

    Preferred over full body height for normalising step length, since step
    length scales with leg length.
    """
    from ..pose.schema import SIDE_LANDMARKS

    lengths = []
    for side in ("left", "right"):
        hip = series.joint(side, "hip")
        ankle = series.joint(side, "ankle")
        d = np.linalg.norm(hip - ankle, axis=1)
        d = d[np.isfinite(d)]
        if d.size:
            # Max over the cycle approximates the fully-extended limb; the
            # median of per-frame distances underestimates it during flexion.
            lengths.append(float(np.percentile(d, 95)))
    _ = SIDE_LANDMARKS  # documented dependency
    return float(np.mean(lengths)) if lengths else float("nan")
