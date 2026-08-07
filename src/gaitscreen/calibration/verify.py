"""Per-session validation of a stored calibration.

The brief's instruction to calibrate once and reuse the scale factor is right --
recalibrating every session would add its own drift and burden the operator. But
"once per camera setup" quietly assumes the camera stays put, and over 90 days in
someone's home a tripod gets nudged, raised, or moved to the other end of the
hallway. A moved camera rescales every distance-based metric while leaving the
timing metrics untouched, which is exactly the pattern of a real gait-speed
decline. It has to be detected, not assumed away.

The check is calibration-free: the subject's own pixel height is a scale proxy
that should be reproducible across sessions for the same camera geometry. A large
shift means the geometry changed, so distance-based metrics are suppressed rather
than reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..pose.to_pixels import subject_pixel_height
from ..types import PixelSeries
from .model import Calibration

#: Fractional change in subject pixel height treated as a moved camera.
DEFAULT_TOLERANCE_FRAC = 0.10


@dataclass
class CalibrationCheck:
    usable: bool
    observed_px_height: float
    expected_px_height: Optional[float]
    deviation_frac: Optional[float]
    reason: Optional[str] = None

    @property
    def note(self) -> Optional[str]:
        return self.reason


def check(
    calibration: Optional[Calibration],
    series: PixelSeries,
    *,
    tolerance_frac: float = DEFAULT_TOLERANCE_FRAC,
) -> CalibrationCheck:
    """Decide whether a stored calibration can be trusted for this recording."""
    observed = subject_pixel_height(series)

    if calibration is None:
        return CalibrationCheck(
            usable=False, observed_px_height=observed, expected_px_height=None,
            deviation_frac=None,
            reason=(
                "no calibration on file for this user, so distances cannot be "
                "converted to metres; scale-free metrics are unaffected"
            ),
        )

    if not calibration.applies_to(series.video.width, series.video.height):
        return CalibrationCheck(
            usable=False, observed_px_height=observed,
            expected_px_height=calibration.expected_subject_px_height,
            deviation_frac=None,
            reason=(
                f"calibration was made at {calibration.frame_width}x"
                f"{calibration.frame_height} but this video is "
                f"{series.video.width}x{series.video.height}; a pixel scale does "
                "not transfer between resolutions"
            ),
        )

    expected = calibration.expected_subject_px_height
    if expected is None or not np.isfinite(expected) or expected <= 0:
        return CalibrationCheck(
            usable=True, observed_px_height=observed, expected_px_height=expected,
            deviation_frac=None,
            reason=(
                "calibration has no stored subject-height reference, so camera "
                "movement between sessions cannot be detected"
            ),
        )

    if not np.isfinite(observed) or observed <= 0:
        return CalibrationCheck(
            usable=False, observed_px_height=observed, expected_px_height=expected,
            deviation_frac=None,
            reason="subject height could not be measured in this recording",
        )

    deviation = float(observed / expected - 1.0)
    if abs(deviation) > tolerance_frac:
        return CalibrationCheck(
            usable=False, observed_px_height=observed, expected_px_height=expected,
            deviation_frac=deviation,
            reason=(
                f"subject appears {abs(deviation):.0%} "
                f"{'larger' if deviation > 0 else 'smaller'} than at calibration "
                f"({observed:.0f}px vs {expected:.0f}px), so the camera has "
                "probably been moved or zoomed; metric distances are suppressed "
                "until recalibration"
            ),
        )

    return CalibrationCheck(
        usable=True, observed_px_height=observed, expected_px_height=expected,
        deviation_frac=deviation,
    )
