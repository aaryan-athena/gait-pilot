"""Build a stage-1 :class:`Extraction` from synthetic landmarks, without video.

Lets the feature and diagnostic layers be tested against known ground truth
without running pose estimation, which keeps the suite fast enough to run on
every change.
"""
from __future__ import annotations

from gaitscreen.calibration.model import Calibration
from gaitscreen.calibration.verify import check as check_calibration
from gaitscreen.io.video import CameraMotion
from gaitscreen.pipeline import Extraction
from gaitscreen.pose.to_pixels import subject_pixel_height, to_pixels
from gaitscreen.quality import feasibility
from gaitscreen.signal import filters, resample

from .synthetic import synthetic_walk

STATIC_CAMERA = CameraMotion(net_px=1.0, path_px=4.0, determinate=True, n_samples=20)


def build_extraction(cfg, *, calibrate: bool = False, camera=None, **kwargs):
    """Return ``(Extraction, GroundTruth)`` for a synthetic walk."""
    raw, truth = synthetic_walk(**kwargs)
    series = to_pixels(raw, cfg)
    series, gaps = resample.fill_short_gaps(series, cfg)
    series = filters.smooth_series(series, cfg)
    segments = resample.analysis_segments(series, cfg)

    calibration = None
    if calibrate:
        scale = truth.scale_m_per_px
        calibration = Calibration.from_two_points(
            "u1", (0.0, 0.0), (1.0 / scale, 0.0), 1.0,
            series.video.width, series.video.height,
        )
        calibration.expected_subject_px_height = subject_pixel_height(series)

    motion = camera if camera is not None else STATIC_CAMERA
    return Extraction(
        info=series.video,
        raw=raw,
        series=series,
        segments=segments,
        camera_motion=motion,
        speed=feasibility.assess_speed(series, motion, cfg),
        calibration=calibration,
        calibration_check=check_calibration(calibration, series),
        gap_summary=gaps,
        subject_px_height=subject_pixel_height(series),
    ), truth
