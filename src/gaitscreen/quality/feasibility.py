"""Whether gait speed is measurable from a given recording.

Gait speed is the first metric in the brief's priority list, and it is the one
most easily faked. It is computed from the subject's displacement in image space,
so it silently becomes meaningless whenever that displacement does not
correspond to real forward travel:

* treadmill walking or walking in place -- the subject never translates;
* a camera that pans to follow the subject -- the translation is cancelled;
* a zooming or handheld camera -- the scale changes mid-measurement.

In all three cases a naive pipeline still produces a number, and a small number
looks exactly like slow walking, which is the tool's high-risk signal. That
number would then enter the user's personal baseline and stay there. So speed is
refused with a stated reason instead of estimated.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..io.video import CameraMotion
from ..pose.schema import HIPS
from ..types import PixelSeries, SpeedFeasibility
from . import view as view_module
from .view import ViewClassification


def assess_speed(
    series: PixelSeries, camera_motion: CameraMotion, cfg: Config,
    *, view: "ViewClassification | None" = None,
) -> SpeedFeasibility:
    """Decide whether image-space displacement reflects real forward travel."""
    width = series.video.width
    hip_x = series.midpoint(*HIPS)[:, 0]
    finite = hip_x[np.isfinite(hip_x)]

    translation_px = float(np.ptp(finite)) if finite.size else 0.0
    translation_frac = translation_px / width if width else 0.0

    camera_px = camera_motion.net_px
    camera_frac = camera_px / width if width and np.isfinite(camera_px) else float("nan")

    min_translation = float(cfg["speed.min_subject_translation_frac"])
    max_camera = float(cfg["speed.max_camera_motion_frac"])

    if translation_frac < min_translation:
        # Someone walking straight at the camera also fails this test, having
        # covered real ground the whole time. Saying they walked in place would
        # send them to fix the wrong thing, so name what actually happened.
        if view is not None and view.kind == view_module.CORONAL:
            reason = (
                "the subject walks towards and away from the camera rather than "
                "across it, so their travel is along the camera's line of sight "
                "where distance cannot be recovered from the image. Gait speed "
                "and step length are not reported. Film from the side of the "
                "walking path to measure them."
            )
        else:
            reason = (
                f"subject moves only {translation_frac:.0%} of the frame width "
                f"across the whole recording (at least {min_translation:.0%} is "
                "required). This is treadmill or in-place walking, or the camera "
                "is following the subject; either way the recording contains no "
                "measurable forward travel, so gait speed and step length in "
                "metres are not reported. Timing metrics are unaffected."
            )
        return SpeedFeasibility(
            feasible=False,
            subject_translation_px=translation_px,
            subject_translation_frac=translation_frac,
            camera_motion_px=camera_px,
            camera_motion_frac=camera_frac,
            reason=reason,
        )

    if camera_motion.determinate and camera_frac > max_camera:
        reason = (
            f"the camera itself moved {camera_frac:.0%} of the frame width "
            f"(limit {max_camera:.0%}), so the subject's apparent displacement is "
            "partly camera motion. Gait speed is not reported. Re-record with the "
            "camera on a fixed tripod."
        )
        return SpeedFeasibility(
            feasible=False,
            subject_translation_px=translation_px,
            subject_translation_frac=translation_frac,
            camera_motion_px=camera_px,
            camera_motion_frac=camera_frac,
            reason=reason,
        )

    reason = None
    if not camera_motion.determinate:
        reason = (
            "camera motion could not be verified because the background has too "
            "little texture to track; gait speed assumes a fixed camera"
        )

    return SpeedFeasibility(
        feasible=True,
        subject_translation_px=translation_px,
        subject_translation_frac=translation_frac,
        camera_motion_px=camera_px,
        camera_motion_frac=camera_frac,
        reason=reason,
    )
