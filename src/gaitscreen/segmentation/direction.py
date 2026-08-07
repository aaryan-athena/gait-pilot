"""Direction of progression and splitting a recording into walk passes.

Every spatial gait measure is defined along the *anterior* axis, not along image
``x``. Getting that axis wrong flips heel strike and toe-off, so it is resolved
explicitly rather than assumed.

Direction is taken from the feet, not from the hips. Hip velocity is the obvious
choice and fails on exactly the footage this project has: treadmill, in-place and
camera-tracked recordings have no net hip translation, so its sign is noise. The
foot's own anterior axis -- toe ahead of heel -- is a body-fixed reference that
survives all three cases. Facing direction from the head is used as a tie-break,
and hip velocity only as a last resort.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..pose.schema import HIPS, PL, SIDE_LANDMARKS
from ..types import PixelSeries, WalkPass


def anterior_sign(series: PixelSeries) -> tuple[int, str, float]:
    """Resolve which image direction is 'forward'.

    Returns ``(sign, method, agreement)`` where ``sign`` is +1 or -1, and
    ``agreement`` is the fraction of frames supporting it (a confidence proxy).
    """
    for method, values in (
        ("foot_axis", _foot_axis_sign(series)),
        ("facing", _facing_sign(series)),
        ("hip_velocity", _hip_velocity_sign(series)),
    ):
        finite = values[np.isfinite(values)]
        if finite.size < 5:
            continue
        positive = float(np.mean(finite > 0))
        agreement = max(positive, 1.0 - positive)
        # A near-50/50 split means the cue carries no information here.
        if agreement >= 0.65:
            return (1 if positive >= 0.5 else -1, method, agreement)

    return 1, "assumed", 0.0


def _foot_axis_sign(series: PixelSeries) -> np.ndarray:
    """Toe is anterior to heel; averaged over both feet and all frames."""
    signs = []
    for side in ("left", "right"):
        toe = series.point(SIDE_LANDMARKS[side]["foot_index"])[:, 0]
        heel = series.point(SIDE_LANDMARKS[side]["heel"])[:, 0]
        signs.append(toe - heel)
    return np.nanmean(np.column_stack(signs), axis=1)


def _facing_sign(series: PixelSeries) -> np.ndarray:
    """Nose is anterior to the ears."""
    nose = series.point(PL.NOSE)[:, 0]
    ears = 0.5 * (series.point(PL.LEFT_EAR)[:, 0] + series.point(PL.RIGHT_EAR)[:, 0])
    return nose - ears


def _hip_velocity_sign(series: PixelSeries) -> np.ndarray:
    hip_x = series.midpoint(*HIPS)[:, 0]
    return np.gradient(hip_x)


def hip_velocity(series: PixelSeries, *, smooth_s: float = 0.3) -> np.ndarray:
    """Pelvis horizontal velocity in px/s, smoothed over a fraction of a stride."""
    hip_x = series.midpoint(*HIPS)[:, 0]
    window = max(3, int(round(smooth_s * series.fps)) | 1)
    kernel = np.ones(window) / window
    padded = np.pad(hip_x, window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[: hip_x.size]
    return np.gradient(smoothed) * series.fps


def find_passes(
    series: PixelSeries, cfg: Config, segments: list[tuple[int, int]]
) -> list[WalkPass]:
    """Split each analysis segment into passes of consistent travel direction.

    A direction reversal is a turn. Turn strides have atypical timing and would
    inflate stride-time variability, so the turn itself is excluded and the walk
    either side becomes a separate pass.

    When the subject does not translate (treadmill, in-place, camera-tracked),
    there is no turn to find and the whole segment is one pass, oriented by the
    body-fixed anterior axis.
    """
    sign, method, _ = anterior_sign(series)
    min_frames = max(
        4, int(round(float(cfg["segmentation.max_stride_time_s"]) * series.fps))
    )
    passes: list[WalkPass] = []

    for start, stop in segments:
        window = series.slice_frames(start, stop)
        velocity = hip_velocity(window)

        # Only treat direction changes as turns if the subject actually travels.
        travel_frac = float(np.ptp(window.midpoint(*HIPS)[:, 0])) / max(series.video.width, 1)
        if method == "hip_velocity" or travel_frac >= float(
            cfg["speed.min_subject_translation_frac"]
        ):
            passes.extend(
                _split_on_reversals(velocity, start, min_frames, series, cfg)
            )
        else:
            passes.append(
                WalkPass(
                    start_frame=start, end_frame=stop, direction=sign,
                    camera_side=infer_camera_side(window),
                )
            )

    return passes


def _split_on_reversals(
    velocity: np.ndarray, offset: int, min_frames: int, series: PixelSeries, cfg: Config
) -> list[WalkPass]:
    """Runs of consistent travel direction, with near-stationary turns dropped."""
    still = max(1, int(cfg["segmentation.turn_min_still_frames"]))
    # Anything slower than a slow shuffle counts as 'not travelling'.
    threshold = 0.05 * series.video.width / max(series.fps, 1) * series.fps * 0.1
    moving_sign = np.where(np.abs(velocity) < threshold, 0, np.sign(velocity)).astype(int)

    passes: list[WalkPass] = []
    run_start = 0
    for index in range(1, moving_sign.size + 1):
        at_end = index == moving_sign.size
        if at_end or moving_sign[index] != moving_sign[run_start]:
            length = index - run_start
            direction = int(moving_sign[run_start])
            if direction != 0 and length >= min_frames:
                window = series.slice_frames(offset + run_start, offset + index)
                passes.append(
                    WalkPass(
                        start_frame=offset + run_start,
                        end_frame=offset + index,
                        direction=direction,
                        camera_side=infer_camera_side(window),
                    )
                )
            elif direction == 0 and length < still:
                pass  # brief stall inside a pass, not a real turn
            run_start = index
    return passes


def infer_camera_side(series: PixelSeries) -> str | None:
    """Which body side faces the camera, from mean landmark visibility.

    Matters because the far limb is occluded through much of stance, so its
    measurements are systematically noisier. Recording one pass in each direction
    lets each limb be the near limb once.
    """
    scores = {}
    for side in ("left", "right"):
        indices = [
            int(SIDE_LANDMARKS[side][joint])
            for joint in ("hip", "knee", "ankle", "heel", "foot_index")
        ]
        scores[side] = float(np.nanmean(series.visibility[:, indices]))

    if not np.isfinite(list(scores.values())).all():
        return None
    near, far = ("left", "right") if scores["left"] >= scores["right"] else ("right", "left")
    # Require a real difference; MediaPipe often reports both sides confidently.
    return near if scores[near] - scores[far] > 0.03 else None
