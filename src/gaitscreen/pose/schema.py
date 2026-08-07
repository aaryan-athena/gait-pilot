"""MediaPipe BlazePose landmark schema (33 landmarks).

Index order is fixed by the model; see the enum below. Two properties of the
raw output drive design decisions elsewhere in this package:

* ``x`` is normalised by image *width* and ``y`` by image *height*
  independently, so raw normalised coordinates are **anisotropic**. Any
  distance or angle computed before scaling back to pixels is silently wrong on
  non-square frames. See :mod:`gaitscreen.pose.to_pixels`.
* ``z`` is a depth estimate relative to the hip midpoint, in units that are
  neither metric nor reliable. It is archived but never computed from.
"""
from __future__ import annotations

from enum import IntEnum

N_LANDMARKS = 33


class PL(IntEnum):
    """BlazePose landmark indices. Left/right are anatomical, not image-side."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


LANDMARK_NAMES = [pl.name.lower() for pl in PL]

#: Per-side joint chains, so downstream code never hardcodes an index.
SIDE_LANDMARKS: dict[str, dict[str, PL]] = {
    "left": {
        "shoulder": PL.LEFT_SHOULDER,
        "elbow": PL.LEFT_ELBOW,
        "wrist": PL.LEFT_WRIST,
        "hip": PL.LEFT_HIP,
        "knee": PL.LEFT_KNEE,
        "ankle": PL.LEFT_ANKLE,
        "heel": PL.LEFT_HEEL,
        "foot_index": PL.LEFT_FOOT_INDEX,
    },
    "right": {
        "shoulder": PL.RIGHT_SHOULDER,
        "elbow": PL.RIGHT_ELBOW,
        "wrist": PL.RIGHT_WRIST,
        "hip": PL.RIGHT_HIP,
        "knee": PL.RIGHT_KNEE,
        "ankle": PL.RIGHT_ANKLE,
        "heel": PL.RIGHT_HEEL,
        "foot_index": PL.RIGHT_FOOT_INDEX,
    },
}

SIDES = ("left", "right")

#: Landmarks whose visibility gates gait-event detection. These are the ones
#: that actually matter for segmentation, so quality scoring weights them.
GAIT_CRITICAL = (
    PL.LEFT_HIP, PL.RIGHT_HIP,
    PL.LEFT_KNEE, PL.RIGHT_KNEE,
    PL.LEFT_ANKLE, PL.RIGHT_ANKLE,
    PL.LEFT_HEEL, PL.RIGHT_HEEL,
    PL.LEFT_FOOT_INDEX, PL.RIGHT_FOOT_INDEX,
)

#: Used for the trunk segment and the pelvis reference frame.
TRUNK = (PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, PL.LEFT_HIP, PL.RIGHT_HIP)

HIPS = (PL.LEFT_HIP, PL.RIGHT_HIP)
SHOULDERS = (PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER)


def other_side(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}")
    return "right" if side == "left" else "left"
