"""Deciding when a walk starts and stops.

The device cannot afford to run pose estimation on every frame just to notice
that someone has walked in -- on a Pi 5 that alone would take most of the CPU
the recorder needs. So the trigger is deliberately crude: background
subtraction on a thumbnail about 160 pixels wide, which costs well under a
millisecond a frame.

Crude is acceptable because the trigger is not the judge. It only has to avoid
*missing* walks; a false start (a door, a shadow, someone stopping to chat) is
caught later, when the analysis finds no gait in the clip and the clip is
discarded. So it errs towards recording, and keeps a pre-roll buffer so the
first steps are not lost to the few frames it takes to be sure.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TriggerEvent:
    kind: str                       # "start" | "frame" | "stop"
    frames: list                    # (image, t) to write with this event
    reason: str = ""


class MotionTrigger:
    """Background-subtraction start/stop with pre-roll."""

    def __init__(self, settings, fps: float):
        import cv2

        cfg = settings.data["trigger"]
        self.analysis_width = int(cfg["analysis_width"])
        self.min_moving = float(cfg["min_moving_frac"])
        self.start_frames = int(cfg["start_frames"])
        self.stop_after = float(cfg["stop_after_s"])
        self.max_clip = float(cfg["max_clip_s"])
        self.warmup = float(cfg["warmup_s"])
        self._pre_roll: deque = deque(maxlen=max(1, int(round(cfg["pre_roll_s"] * fps))))
        # Background model with a long memory: a person walking through moves
        # too fast to be absorbed into it, but slow lighting changes are.
        self._bg = cv2.createBackgroundSubtractorMOG2(history=int(fps * 20),
                                                      varThreshold=32,
                                                      detectShadows=False)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._streak = 0
        self._recording = False
        self._clip_start: Optional[float] = None
        self._last_motion: Optional[float] = None
        self._first_t: Optional[float] = None
        self.last_moving_frac = 0.0

    @property
    def recording(self) -> bool:
        return self._recording

    def moving_fraction(self, image: np.ndarray) -> float:
        import cv2

        height = int(round(image.shape[0] * self.analysis_width / image.shape[1]))
        small = cv2.resize(image, (self.analysis_width, max(1, height)),
                           interpolation=cv2.INTER_AREA)
        mask = self._bg.apply(small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return float(np.count_nonzero(mask)) / mask.size

    def feed(self, image: np.ndarray, t: float) -> TriggerEvent:
        if self._first_t is None:
            self._first_t = t
        frac = self.moving_fraction(image)
        self.last_moving_frac = frac
        moving = frac >= self.min_moving and (t - self._first_t) >= self.warmup

        if not self._recording:
            self._pre_roll.append((image, t))
            self._streak = self._streak + 1 if moving else 0
            if self._streak >= self.start_frames:
                self._recording = True
                self._clip_start = self._pre_roll[0][1]
                self._last_motion = t
                frames = list(self._pre_roll)
                self._pre_roll.clear()
                return TriggerEvent("start", frames, "motion")
            return TriggerEvent("idle", [])

        if moving:
            self._last_motion = t
        if t - self._clip_start >= self.max_clip:
            return self._stop("maximum clip length reached")
        if t - self._last_motion >= self.stop_after:
            return self._stop("motion stopped")
        return TriggerEvent("frame", [(image, t)])

    def flush(self) -> Optional[TriggerEvent]:
        """End an open clip when the camera stops (shutdown, end of file)."""
        return self._stop("camera stopped") if self._recording else None

    def _stop(self, reason: str) -> TriggerEvent:
        self._recording = False
        self._streak = 0
        self._clip_start = None
        return TriggerEvent("stop", [], reason)


class ContinuousTrigger:
    """Fixed-length back-to-back clips, for a corridor with constant traffic."""

    def __init__(self, settings, fps: float):
        self.length = float(settings["trigger.max_clip_s"])
        self._start: Optional[float] = None

    def feed(self, image, t) -> TriggerEvent:
        if self._start is None:
            self._start = t
            return TriggerEvent("start", [(image, t)], "continuous")
        if t - self._start >= self.length:
            self._start = None
            return TriggerEvent("stop", [], "clip length reached")
        return TriggerEvent("frame", [(image, t)])

    def flush(self) -> Optional[TriggerEvent]:
        return TriggerEvent("stop", [], "camera stopped") if self._start else None


def make_trigger(settings, fps: float):
    mode = settings["trigger.mode"]
    if mode == "continuous":
        return ContinuousTrigger(settings, fps)
    return MotionTrigger(settings, fps)
