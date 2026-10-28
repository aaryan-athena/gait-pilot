"""Camera sources: each yields (BGR frame, capture time in seconds).

Three backends behind one interface, chosen by ``camera.backend``:

* ``picamera2`` -- a Raspberry Pi camera on the ribbon connector. Capture time
  comes from the sensor's own timestamp, which is taken when the exposure
  happens, not when Python gets round to reading the frame. That matters on a
  busy device: a frame read late is still stamped when it was taken.
* ``opencv`` -- a USB webcam through V4L2. USB cameras do not hand over a
  usable sensor timestamp through OpenCV, so frames are stamped on arrival.
  Most only reach 60 fps when asked for MJPEG, so that is requested.
* ``file`` -- replays a recording, stamped with its own timestamps, optionally
  at real speed. This is how the whole device loop is tested on a laptop.

The analysis needs the person to be big enough in frame and the camera to stay
put; nothing here can check that, which is what the recording feedback in the
analysis is for.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from .settings import Settings

_ROTATIONS = {90: 0, 180: 1, 270: 2}  # -> cv2.ROTATE_* codes


class CameraError(RuntimeError):
    pass


class FrameSource:
    width: int
    height: int
    fps: float

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_source(settings: Settings) -> FrameSource:
    backend = settings["camera.backend"]
    if backend == "picamera2":
        source: FrameSource = Picamera2Source(settings)
    elif backend == "opencv":
        source = OpenCVSource(settings)
    else:
        source = FileSource(settings)
    rotate = int(settings["camera.rotate"])
    return Rotated(source, rotate) if rotate else source


# --------------------------------------------------------------------------
class Picamera2Source(FrameSource):
    def __init__(self, settings: Settings):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise CameraError(
                "picamera2 is not available. On Raspberry Pi OS install it with "
                "`sudo apt install python3-picamera2` and create the virtual "
                "environment with --system-site-packages (see deploy/install_pi.sh)."
            ) from exc
        self.width = int(settings["camera.width"])
        self.height = int(settings["camera.height"])
        self.fps = float(settings["camera.fps"])
        frame_us = int(round(1_000_000 / self.fps))
        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            # "RGB888" in libcamera is BGR byte order in memory: what OpenCV
            # and the recorder expect, with no conversion.
            main={"size": (self.width, self.height), "format": "RGB888"},
            # Fixed frame duration: auto-exposure may otherwise lengthen frames
            # in dim light, which is a frame-rate drop by another name.
            controls={"FrameDurationLimits": (frame_us, frame_us)},
            buffer_count=6,
        )
        self._cam.configure(config)
        self._cam.start()

    def frames(self):
        while True:
            request = self._cam.capture_request()
            try:
                image = request.make_array("main")
                stamp_ns = request.get_metadata().get("SensorTimestamp")
            finally:
                request.release()
            t = stamp_ns / 1e9 if stamp_ns else time.monotonic()
            yield image, t

    def close(self) -> None:
        try:
            self._cam.stop()
            self._cam.close()
        except Exception:  # noqa: BLE001
            pass


class OpenCVSource(FrameSource):
    def __init__(self, settings: Settings):
        import cv2

        index = settings["camera.index"]
        api = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") and _is_linux() else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, api)
        if not self._cap.isOpened():
            raise CameraError(f"could not open camera {index}. {_v4l2_hint(index)}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(settings["camera.width"]))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(settings["camera.height"]))
        self._cap.set(cv2.CAP_PROP_FPS, float(settings["camera.fps"]))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or settings["camera.fps"])

    def frames(self):
        while True:
            ok, image = self._cap.read()
            if not ok:
                raise CameraError("camera stopped delivering frames")
            yield image, time.monotonic()

    def close(self) -> None:
        self._cap.release()


class FileSource(FrameSource):
    """Replays a recording as if it were a camera, with its real timestamps."""

    def __init__(self, settings: Settings):
        import cv2

        self.path = settings["camera.source"]
        self.realtime = bool(settings["camera.realtime"])
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise CameraError(f"could not open {self.path}")
        native = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or settings["camera.fps"])
        cap.release()
        # Frames are delivered at the configured size, as a camera would be.
        self.width, self.height = int(settings["camera.width"]), int(settings["camera.height"])
        self._resize = native != (self.width, self.height)

    def frames(self):
        import cv2

        from gaitscreen.io.video import timed_frames

        start_wall: Optional[float] = None
        first_stamp: Optional[float] = None
        for _, image, stamp_ms in timed_frames(self.path):
            t = stamp_ms / 1000.0
            if first_stamp is None:
                first_stamp, start_wall = t, time.monotonic()
            if self.realtime:
                delay = (t - first_stamp) - (time.monotonic() - start_wall)
                if delay > 0:
                    time.sleep(delay)
            if self._resize:
                image = cv2.resize(image, (self.width, self.height),
                                   interpolation=cv2.INTER_AREA)
            yield image, t


class Rotated(FrameSource):
    def __init__(self, inner: FrameSource, degrees: int):
        self.inner = inner
        self._code = _ROTATIONS[degrees]
        swap = degrees in (90, 270)
        self.width = inner.height if swap else inner.width
        self.height = inner.width if swap else inner.height
        self.fps = inner.fps

    def frames(self):
        import cv2

        for image, t in self.inner.frames():
            yield cv2.rotate(image, self._code), t

    def close(self) -> None:
        self.inner.close()


def _is_linux() -> bool:
    import sys

    return sys.platform.startswith("linux")


def _v4l2_hint(index) -> str:
    """Why a USB camera did not open, in terms someone at the device can act on."""
    import glob
    import os

    if not _is_linux():
        return ""
    device = index if isinstance(index, str) else f"/dev/video{index}"
    stable = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    listing = ", ".join(stable) or "none found"
    if not os.path.exists(device):
        return (f"{device} does not exist: the camera is unplugged or was given another "
                f"number. USB cameras present: {listing}. Set camera.index to one of "
                "these paths so the number cannot change.")
    return (f"{device} exists but would not open: another program is probably using it "
            f"(check with `sudo fuser -v {device}`), or it is not a capture device. "
            f"USB cameras present: {listing}.")
