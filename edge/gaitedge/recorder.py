"""Writing recorded walks to disk, with each frame's real capture time.

Two decisions here protect the measurement rather than the file:

* **Every frame is stamped with the moment the camera captured it**, not with
  its position in the file. A Raspberry Pi 5 compresses video on the CPU, and
  if the CPU falls behind, frames are dropped. The gait pipeline rebuilds its
  clock from these stamps (see ``place_on_grid`` in ``gaitscreen``), so a
  dropped frame becomes a visible gap rather than a silent compression of time
  that would mis-time every stride around it.
* **Recordings are Matroska (.mkv), not MP4.** An MP4 writes its index last,
  so one cut off by a power loss is unreadable; a Matroska file interrupted
  mid-walk plays up to the last frame written. A device on a wall socket in a
  care home will lose power mid-recording sooner or later.

Compression runs on its own thread behind a bounded queue, so a slow encoder
never stalls the camera. When the queue is full the newest frame is dropped
and counted, which the stamps then record as a gap.
"""
from __future__ import annotations

import queue
import threading
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

#: Timestamps are written in milliseconds. Finer than any camera's frame
#: interval, and the unit the gait pipeline reads them back in.
TIME_BASE = Fraction(1, 1000)


class ClipWriter:
    """Synchronous encoder: one frame in, one frame written."""

    def __init__(self, path: Path, width: int, height: int, *, fps: float,
                 codec: str = "h264", crf: int = 23, preset: str = "ultrafast"):
        import av

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(width), int(height)
        self._container = av.open(str(self.path), mode="w", format="matroska")
        if codec == "h264":
            stream = self._container.add_stream("libx264", rate=int(round(fps)))
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "preset": preset,
                              # Every frame its own decision: no B-frames means
                              # nothing is held back waiting for a later frame,
                              # so a cut-off file loses at most one frame.
                              "bf": "0", "g": str(int(round(fps)))}
        elif codec == "mjpeg":
            stream = self._container.add_stream("mjpeg", rate=int(round(fps)))
            stream.pix_fmt = "yuvj420p"
            stream.options = {"q:v": "4"}
        else:
            raise ValueError(f"unknown codec {codec!r}")
        stream.width, stream.height = self.width, self.height
        stream.time_base = TIME_BASE
        stream.codec_context.time_base = TIME_BASE
        self._stream = stream
        self._t0: Optional[float] = None
        self._last_pts = -1
        self.frames = 0
        self.first_t: Optional[float] = None
        self.last_t: Optional[float] = None

    def write(self, image_bgr: np.ndarray, t: float) -> None:
        """Append one frame captured at ``t`` seconds (any monotonic clock)."""
        import av

        if image_bgr.shape[1] != self.width or image_bgr.shape[0] != self.height:
            raise ValueError(
                f"frame is {image_bgr.shape[1]}x{image_bgr.shape[0]}, "
                f"writer expects {self.width}x{self.height}"
            )
        if self._t0 is None:
            self._t0 = t
            self.first_t = t
        pts = int(round((t - self._t0) * 1000.0))
        # Timestamps must strictly increase. Two frames inside one millisecond
        # can only come from a clock glitch; nudging by a millisecond keeps the
        # file valid without inventing a gap.
        if pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts
        frame = av.VideoFrame.from_ndarray(image_bgr, format="bgr24")
        frame.pts = pts
        frame.time_base = TIME_BASE
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames += 1
        self.last_t = t

    @property
    def duration_s(self) -> float:
        if self.first_t is None or self.last_t is None:
            return 0.0
        return self.last_t - self.first_t

    def close(self) -> None:
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()


class BackgroundWriter:
    """A ClipWriter on its own thread, so the camera loop never waits on it."""

    def __init__(self, writer: ClipWriter, *, max_queue: int = 240):
        self.writer = writer
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="clip-writer", daemon=True)
        self._thread.start()

    def write(self, image_bgr: np.ndarray, t: float) -> None:
        try:
            self._queue.put_nowait((image_bgr, t))
        except queue.Full:
            # Never block the camera. A dropped frame here is a gap in the
            # timestamps, which the analysis already knows how to handle.
            self.dropped += 1

    def close(self, timeout: float = 120.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout)
        if self.error is not None:
            raise RuntimeError(f"recording failed: {self.error}") from self.error

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self.writer.write(*item)
        except BaseException as exc:  # noqa: BLE001 - reported from close()
            self.error = exc
            # Drain so producers never block on a dead writer.
            while not self._queue.empty():
                self._queue.get_nowait()
        finally:
            try:
                self.writer.close()
            except BaseException as exc:  # noqa: BLE001
                self.error = self.error or exc
