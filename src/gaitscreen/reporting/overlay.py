"""Render the source video with the tracking and detected events drawn on it.

This exists for transparency rather than decoration. Everything the tool reports
rests on two things a viewer cannot otherwise check: that the skeleton followed
the right person's joints, and that heel strikes were detected at the moments the
foot actually landed. A table of numbers cannot show either. A video with the
skeleton and the events drawn on it shows both in a few seconds, to someone with
no training in gait analysis.

**Encoding.** Frames are piped to ffmpeg and encoded with libx264, because that
is the only *reliably* browser-playable option.

OpenCV's own H.264 writer is the fallback, and it cannot be trusted at face
value: on this machine it succeeds at 640x360 but at 160x120 it reports
``isOpened() == True``, logs an openh264 load failure, and then writes about a
kilobyte containing nothing. So the fallback validates the size of what it
actually produced rather than believing the API. Its last resort, ``mp4v``, is
MPEG-4 Part 2, which most browsers refuse to play -- that path still returns a
file, but says it is not playable so the app can offer a download instead of a
dead player.

One platform trap, recorded because it cost a deployment: ``Popen.communicate()``
must be left to close the stdin pipe itself. See :meth:`_FfmpegWriter.close`.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ..config import Config
from ..pose.schema import PL, SIDE_LANDMARKS
from ..types import GaitCycle, GaitEvent

#: Limbs to draw, as landmark pairs. Legs are drawn thicker than arms because
#: the legs are what the measurements come from.
SKELETON: tuple[tuple[PL, PL, str], ...] = (
    (PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, "trunk"),
    (PL.LEFT_SHOULDER, PL.LEFT_HIP, "trunk"),
    (PL.RIGHT_SHOULDER, PL.RIGHT_HIP, "trunk"),
    (PL.LEFT_HIP, PL.RIGHT_HIP, "trunk"),
    (PL.LEFT_SHOULDER, PL.LEFT_ELBOW, "left_arm"),
    (PL.LEFT_ELBOW, PL.LEFT_WRIST, "left_arm"),
    (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, "right_arm"),
    (PL.RIGHT_ELBOW, PL.RIGHT_WRIST, "right_arm"),
    (PL.LEFT_HIP, PL.LEFT_KNEE, "left_leg"),
    (PL.LEFT_KNEE, PL.LEFT_ANKLE, "left_leg"),
    (PL.LEFT_ANKLE, PL.LEFT_HEEL, "left_leg"),
    (PL.LEFT_HEEL, PL.LEFT_FOOT_INDEX, "left_leg"),
    (PL.RIGHT_HIP, PL.RIGHT_KNEE, "right_leg"),
    (PL.RIGHT_KNEE, PL.RIGHT_ANKLE, "right_leg"),
    (PL.RIGHT_ANKLE, PL.RIGHT_HEEL, "right_leg"),
    (PL.RIGHT_HEEL, PL.RIGHT_FOOT_INDEX, "right_leg"),
)

# OpenCV colours are **BGR**, not RGB. Written as (B, G, R) with the intended
# colour named, because getting this backwards is silent -- it produces a
# perfectly nice-looking video in the wrong colours, and the first version of
# this file drew the left leg in the same blue as the toe-off marker.
COLOUR = {
    "trunk": (215, 215, 215),      # light grey
    "left_arm": (170, 150, 120),   # muted brown
    "right_arm": (170, 150, 120),
    "left_leg": (40, 145, 255),    # orange
    "right_leg": (80, 200, 80),    # green
}
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
EVENT_STRIKE = (60, 60, 235)   # red
EVENT_TOEOFF = (235, 190, 60)  # cyan

#: Drawn into the frame rather than only shown beside it, so a downloaded or
#: shared clip still explains itself.
LEGEND = (
    ("left leg", COLOUR["left_leg"]),
    ("right leg", COLOUR["right_leg"]),
    ("heel down", EVENT_STRIKE),
    ("toe off", EVENT_TOEOFF),
)


@dataclass
class OverlayResult:
    path: Path
    codec: str
    browser_playable: bool
    n_frames: int
    note: Optional[str] = None


def render_overlay_video(
    extraction,
    analysis,
    output_path: str | Path,
    cfg: Config,
    *,
    max_width: int = 720,
    progress=None,
) -> OverlayResult:
    """Draw the skeleton and gait events over the source video.

    Reads the original file again rather than working from stored frames: the
    pipeline never keeps decoded frames, and re-reading is cheap next to pose
    estimation.
    """
    section = cfg.section("reporting.overlay")
    info = extraction.info
    series = extraction.series
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scale = min(1.0, max_width / max(info.width, 1))
    out_w = int(round(info.width * scale / 2) * 2)  # libx264 needs even dimensions
    out_h = int(round(info.height * scale / 2) * 2)

    events_by_frame = _events_by_frame(analysis.events, series, info.fps)
    cycle_spans = [(c, _frames_for(c, series)) for c in analysis.cycles]

    writer = _open_writer(output_path, info.fps, out_w, out_h,
                          crf=int(section["crf"]))
    frames_written = 0

    capture = cv2.VideoCapture(str(info.path))
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok or index >= series.n_frames:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
            _draw_frame(frame, series, index, scale, events_by_frame,
                        cycle_spans, analysis, info)
            writer.write(frame)
            frames_written += 1
            index += 1
            if progress is not None and index % 30 == 0:
                progress(index)
    finally:
        capture.release()
        result = writer.close()

    result.n_frames = frames_written
    return result


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def _draw_frame(frame, series, index, scale, events_by_frame, cycle_spans,
                analysis, info) -> None:
    height = frame.shape[0]

    def point(landmark) -> Optional[tuple[int, int]]:
        x, y_up = series.xy[index, int(landmark)]
        if not (np.isfinite(x) and np.isfinite(y_up)):
            return None
        # Stored coordinates are y-up in source pixels; the image is y-down.
        return int(round(x * scale)), int(round(height - y_up * scale))

    for a, b, group in SKELETON:
        pa, pb = point(a), point(b)
        if pa is None or pb is None:
            continue
        thickness = 3 if group.endswith("leg") else 2
        # Dark underlay first: pilot footage runs from bright sky to dark
        # tarmac, and no single line colour is visible against both.
        cv2.line(frame, pa, pb, BLACK, thickness + 3, cv2.LINE_AA)
        cv2.line(frame, pa, pb, COLOUR[group], thickness, cv2.LINE_AA)

    for side in ("left", "right"):
        colour = COLOUR[f"{side}_leg"]
        for joint in ("hip", "knee", "ankle", "heel", "foot_index"):
            landmark = SIDE_LANDMARKS[side][joint]
            p = point(landmark)
            if p is None:
                continue
            cv2.circle(frame, p, 6, BLACK, -1, cv2.LINE_AA)
            # A joint that was interpolated across a dropout is drawn hollow.
            # Drawing it solid would present an inferred position as an observed
            # one, which is the opposite of what this video is for.
            observed = bool(series.valid[index, int(landmark)])
            cv2.circle(frame, p, 4, colour, -1 if observed else 1, cv2.LINE_AA)

    _draw_events(frame, point, events_by_frame.get(index, []))
    _draw_banner(frame, index, series, cycle_spans, analysis, info)
    _draw_legend(frame)


def _draw_events(frame, point, events: Sequence[GaitEvent]) -> None:
    """Mark events on the frame they fall in, and label them.

    An event is drawn for a few frames either side of its own so it is visible
    at normal playback speed -- a single-frame marker at 60 fps is 17 ms on
    screen and effectively invisible.
    """
    for event, offset in events:
        joint = "heel" if event.kind == "heel_strike" else "foot_index"
        p = point(SIDE_LANDMARKS[event.side][joint])
        if p is None:
            continue
        strike = event.kind == "heel_strike"
        colour = EVENT_STRIKE if strike else EVENT_TOEOFF
        # Fades out over the frames following the event.
        radius = 10 + 4 * offset
        cv2.circle(frame, p, radius, colour, 2, cv2.LINE_AA)
        if offset == 0:
            label = "HEEL DOWN" if strike else "toe off"
            _text(frame, label, (p[0] - 34, p[1] - radius - 8), colour,
                  scale=0.45 if strike else 0.4)


def _draw_banner(frame, index, series, cycle_spans, analysis, info) -> None:
    """A caption strip: elapsed time, and whether this moment is being measured."""
    width = frame.shape[1]
    t = index / info.fps if info.fps else 0.0

    inside_valid = None
    for cycle, (start, stop) in cycle_spans:
        if start <= index < stop:
            inside_valid = cycle
            break

    if inside_valid is None:
        state, colour = "not measured here", (150, 150, 150)
    elif inside_valid.valid:
        state, colour = "measuring this step", (90, 200, 90)
    else:
        state, colour = "step skipped", (120, 170, 235)

    cv2.rectangle(frame, (0, 0), (width, 30), BLACK, -1)
    _text(frame, f"{t:5.1f}s", (8, 21), WHITE, scale=0.5)
    _text(frame, state, (76, 21), colour, scale=0.5)
    if inside_valid is not None and not inside_valid.valid:
        reason = (inside_valid.exclusion_reason or "").split("(")[0].strip()
        if reason:
            _text(frame, f"- {reason[:44]}", (206, 21), (170, 170, 170),
                  scale=0.42)


def _draw_legend(frame) -> None:
    """Colour key along the bottom edge."""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 24), (width, height), BLACK, -1)
    x = 8
    for label, colour in LEGEND:
        cv2.circle(frame, (x + 5, height - 12), 5, colour, -1, cv2.LINE_AA)
        _text(frame, label, (x + 15, height - 8), (235, 235, 235), scale=0.4)
        x += 22 + int(7.0 * len(label))
    _text(frame, "hollow joint = position estimated, not seen",
          (max(x + 6, width - 260), height - 8), (150, 150, 150), scale=0.36)


def _text(frame, text, origin, colour, *, scale=0.5) -> None:
    """Draw text with a dark outline so it survives any background."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK,
                3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                1, cv2.LINE_AA)


def _events_by_frame(events, series, fps) -> dict[int, list]:
    """Group events by the frames they should be visible on."""
    out: dict[int, list] = {}
    hold = max(2, int(round(0.08 * fps)))  # ~80 ms, visible at normal speed
    for event in events:
        base = int(np.searchsorted(series.t, event.t))
        for offset in range(hold):
            out.setdefault(base + offset, []).append((event, offset))
    return out


def _frames_for(cycle: GaitCycle, series) -> tuple[int, int]:
    start = int(np.searchsorted(series.t, cycle.t_start))
    stop = int(np.searchsorted(series.t, cycle.t_end))
    return start, max(stop, start + 1)


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
class _FfmpegWriter:
    """Pipes raw BGR frames to ffmpeg for H.264 encoding."""

    def __init__(self, path: Path, fps: float, width: int, height: int, crf: int):
        self.path = path
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
                "-i", "pipe:0",
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                # yuv420p and faststart are what make the result play in a
                # browser rather than merely being valid H.264.
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, ValueError) as exc:
            # ffmpeg has gone away mid-render. Its own stderr says why; a raw
            # pipe error does not.
            raise RuntimeError(self._exit_message()) from exc

    def close(self) -> OverlayResult:
        stderr = self._shutdown()
        if self._proc.returncode != 0:
            raise RuntimeError(self._exit_message(stderr))
        return OverlayResult(path=self.path, codec="h264", browser_playable=True,
                             n_frames=0)

    def _shutdown(self) -> bytes:
        """Signal end-of-input, drain ffmpeg's stderr, and reap the process.

        Deliberately does not use ``communicate()``. That helper flushes stdin
        before doing anything else and tolerates only ``BrokenPipeError``, so if
        stdin has already been closed -- which happens whenever this is reached
        twice, or after a write failed -- it raises
        ``ValueError("flush of closed file")`` on POSIX while working fine on
        Windows. Depending on those internals is what let a Linux-only failure
        through a green Windows test suite, so the sequence is spelled out here
        instead: close stdin so ffmpeg sees EOF and finalises the file, read its
        stderr to the end, then reap it.

        Draining stderr before ``wait()`` is the ordering that matters: waiting
        first would deadlock if ffmpeg had filled the stderr pipe. With stdout
        sent to /dev/null there is only this one pipe to drain, so a plain read
        is sufficient.
        """
        stdin = self._proc.stdin
        if stdin is not None and not stdin.closed:
            try:
                stdin.close()
            except OSError:
                pass  # ffmpeg already exited; nothing left to tell it.

        stderr = b""
        pipe = self._proc.stderr
        if pipe is not None and not pipe.closed:
            try:
                stderr = pipe.read() or b""
            except OSError:
                pass
            finally:
                pipe.close()

        self._proc.wait()
        return stderr

    def _exit_message(self, stderr: bytes | None = None) -> str:
        """ffmpeg's own explanation, which is far more useful than a pipe error."""
        if stderr is None:
            stderr = self._shutdown()
        detail = (stderr or b"").decode("utf-8", "replace").strip()
        return (
            f"ffmpeg exited with code {self._proc.returncode}"
            + (f": {detail[:400]}" if detail else " and gave no explanation")
        )


class _OpenCvWriter:
    """Fallback when ffmpeg is unavailable. Verifies it actually wrote something.

    OpenCV's H.264 writers on this platform report ``isOpened() == True`` and
    then produce a file of about a kilobyte, so success has to be judged from
    the output rather than from the API.
    """

    MIN_PLAUSIBLE_BYTES = 20_000

    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.path = path
        self.size = (width, height)
        self.codec = None
        self._writer = None
        for tag in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*tag), fps, (width, height)
            )
            if writer.isOpened():
                self._writer, self.codec = writer, tag
                break
            writer.release()
        if self._writer is None:
            raise RuntimeError("no usable video encoder available")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def close(self) -> OverlayResult:
        self._writer.release()
        size = self.path.stat().st_size if self.path.exists() else 0
        if size < self.MIN_PLAUSIBLE_BYTES:
            raise RuntimeError(
                f"the {self.codec} encoder reported success but wrote only "
                f"{size} bytes; no working video encoder is installed. Install "
                "ffmpeg (see packages.txt) to enable the annotated video."
            )
        playable = self.codec == "avc1"
        return OverlayResult(
            path=self.path, codec=self.codec, browser_playable=playable,
            n_frames=0,
            note=None if playable else (
                "Encoded as MPEG-4 Part 2 because ffmpeg is not installed. Most "
                "browsers cannot play this; download it to view, or install "
                "ffmpeg."
            ),
        )


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _open_writer(path: Path, fps: float, width: int, height: int, *, crf: int):
    if ffmpeg_available():
        return _FfmpegWriter(path, fps, width, height, crf)
    return _OpenCvWriter(path, fps, width, height)
