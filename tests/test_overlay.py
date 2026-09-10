"""The annotated playback video.

The end-to-end tests render a real file and inspect it with ffprobe, because the
failure this code exists to guard against is one that looks like success: at some
frame sizes OpenCV's H.264 writer reports ``isOpened() == True``, fails to load
its encoder, and writes about a kilobyte of nothing. Only the output tells the
truth, so the output is what gets checked.
"""
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from gaitscreen.features.session import analyse
from gaitscreen.reporting import overlay as overlay_module
from gaitscreen.reporting.overlay import (COLOUR, EVENT_STRIKE, EVENT_TOEOFF,
                                          LEGEND, SKELETON,
                                          render_overlay_video)
from fixtures.extraction import build_extraction

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg is required to encode browser-playable video",
)


def _synthetic_clip(cfg, tmp_path, **kwargs):
    """A real video file whose frames line up with synthetic landmarks.

    The overlay reads the source video back off disk, so it needs an actual
    file with the same frame count and dimensions as the landmark series.
    """
    params = dict(fps=30.0, n_strides=4, leg_length_px=200, width=640,
                  height=360, in_place=True)
    params.update(kwargs)
    extraction, _ = build_extraction(cfg, **params)

    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), params["fps"],
        (params["width"], params["height"]),
    )
    rng = np.random.default_rng(0)
    for _ in range(extraction.series.n_frames):
        # Textured noise rather than a flat colour, so the drawn overlay has to
        # actually stand out against something.
        writer.write(rng.integers(60, 200, (params["height"], params["width"], 3),
                                  dtype=np.uint8))
    writer.release()

    extraction.info.path = path
    return extraction, analyse(extraction, cfg)


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,codec_name", "-of",
         "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return {"codec": out[0], "width": int(out[1]), "height": int(out[2]),
            "frames": int(out[3])}


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------
def test_skeleton_covers_both_legs_and_the_trunk():
    groups = {group for _, _, group in SKELETON}
    assert {"trunk", "left_leg", "right_leg"} <= groups
    for _, _, group in SKELETON:
        assert group in COLOUR, f"{group} has no colour"


def test_left_and_right_legs_are_visually_distinct():
    """They carry the left/right reading, so they must not be confusable."""
    left, right = COLOUR["left_leg"], COLOUR["right_leg"]
    distance = sum(abs(a - b) for a, b in zip(left, right))
    assert distance > 150, f"leg colours are too close: {left} vs {right}"


def test_leg_colours_do_not_collide_with_the_event_markers():
    """An earlier version drew the left leg in the same blue as 'toe off'."""
    for leg in ("left_leg", "right_leg"):
        for event in (EVENT_STRIKE, EVENT_TOEOFF):
            distance = sum(abs(a - b) for a, b in zip(COLOUR[leg], event))
            assert distance > 120, f"{leg} is too close to an event colour"


def test_legend_explains_every_drawn_meaning():
    labels = {label for label, _ in LEGEND}
    assert {"left leg", "right leg", "heel down", "toe off"} == labels


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
@needs_ffmpeg
def test_renders_a_browser_playable_video(cfg, tmp_path):
    extraction, analysis = _synthetic_clip(cfg, tmp_path)
    out = tmp_path / "annotated.mp4"

    result = render_overlay_video(extraction, analysis, out, cfg)

    assert result.browser_playable
    assert result.codec == "h264"
    assert result.n_frames == extraction.series.n_frames
    probed = _probe(out)
    assert probed["codec"] == "h264", "browsers will not play anything else here"
    assert probed["frames"] == extraction.series.n_frames


@needs_ffmpeg
def test_output_is_downscaled_to_even_dimensions(cfg, tmp_path):
    """libx264 rejects odd dimensions, and 720p is plenty for playback."""
    extraction, analysis = _synthetic_clip(cfg, tmp_path, width=1281, height=723)
    out = tmp_path / "annotated.mp4"

    render_overlay_video(extraction, analysis, out, cfg, max_width=720)
    probed = _probe(out)
    assert probed["width"] % 2 == 0 and probed["height"] % 2 == 0
    assert probed["width"] <= 720


@needs_ffmpeg
def test_overlay_actually_draws_on_the_frames(cfg, tmp_path):
    """Guards against a silently empty render -- the failure that looks like success."""
    extraction, analysis = _synthetic_clip(cfg, tmp_path)
    out = tmp_path / "annotated.mp4"
    render_overlay_video(extraction, analysis, out, cfg)

    capture = cv2.VideoCapture(str(out))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok

    # The banner strip along the top and the legend along the bottom are drawn
    # on every frame, so both edges must be near-black.
    assert frame[4, :, :].mean() < 60, "no caption banner was drawn"
    assert frame[-6, :, :].mean() < 90, "no legend strip was drawn"


@needs_ffmpeg
def test_video_is_small_enough_to_stream(cfg, tmp_path):
    extraction, analysis = _synthetic_clip(cfg, tmp_path, n_strides=6)
    out = tmp_path / "annotated.mp4"
    render_overlay_video(extraction, analysis, out, cfg)

    per_second = out.stat().st_size / max(extraction.info.duration_s, 1e-6)
    assert per_second < 2_000_000, (
        f"{per_second/1e6:.1f} MB/s is too heavy for the app to serve inline"
    )


@needs_ffmpeg
def test_progress_is_reported(cfg, tmp_path):
    extraction, analysis = _synthetic_clip(cfg, tmp_path)
    seen = []
    render_overlay_video(extraction, analysis, tmp_path / "a.mp4", cfg,
                         progress=seen.append)
    assert seen and max(seen) <= extraction.series.n_frames


# --------------------------------------------------------------------------
# encoder fallback
# --------------------------------------------------------------------------
def test_fallback_refuses_to_call_an_empty_file_a_success(cfg, tmp_path,
                                                          monkeypatch):
    """The guard that catches OpenCV succeeding in name only.

    ``isOpened()`` cannot be trusted -- at some frame sizes this platform's
    OpenCV opens an H.264 writer, fails to load its encoder, and writes about a
    kilobyte of nothing. The only reliable signal is the size of the file that
    came out, so that check is tested directly rather than by hoping the
    platform misbehaves on cue.
    """
    monkeypatch.setattr(overlay_module, "ffmpeg_available", lambda: False)
    monkeypatch.setattr(overlay_module._OpenCvWriter, "MIN_PLAUSIBLE_BYTES",
                        10 ** 12)
    extraction, analysis = _synthetic_clip(cfg, tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        render_overlay_video(extraction, analysis, tmp_path / "b.mp4", cfg)

    message = str(excinfo.value)
    assert "reported success" in message
    assert "ffmpeg" in message, "the error must say how to fix it"


def test_fallback_reports_unplayable_output_rather_than_hiding_it(cfg, tmp_path,
                                                                 monkeypatch):
    """An mp4v file is real but most browsers will not play it.

    Returning it while claiming it is playable would give the viewer a dead
    player and no explanation.
    """
    monkeypatch.setattr(overlay_module, "ffmpeg_available", lambda: False)
    # Force the mp4v branch by making the H.264 attempt refuse to open.
    real_writer = cv2.VideoWriter
    fourcc_mp4v = cv2.VideoWriter_fourcc(*"mp4v")

    def only_mp4v(path, fourcc, fps, size):
        if fourcc != fourcc_mp4v:
            class Closed:
                def isOpened(self): return False
                def release(self): pass
            return Closed()
        return real_writer(path, fourcc, fps, size)

    monkeypatch.setattr(cv2, "VideoWriter", only_mp4v)
    extraction, analysis = _synthetic_clip(cfg, tmp_path)

    result = render_overlay_video(extraction, analysis, tmp_path / "c.mp4", cfg)
    assert result.codec == "mp4v"
    assert not result.browser_playable
    assert result.note and "browsers cannot play" in result.note
