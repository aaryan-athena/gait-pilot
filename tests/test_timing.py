"""Timing integrity: the clock comes from the camera, not from the frame count.

Phones record at a variable rate. In dim light they skip frames and write
nothing where those frames would have been, and the container then reports an
*average* frame rate. Counting decoded frames at that average rate is right
over the whole clip and wrong at every point inside it: on the pilot metronome
clips it misplaced events by up to 1.8 s and mis-timed individual strides by
5-20%, several times the stride-time variability the tool exists to measure.

These tests pin the arithmetic that fixes it, without needing a video file.
"""
import numpy as np
import pytest

from gaitscreen.io.video import frame_grid, nominal_interval_ms
from gaitscreen.pose.landmarker import place_on_grid
from gaitscreen.pose.schema import N_LANDMARKS
from gaitscreen.types import VideoInfo


def _timestamps(fps, n, missing=()):
    """Presentation timestamps (ms) for an ``n``-frame grid minus ``missing``."""
    missing = set(missing)
    return np.array([i * 1000.0 / fps for i in range(n) if i not in missing])


def _scattered_drops(n, rate, seed=7):
    """Short runs of skipped frames, the pattern dim light produces."""
    rng = np.random.default_rng(seed)
    missing, i = set(), 1
    while i < n - 1:
        if rng.random() < rate:
            run = int(rng.integers(1, 6))
            missing.update(range(i, min(i + run, n - 1)))
            i += run
        i += 1
    return missing


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------
def test_a_clean_clip_maps_one_row_per_frame():
    grid = frame_grid(_timestamps(60.0, 300), container_fps=60.0)
    assert grid.rows.tolist() == list(range(300))
    assert grid.dropped == 0
    assert grid.fps == pytest.approx(60.0)


def test_skipped_frames_become_empty_rows():
    grid = frame_grid(_timestamps(60.0, 120, missing=[60, 61, 90]), container_fps=58.0)
    assert grid.n_rows == 120
    assert grid.dropped == 3
    assert grid.max_gap == 2
    assert sorted(set(range(120)) - set(grid.rows.tolist())) == [60, 61, 90]


def test_the_grid_rate_is_not_the_container_average():
    """The container reports the average rate; using it breaks the grid.

    On the metronome clip shot at 60 fps the container said 43.2. Rounding
    timestamps 16.7 ms apart onto a 23 ms grid would land consecutive frames on
    the same row, which is the failure the grid rate must be measured to avoid.
    """
    missing = _scattered_drops(4000, rate=0.3)
    ts = _timestamps(60.0, 4000, missing)
    container = ts.size / (ts[-1] / 1000.0)
    assert container < 50  # the average really is far from the grid rate

    grid = frame_grid(ts, container_fps=container)
    assert grid.fps == pytest.approx(60.0, rel=1e-6)
    assert grid.dropped == len(missing)


def test_59_94_is_not_rounded_to_60():
    """Treating 59.94 as 60 misplaces a frame every 16 seconds."""
    fps = 60000 / 1001
    grid = frame_grid(_timestamps(fps, 4000), container_fps=fps)
    assert grid.fps == pytest.approx(fps, rel=1e-6)
    assert grid.dropped == 0


def test_sub_frame_jitter_does_not_invent_drops():
    rng = np.random.default_rng(0)
    interval = 1000.0 / 60.0
    ts = np.arange(600) * interval
    ts[1:] += rng.uniform(-0.3 * interval, 0.3 * interval, 599)
    grid = frame_grid(np.maximum.accumulate(ts), container_fps=60.0)
    assert grid.dropped == 0


def test_unusable_timestamps_fall_back_to_the_old_assumption():
    """A backend reporting zeros must not produce a grid of collisions.

    The fallback is what the pipeline did before this fix, so it can only
    under-report drops, never invent them.
    """
    grid = frame_grid(np.zeros(200), container_fps=30.0)
    assert not grid.from_timestamps
    assert grid.rows.tolist() == list(range(200))
    assert grid.fps == 30.0


def test_the_rebuilt_clock_matches_the_camera():
    missing = _scattered_drops(3000, rate=0.25, seed=3)
    ts = _timestamps(60.0, 3000, missing)
    grid = frame_grid(ts, container_fps=float(ts.size / (ts[-1] / 1000.0)))

    rebuilt = grid.rows / grid.fps * 1000.0
    assert np.max(np.abs(rebuilt - ts)) < 1e-6

    # ...where counting frames at the average rate, as before, drifts badly.
    naive = np.arange(ts.size) * (ts[-1] / (ts.size - 1))
    assert np.max(np.abs(naive - ts)) > 100


def test_nominal_interval_needs_enough_frames():
    assert nominal_interval_ms(np.arange(5) * 16.7) is None


# --------------------------------------------------------------------------
# placing landmarks on it
# --------------------------------------------------------------------------
def _decoded(n):
    xy = np.random.default_rng(1).uniform(0.2, 0.8, size=(n, N_LANDMARKS, 2))
    return (xy, np.zeros((n, N_LANDMARKS)),
            np.full((n, N_LANDMARKS), 0.9), np.ones(n, dtype=bool))


def test_landmarks_land_on_their_true_rows(cfg):
    ts = _timestamps(60.0, 200, missing=[50, 51, 52, 120])
    xy, z, vis, det = _decoded(ts.size)
    info = VideoInfo(path="x.mp4", width=1920, height=1080, fps=58.9,
                     n_frames=ts.size)

    raw = place_on_grid(info, cfg, ts, xy, z, vis, det)

    assert raw.n_frames == 200
    assert info.fps == pytest.approx(60.0)
    assert info.container_fps == pytest.approx(58.9)
    assert info.dropped_frames == 4
    assert np.isnan(raw.xy[50:53]).all() and np.isnan(raw.xy[120]).all()
    np.testing.assert_allclose(raw.xy[53], xy[50])  # first frame after the gap
    np.testing.assert_allclose(raw.t, np.arange(200) / 60.0)


def test_skipped_frames_do_not_count_against_the_tracker(cfg):
    """Nothing was there to detect; the lighting is not the tracker's fault."""
    missing = _scattered_drops(600, rate=0.3)
    ts = _timestamps(60.0, 600, missing)
    xy, z, vis, det = _decoded(ts.size)
    info = VideoInfo(path="x.mp4", width=1920, height=1080, fps=45.0,
                     n_frames=ts.size)

    raw = place_on_grid(info, cfg, ts, xy, z, vis, det)
    assert raw.detection_rate == pytest.approx(1.0)
    assert raw.frame_rows.size == ts.size


def test_a_heavily_dropped_clip_says_so(cfg):
    missing = _scattered_drops(900, rate=0.3)
    ts = _timestamps(60.0, 900, missing)
    xy, z, vis, det = _decoded(ts.size)
    info = VideoInfo(path="x.mp4", width=1920, height=1080, fps=45.0,
                     n_frames=ts.size)

    place_on_grid(info, cfg, ts, xy, z, vis, det)
    assert any("skipped" in w for w in info.warnings)


def test_frame_rate_warnings_describe_the_real_rate(cfg):
    """The average rate triggered a "below 50 fps" warning at probe time.

    A 60 fps camera that skipped frames did not record at 43 fps, and telling
    the operator to switch to 60 fps would send them to fix a setting that is
    already right. The warning about the skipped frames replaces it.
    """
    from gaitscreen.io.video import fps_warnings

    missing = _scattered_drops(900, rate=0.3)
    ts = _timestamps(60.0, 900, missing)
    xy, z, vis, det = _decoded(ts.size)
    info = VideoInfo(path="x.mp4", width=1920, height=1080, fps=43.2,
                     n_frames=ts.size, warnings=list(fps_warnings(43.2, cfg)))
    assert info.warnings  # the stale warning is really there to begin with

    place_on_grid(info, cfg, ts, xy, z, vis, det)
    assert not any("43.2" in w for w in info.warnings)


# --------------------------------------------------------------------------
# end to end: a walk with known strides, filmed by a camera that skips frames
# --------------------------------------------------------------------------
def _through_pipeline(cfg, raw):
    from gaitscreen.features.session import analyse
    from gaitscreen.pipeline import extraction_from_raw

    return analyse(extraction_from_raw(raw, cfg), cfg)


def _film_with_skips(raw, rate, seed=5):
    """What the phone writes: the frames it kept, and their timestamps."""
    missing = _scattered_drops(raw.n_frames, rate=rate, seed=seed)
    keep = np.array([i for i in range(raw.n_frames) if i not in missing])
    stamps = raw.t[keep] * 1000.0
    kept = (raw.xy[keep], raw.z[keep], raw.visibility[keep], raw.detected[keep])
    return stamps, kept


def _cv_error(cfg, analysis, truth):
    return abs(analysis.metrics.stride_time_cv_pct - truth.stride_time_cv_pct)


def test_skipped_frames_no_longer_distort_stride_times(cfg):
    """The pilot metronome clip, reproduced with ground truth.

    Counting the frames a camera kept as if they were evenly spaced gave a
    stride-time CV of 11.8% on a metronome-paced walk -- above the 5% line
    this tool treats as high fall risk -- where the rebuilt clock gave 2.1%.
    Here the true CV is prescribed, so the comparison is exact.
    """
    from fixtures.synthetic import synthetic_walk

    raw, truth = synthetic_walk(fps=60.0, n_strides=24, stride_time_cv=0.02)
    stamps, (xy, z, vis, det) = _film_with_skips(raw, rate=0.25)
    container_fps = stamps.size / (stamps[-1] / 1000.0)

    rebuilt = place_on_grid(
        VideoInfo(path="x.mp4", width=raw.video.width, height=raw.video.height,
                  fps=container_fps, n_frames=stamps.size),
        cfg, stamps, xy, z, vis, det,
    )
    naive = type(raw)(
        t=np.arange(stamps.size) / container_fps, xy=xy, z=z, visibility=vis,
        detected=det,
        video=VideoInfo(path="x.mp4", width=raw.video.width,
                        height=raw.video.height, fps=container_fps,
                        n_frames=stamps.size),
    )

    fixed = _through_pipeline(cfg, rebuilt)
    broken = _through_pipeline(cfg, naive)

    assert fixed.metrics.stride_time_cv_pct is not None
    assert _cv_error(cfg, fixed, truth) < 0.8
    # The failure being fixed, so this test cannot pass vacuously.
    assert broken.metrics.stride_time_cv_pct > truth.stride_time_cv_pct + 2.0

    expected = float(np.mean(truth.all_stride_times))
    assert abs(fixed.metrics.stride_time_mean_s - expected) < 0.01


def test_skipped_frames_do_not_cost_strides(cfg):
    """A stride is not rejected just because the camera skipped a frame at it.

    The per-event confidence gate exists for the tracker losing a foot, when
    the landmark it still reports is a guess. A skipped frame has genuine
    detections either side. On the pilot metronome clip the strides rejected
    for skipped frames matched the metronome exactly as well as the rest, and
    rejecting them cost a quarter of the usable strides.
    """
    from fixtures.synthetic import synthetic_walk

    raw, _ = synthetic_walk(fps=60.0, n_strides=24, stride_time_cv=0.0)
    clean = _through_pipeline(cfg, raw)

    stamps, (xy, z, vis, det) = _film_with_skips(raw, rate=0.25)
    rebuilt = place_on_grid(
        VideoInfo(path="x.mp4", width=raw.video.width, height=raw.video.height,
                  fps=45.0, n_frames=stamps.size),
        cfg, stamps, xy, z, vis, det,
    )
    skipped = _through_pipeline(cfg, rebuilt)

    assert skipped.metrics.n_strides_valid >= clean.metrics.n_strides_valid - 1
