"""Gap handling: interpolate short dropouts, refuse long ones, never drop frames."""
import numpy as np

from gaitscreen.pose.schema import PL
from gaitscreen.pose.to_pixels import to_pixels
from gaitscreen.signal import resample
from fixtures.synthetic import occlude, synthetic_walk


def test_short_gaps_are_interpolated(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=8)
    max_gap = int(cfg["landmarks.max_interpolation_gap_frames"])
    occlude(raw, PL.LEFT_KNEE, 30, 30 + max_gap)

    series, summary = resample.fill_short_gaps(to_pixels(raw, cfg), cfg)
    assert summary["samples_interpolated"] == 2 * max_gap  # both axes
    assert np.isfinite(series.xy[30:30 + max_gap, int(PL.LEFT_KNEE), :]).all()
    # Interpolated samples stay flagged, so quality scoring can discount them.
    assert not series.valid[30:30 + max_gap, int(PL.LEFT_KNEE)].any()


def test_long_gaps_are_left_open(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=8)
    max_gap = int(cfg["landmarks.max_interpolation_gap_frames"])
    occlude(raw, PL.LEFT_KNEE, 40, 40 + max_gap + 5)

    series, summary = resample.fill_short_gaps(to_pixels(raw, cfg), cfg)
    assert summary["samples_interpolated"] == 0
    assert np.isnan(series.xy[40:40 + max_gap + 5, int(PL.LEFT_KNEE), 0]).all()


def test_frame_count_is_never_reduced(cfg):
    """Dropping frames would compress the time axis and shorten stride times."""
    raw, _ = synthetic_walk(fps=60.0, n_strides=8)
    occlude(raw, PL.RIGHT_ANKLE, 10, 40)
    n_before = raw.n_frames

    series, _ = resample.fill_short_gaps(to_pixels(raw, cfg), cfg)
    assert series.n_frames == n_before
    np.testing.assert_allclose(np.diff(series.t), 1.0 / 60.0, atol=1e-12)


def test_long_gap_splits_the_recording_into_segments(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=14)
    # Blank a gait-critical landmark for well over a stride, mid-recording.
    middle = raw.n_frames // 2
    occlude(raw, PL.RIGHT_HEEL, middle, middle + 120)

    series, _ = resample.fill_short_gaps(to_pixels(raw, cfg), cfg)
    segments = resample.analysis_segments(series, cfg)
    assert len(segments) == 2, f"expected the gap to split the walk, got {segments}"
    for start, stop in segments:
        assert not (start < middle < stop)


def test_segments_shorter_than_a_stride_are_discarded(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=14)
    # Leave only a very short usable island at the start.
    occlude(raw, PL.LEFT_ANKLE, 20, raw.n_frames)

    series, _ = resample.fill_short_gaps(to_pixels(raw, cfg), cfg)
    assert resample.analysis_segments(series, cfg) == []


def test_leading_and_trailing_gaps_are_not_extrapolated(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=8)
    occlude(raw, PL.LEFT_KNEE, 0, 3)
    occlude(raw, PL.LEFT_KNEE, raw.n_frames - 3, raw.n_frames)

    series, _ = resample.fill_short_gaps(to_pixels(raw, cfg), cfg)
    assert np.isnan(series.xy[0:3, int(PL.LEFT_KNEE), 0]).all()
    assert np.isnan(series.xy[-3:, int(PL.LEFT_KNEE), 0]).all()
