"""Gap handling on a uniform time base.

Two rules drive this module:

1. **Never drop frames.** Both the zero-phase filter and the peak detector
   assume evenly spaced samples. Removing low-visibility frames would compress
   the time axis and shorten every stride time it touched -- a systematic bias
   in exactly the metric the tool cares most about.
2. **Never interpolate across a long gap.** A short dropout (a few frames of
   the far leg passing behind the near leg) interpolates cleanly. A long one
   means the limb was genuinely unobserved, and inventing a trajectory there
   would invent gait events with it. Long gaps split the recording into separate
   analysis segments instead.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..pose.schema import GAIT_CRITICAL
from ..types import PixelSeries


def fill_short_gaps(series: PixelSeries, cfg: Config) -> tuple[PixelSeries, dict[str, int]]:
    """Linearly interpolate gaps up to ``landmarks.max_interpolation_gap_frames``.

    Returns the filled series and a summary of what was filled. Interpolated
    samples stay ``False`` in ``valid`` so quality scoring and event confidence
    can discount them.
    """
    max_gap = int(cfg["landmarks.max_interpolation_gap_frames"])
    xy = series.xy.copy()
    n_frames, n_landmarks, _ = xy.shape
    filled = 0
    left_open = 0

    for lm in range(n_landmarks):
        for axis in range(2):
            column = xy[:, lm, axis]
            good = np.isfinite(column)
            if good.all() or not good.any():
                continue
            for start, stop in _gap_runs(good):
                # Leading and trailing gaps have nothing to interpolate between.
                if start == 0 or stop == n_frames:
                    left_open += stop - start
                    continue
                if stop - start > max_gap:
                    left_open += stop - start
                    continue
                span = stop - start + 1
                column[start:stop] = np.interp(
                    np.arange(start, stop),
                    [start - 1, stop],
                    [column[start - 1], column[stop]],
                )
                filled += stop - start
                del span
            xy[:, lm, axis] = column

    out = PixelSeries(
        t=series.t, xy=xy, visibility=series.visibility,
        valid=series.valid.copy(), video=series.video,
    )
    return out, {"samples_interpolated": filled, "samples_left_missing": left_open}


def analysis_segments(
    series: PixelSeries, cfg: Config, *, min_frames: int | None = None
) -> list[tuple[int, int]]:
    """Frame ranges where all gait-critical landmarks are present.

    Segments shorter than ``min_frames`` (default: one maximum stride) are
    discarded -- they cannot contain a complete gait cycle.
    """
    if min_frames is None:
        min_stride = float(cfg["segmentation.max_stride_time_s"])
        min_frames = max(4, int(round(min_stride * series.fps)))

    usable = np.isfinite(series.xy[:, list(GAIT_CRITICAL), :]).all(axis=(1, 2))
    return [
        (start, stop)
        for start, stop in _true_runs(usable)
        if stop - start >= min_frames
    ]


def _gap_runs(good: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ranges of consecutive False values in ``good``."""
    return _true_runs(~good)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ranges of consecutive True values."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))
