"""Peak/extremum detection with sub-frame timing.

Sub-frame refinement is not a nicety here, it is what makes the headline metric
possible at all. Stride-time variability in elderly gait is a standard deviation
of roughly 25-35 ms. Taking the integer frame index of an extremum quantises
every event to the frame interval -- 40 ms at 25 fps, 33 ms at 30 fps -- which
is the same size as the quantity being measured, so the measured variability
would be dominated by the sampling grid rather than by the walker.

Fitting a parabola through the extremum and its two neighbours recovers the
underlying continuous extremum to a fraction of a frame. It is exact for a
locally quadratic signal, which a smoothed landmark trajectory is near its
turning points.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def find_extrema(
    values: np.ndarray,
    *,
    fps: float,
    kind: str = "max",
    min_separation_s: float = 0.0,
    prominence_frac: float = 0.0,
) -> np.ndarray:
    """Locate local maxima (or minima) with separation and prominence gating.

    ``prominence_frac`` is expressed as a fraction of the signal's peak-to-peak
    range, so the same configured value works across subjects and camera
    distances without rescaling.
    """
    values = np.asarray(values, dtype=float)
    signal = values if kind == "max" else -values

    finite = signal[np.isfinite(signal)]
    if finite.size < 3:
        return np.empty(0, dtype=int)

    span = float(np.ptp(finite))
    prominence = prominence_frac * span if prominence_frac > 0 and span > 0 else None
    distance = max(1, int(round(min_separation_s * fps))) if min_separation_s > 0 else None

    filled = np.where(np.isfinite(signal), signal, -np.inf)
    peaks, _ = find_peaks(filled, distance=distance, prominence=prominence)
    return peaks


def refine_extremum(values: np.ndarray, index: int) -> float:
    """Sub-frame position of an extremum, as a fractional index.

    Returns ``index + delta`` with ``delta`` in (-0.5, 0.5) from a three-point
    parabolic fit. Falls back to the integer index at array edges or where the
    curvature is degenerate.
    """
    values = np.asarray(values, dtype=float)
    if index <= 0 or index >= values.size - 1:
        return float(index)

    y_prev, y_mid, y_next = values[index - 1], values[index], values[index + 1]
    if not np.isfinite([y_prev, y_mid, y_next]).all():
        return float(index)

    denominator = y_prev - 2.0 * y_mid + y_next
    if abs(denominator) < 1e-12:
        return float(index)

    delta = 0.5 * (y_prev - y_next) / denominator
    if not np.isfinite(delta) or abs(delta) > 0.5:
        return float(index)
    return float(index) + float(delta)


def refine_zero_crossing(values: np.ndarray, index: int) -> float:
    """Sub-frame position of a sign change between ``index`` and ``index + 1``."""
    values = np.asarray(values, dtype=float)
    if index < 0 or index >= values.size - 1:
        return float(index)
    a, b = values[index], values[index + 1]
    if not np.isfinite([a, b]).all() or a == b or np.sign(a) == np.sign(b):
        return float(index)
    return float(index) + float(a / (a - b))


def fractional_time(t: np.ndarray, fractional_index: float) -> float:
    """Convert a fractional frame index to a timestamp on a uniform time base."""
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return float("nan")
    low = int(np.floor(fractional_index))
    if low < 0:
        return float(t[0])
    if low >= t.size - 1:
        return float(t[-1])
    frac = fractional_index - low
    return float(t[low] + frac * (t[low + 1] - t[low]))


def resample_to_percent(values: np.ndarray, n_points: int = 101) -> np.ndarray:
    """Resample one cycle onto 0-100% of the cycle, for cross-cycle comparison."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.full(n_points, np.nan)
    source = np.linspace(0.0, 100.0, values.size)
    target = np.linspace(0.0, 100.0, n_points)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full(n_points, np.nan)
    return np.interp(target, source[finite], values[finite])
