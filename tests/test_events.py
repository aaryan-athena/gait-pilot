"""Sub-frame event timing.

This is the test that matters most for the headline metric. At 25 fps the frame
interval is 40 ms and elderly stride-time SD is roughly 25-35 ms, so integer
frame indices quantise the entire signal away. These tests quantify how much
sub-frame refinement buys back.
"""
import numpy as np

from gaitscreen.signal.events import (find_extrema, fractional_time,
                                      refine_extremum, refine_zero_crossing,
                                      resample_to_percent)


def _cosine_with_peak_at(peak_t: float, fps: float, duration: float, period: float):
    t = np.arange(0.0, duration, 1.0 / fps)
    return t, np.cos(2 * np.pi * (t - peak_t) / period)


def test_refinement_recovers_an_off_grid_peak():
    """A peak deliberately placed between two samples must still be located."""
    fps, period = 25.0, 1.1
    peak_t = 0.5 * period + 0.5 / fps  # exactly half a frame off the sample grid
    t, values = _cosine_with_peak_at(peak_t, fps, 4.0, period)

    index = int(np.argmax(values[: int(period * fps)]))
    integer_error = abs(t[index] - peak_t)
    refined_error = abs(fractional_time(t, refine_extremum(values, index)) - peak_t)

    assert refined_error < 0.35 * integer_error, (
        f"refinement should beat the frame grid: {refined_error * 1000:.1f} ms vs "
        f"{integer_error * 1000:.1f} ms"
    )


def test_refinement_error_stays_below_the_stride_time_sd_at_25fps():
    """Guards the claim that 25 fps is workable *with* refinement, not without.

    Elderly stride-time SD is around 25-35 ms. Refined event timing must land
    well inside that, or the measured variability is just the sampling grid.
    """
    fps, period = 25.0, 1.1
    errors_integer, errors_refined = [], []

    for offset in np.linspace(0.0, 1.0, 11):
        peak_t = 0.5 * period + offset / fps
        t, values = _cosine_with_peak_at(peak_t, fps, 4.0, period)
        index = int(np.argmax(values[: int(period * fps)]))
        errors_integer.append(abs(t[index] - peak_t))
        errors_refined.append(
            abs(fractional_time(t, refine_extremum(values, index)) - peak_t)
        )

    worst_integer = max(errors_integer) * 1000
    worst_refined = max(errors_refined) * 1000
    assert worst_integer > 15.0, "sanity: integer indexing should be badly quantised"
    assert worst_refined < 8.0, (
        f"refined timing error {worst_refined:.1f} ms is too large a share of a "
        "25-35 ms stride-time SD"
    )


def test_find_extrema_respects_minimum_separation():
    fps = 60.0
    t = np.arange(0.0, 8.0, 1.0 / fps)
    # Two superimposed rhythms; the fast one must not be counted as strides.
    values = np.cos(2 * np.pi * t / 1.1) + 0.15 * np.cos(2 * np.pi * t / 0.2)

    unfiltered = find_extrema(values, fps=fps, kind="max")
    strides = find_extrema(
        values, fps=fps, kind="max", min_separation_s=0.6, prominence_frac=0.15
    )
    assert len(strides) < len(unfiltered)
    assert 6 <= len(strides) <= 8, f"expected ~7 cycles in 8s, got {len(strides)}"


def test_find_minima():
    fps = 60.0
    t = np.arange(0.0, 4.0, 1.0 / fps)
    values = np.cos(2 * np.pi * t / 1.0)
    minima = find_extrema(values, fps=fps, kind="min", min_separation_s=0.5)
    assert len(minima) >= 3
    np.testing.assert_allclose(t[minima[0]], 0.5, atol=1.0 / fps)


def test_zero_crossing_refinement():
    values = np.array([-2.0, -1.0, 1.0, 2.0])
    # Crossing sits exactly midway between indices 1 and 2.
    assert np.isclose(refine_zero_crossing(values, 1), 1.5)
    # No sign change -> unchanged index.
    assert refine_zero_crossing(values, 2) == 2.0


def test_resample_to_percent_normalises_cycle_length():
    short = np.linspace(0.0, 10.0, 17)
    long = np.linspace(0.0, 10.0, 44)
    a = resample_to_percent(short, 101)
    b = resample_to_percent(long, 101)
    assert a.shape == b.shape == (101,)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_extrema_on_degenerate_input_do_not_raise():
    assert find_extrema(np.array([1.0]), fps=25.0).size == 0
    assert find_extrema(np.full(10, np.nan), fps=25.0).size == 0
    assert refine_extremum(np.array([1.0, 2.0, 3.0]), 0) == 0.0
    assert refine_extremum(np.full(5, 2.0), 2) == 2.0
