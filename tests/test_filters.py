"""Filter behaviour, and the specific reason One-Euro is not the default.

The claim in :mod:`gaitscreen.signal.filters` is that a causal filter shifts
event timing while ``filtfilt`` does not, and that One-Euro's velocity-adaptive
cutoff makes the shift depend on how fast the signal is moving -- which biases
stride timing. These tests hold that claim to account rather than taking it on
faith, because it is the justification for departing from the brief.
"""
import numpy as np

from gaitscreen.signal.filters import butterworth, one_euro


def _noisy_sine(frequency_hz: float, fps: float, seconds: float, noise: float, seed: int = 3):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 1.0 / fps)
    clean = np.sin(2 * np.pi * frequency_hz * t)
    return t, clean, clean + rng.normal(0.0, noise, t.size)


def test_butterworth_removes_noise(cfg):
    fps = 60.0
    _, clean, noisy = _noisy_sine(1.0, fps, 6.0, noise=0.20)
    filtered = butterworth(noisy, fps=fps, cutoff_hz=6.0, order=4)

    error_before = np.abs(noisy - clean).mean()
    error_after = np.abs(filtered - clean).mean()
    # A 6 Hz cutoff at 60 fps passes the lowest fifth of the spectrum, so the
    # floor for broadband noise is around sqrt(6/30) ~ 0.45 of the input error.
    assert error_after < 0.6 * error_before


def test_butterworth_preserves_peak_timing_but_one_euro_lags():
    """Zero-phase vs causal, measured as a shift in the detected peak."""
    fps = 60.0
    t, clean, noisy = _noisy_sine(1.0, fps, 6.0, noise=0.05)

    # One period only: a longer window contains several equal peaks, and argmax
    # between them is decided by noise rather than by phase.
    window = int(fps / 1.0)
    true_peak = int(np.argmax(clean[:window]))
    butter_peak = int(np.argmax(butterworth(noisy, fps=fps, cutoff_hz=6.0, order=4)[:window]))
    euro_peak = int(np.argmax(
        one_euro(noisy, fps=fps, min_cutoff_hz=1.0, beta=0.02, d_cutoff_hz=1.0)[:window]
    ))

    butter_shift = abs(butter_peak - true_peak)
    euro_shift = abs(euro_peak - true_peak)

    assert butter_shift <= 1, f"filtfilt should not shift peaks (shifted {butter_shift})"
    assert euro_shift > butter_shift, (
        "One-Euro is causal and should lag; if it does not, the comparison in "
        "filters.py needs revisiting"
    )


def test_one_euro_lag_depends_on_signal_speed():
    """The reason One-Euro biases variability: its lag is not constant.

    A filter whose lag varies with stride speed shifts fast and slow strides by
    different amounts, which changes the measured spread of stride times -- the
    metric this tool leans on hardest.
    """
    fps = 60.0
    lags = []
    for frequency in (0.7, 2.0):
        t, clean, _ = _noisy_sine(frequency, fps, 6.0, noise=0.0)
        filtered = one_euro(clean, fps=fps, min_cutoff_hz=1.0, beta=0.02, d_cutoff_hz=1.0)
        window = int(1.5 * fps / frequency)
        lags.append(int(np.argmax(filtered[:window])) - int(np.argmax(clean[:window])))

    assert lags[0] != lags[1], (
        f"expected speed-dependent lag, got identical shifts {lags}"
    )


def test_butterworth_falls_back_gracefully_on_short_segments():
    """Short runs cannot satisfy filtfilt's padding; the fallback must stay finite."""
    for length in (4, 6, 9, 15):
        values = np.linspace(0.0, 1.0, length) + 0.01
        out = butterworth(values, fps=25.0, cutoff_hz=6.0, order=4)
        assert out.shape == values.shape
        assert np.isfinite(out).all()


def test_cutoff_above_nyquist_is_a_no_op():
    """At low fps a 6 Hz cutoff can exceed Nyquist; that must not raise."""
    values = np.linspace(0.0, 1.0, 40)
    out = butterworth(values, fps=10.0, cutoff_hz=6.0, order=4)
    np.testing.assert_allclose(out, values)
