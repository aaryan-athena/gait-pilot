"""Temporal smoothing of landmark trajectories.

Choice of default, and the trade-off the brief asked about:

**Butterworth via ``filtfilt`` (default).** Zero-phase: the filter runs forward
then backward, so group delay cancels exactly and event *timing* is preserved.
Fourth order at 6 Hz is the long-standing convention in gait biomechanics --
normal walking harmonics sit below about 6 Hz, so this removes landmark jitter
without touching the kinematics. It is non-causal, which is fine here because
the brief specifies batch processing of recorded files.

**One-Euro (available, not default).** Designed for interactive/real-time input:
it trades accuracy for latency by adapting its cutoff to the observed speed of
the signal. Two consequences make it a poor fit for this tool. It is causal, so
it lags, shifting detected heel strikes later by an amount that depends on
signal speed. And because the cutoff *varies with velocity*, fast and slow
strides receive different effective smoothing -- which biases stride-time
variability, the single metric the tool most depends on, in a direction that
cannot be detected from the output. Kept behind the same interface for a future
live-camera path.

**Kalman (available, not default).** A constant-velocity Kalman filter handles
missing observations natively, which is genuinely attractive for occluded limbs.
But the forward-only pass is again causal and lags; getting zero-phase behaviour
requires an RTS smoother, at which point it offers little over ``filtfilt``
while adding two tuning parameters. Explicit gap interpolation plus ``filtfilt``
gets the same benefit more transparently.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from ..config import Config
from ..types import PixelSeries

__all__ = ["smooth_series", "butterworth", "one_euro", "kalman_constant_velocity"]


def smooth_series(series: PixelSeries, cfg: Config) -> PixelSeries:
    """Apply the configured filter to every landmark trajectory.

    Filtering is applied per contiguous run of finite samples, so it never
    smears data across a gap that :mod:`gaitscreen.signal.resample` deliberately
    left open.
    """
    kind = str(cfg["filter.kind"]).lower()
    fps = series.fps
    xy = series.xy.copy()

    if kind == "butterworth":
        cutoff = float(cfg["filter.butterworth.cutoff_hz"])
        order = int(cfg["filter.butterworth.order"])
        fn = lambda v: butterworth(v, fps=fps, cutoff_hz=cutoff, order=order)  # noqa: E731
    elif kind == "one_euro":
        params = cfg.section("filter.one_euro")
        fn = lambda v: one_euro(  # noqa: E731
            v, fps=fps,
            min_cutoff_hz=float(params["min_cutoff_hz"]),
            beta=float(params["beta"]),
            d_cutoff_hz=float(params["d_cutoff_hz"]),
        )
    elif kind == "kalman":
        params = cfg.section("filter.kalman")
        fn = lambda v: kalman_constant_velocity(  # noqa: E731
            v, fps=fps,
            process_var=float(params["process_var"]),
            measurement_var=float(params["measurement_var"]),
        )
    else:
        raise ValueError(f"unknown filter kind: {kind!r}")

    from .resample import _true_runs  # local import: shared run-length helper

    for lm in range(xy.shape[1]):
        for axis in range(2):
            column = xy[:, lm, axis]
            for start, stop in _true_runs(np.isfinite(column)):
                if stop - start >= 4:
                    column[start:stop] = fn(column[start:stop])
            xy[:, lm, axis] = column

    return PixelSeries(
        t=series.t, xy=xy, visibility=series.visibility,
        valid=series.valid, video=series.video,
    )


def butterworth(values: np.ndarray, *, fps: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase low-pass filter.

    Short runs cannot satisfy ``filtfilt``'s padding requirement; those fall back
    to a Savitzky-Golay filter, which is also symmetric (zero-phase) and so does
    not introduce a timing bias that would differ between segments.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    nyquist = 0.5 * fps
    wn = cutoff_hz / nyquist
    if not 0 < wn < 1:
        # Cutoff at or above Nyquist: nothing meaningful to remove.
        return values.copy()

    b, a = butter(order, wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    if n > padlen:
        return filtfilt(b, a, values, padlen=padlen)

    window = min(n if n % 2 else n - 1, 7)
    if window >= 5:
        return savgol_filter(values, window_length=window, polyorder=2)
    return values.copy()


def one_euro(
    values: np.ndarray, *, fps: float, min_cutoff_hz: float, beta: float,
    d_cutoff_hz: float,
) -> np.ndarray:
    """Causal One-Euro filter (Casiez et al. 2012). Provided for a real-time path."""
    values = np.asarray(values, dtype=float)
    dt = 1.0 / fps
    out = np.empty_like(values)

    def alpha(cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    x_prev = values[0]
    dx_prev = 0.0
    out[0] = x_prev
    a_d = alpha(d_cutoff_hz)
    for i in range(1, values.size):
        dx = (values[i] - x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff_hz + beta * abs(dx_hat)
        a = alpha(cutoff)
        x_hat = a * values[i] + (1 - a) * x_prev
        out[i] = x_hat
        x_prev, dx_prev = x_hat, dx_hat
    return out


def kalman_constant_velocity(
    values: np.ndarray, *, fps: float, process_var: float, measurement_var: float
) -> np.ndarray:
    """Forward-pass constant-velocity Kalman filter. Causal; lags by design."""
    values = np.asarray(values, dtype=float)
    dt = 1.0 / fps
    scale = np.nanstd(values) or 1.0

    A = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * process_var * scale**2
    R = np.array([[measurement_var * scale**2]])

    x = np.array([[values[0]], [0.0]])
    P = np.eye(2) * scale**2
    out = np.empty_like(values)
    out[0] = values[0]

    for i in range(1, values.size):
        x = A @ x
        P = A @ P @ A.T + Q
        residual = values[i] - (H @ x)[0, 0]
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T) / S
        x = x + K * residual
        P = (np.eye(2) - K @ H) @ P
        out[i] = x[0, 0]
    return out
