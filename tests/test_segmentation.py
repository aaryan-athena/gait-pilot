"""Segmentation against prescribed ground truth.

These are the tests that decide whether the tool measures gait or invents it.
The synthetic fixture is built from *known* heel-strike times, so recovered
event timing, stride times and stride-time variability can be compared against
the values that generated them.
"""
import numpy as np
import pytest

from gaitscreen.pose.to_pixels import to_pixels
from gaitscreen.segmentation import cycles as cycles_module
from gaitscreen.segmentation import direction as direction_module
from gaitscreen.segmentation.events import (detect_events,
                                            estimate_stride_period,
                                            independent_cadence_spm)
from gaitscreen.signal import filters, resample
from fixtures.synthetic import synthetic_walk


def _prepare(cfg, **kwargs):
    raw, truth = synthetic_walk(**kwargs)
    series = to_pixels(raw, cfg)
    series, _ = resample.fill_short_gaps(series, cfg)
    series = filters.smooth_series(series, cfg)
    segments = resample.analysis_segments(series, cfg)
    return series, truth, segments


# --------------------------------------------------------------------------
# direction
# --------------------------------------------------------------------------
def test_direction_from_body_axis_not_hip_velocity(cfg):
    """Must work on treadmill footage, where hip velocity carries no signal."""
    series, _, _ = _prepare(cfg, in_place=True, n_strides=10)
    sign, method, agreement = direction_module.anterior_sign(series)

    assert sign == 1
    assert method == "foot_axis", "should use a body-fixed cue, not hip motion"
    assert agreement > 0.9


def test_direction_detected_for_overground_walk(cfg):
    series, _, _ = _prepare(cfg, n_strides=10, speed_px_s=180.0)
    sign, _, _ = direction_module.anterior_sign(series)
    assert sign == 1


def test_in_place_walk_yields_one_pass(cfg):
    series, _, segments = _prepare(cfg, in_place=True, n_strides=10)
    passes = direction_module.find_passes(series, cfg, segments)
    assert len(passes) == 1
    assert passes[0].direction == 1


# --------------------------------------------------------------------------
# period estimation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stride_time", [0.85, 1.10, 1.60])
def test_stride_period_recovered_across_walking_speeds(cfg, stride_time):
    """A fixed minimum separation cannot serve both fast and slow walkers."""
    series, _, _ = _prepare(cfg, n_strides=12, stride_time_s=stride_time,
                            stride_time_cv=0.0)
    from gaitscreen.segmentation.events import _anterior

    signal = _anterior(series, "left", "heel", 1)
    period = estimate_stride_period(signal, series.fps, cfg)

    assert period is not None
    assert abs(period - stride_time) < 0.1 * stride_time


def test_period_estimation_survives_a_drifting_signal(cfg):
    """A panning camera adds drift whose autocorrelation swamps the gait rhythm."""
    series, _, _ = _prepare(cfg, n_strides=12, stride_time_s=1.1, stride_time_cv=0.0)
    from gaitscreen.segmentation.events import _anterior

    signal = _anterior(series, "left", "heel", 1)
    drift = np.linspace(0.0, 6.0 * np.ptp(signal), signal.size)
    period = estimate_stride_period(signal + drift, series.fps, cfg)

    assert period is not None
    assert abs(period - 1.1) < 0.15


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
def test_heel_strikes_recovered_at_prescribed_times(cfg):
    series, truth, segments = _prepare(cfg, fps=60.0, n_strides=12,
                                       stride_time_cv=0.03)
    passes = direction_module.find_passes(series, cfg, segments)
    events = detect_events(series, passes[0], cfg)

    for side in ("left", "right"):
        detected = sorted(e.t for e in events.of("heel_strike", side))
        expected = [t for t in truth.heel_strikes_s[side] if 0 < t < series.t[-1]]
        assert len(detected) >= len(expected) - 2, (
            f"{side}: found {len(detected)} of {len(expected)} heel strikes"
        )
        for t in expected[1:-1]:
            closest = min(detected, key=lambda d: abs(d - t))
            assert abs(closest - t) < 0.05, (
                f"{side} heel strike at {t:.3f}s recovered at {closest:.3f}s"
            )


def test_toe_off_events_are_detected(cfg):
    """Double-support time is uncomputable without them."""
    series, _, segments = _prepare(cfg, fps=60.0, n_strides=10)
    passes = direction_module.find_passes(series, cfg, segments)
    events = detect_events(series, passes[0], cfg)

    for side in ("left", "right"):
        assert len(events.of("toe_off", side)) >= 6


def test_toe_off_falls_around_60_percent_of_the_cycle(cfg):
    series, _, segments = _prepare(cfg, fps=60.0, n_strides=10, stride_time_cv=0.0)
    passes = direction_module.find_passes(series, cfg, segments)
    events = detect_events(series, passes[0], cfg)
    built = cycles_module.build_cycles(series, passes[0], 0, events, cfg)

    phases = [
        (c.ipsi_toe_off_t - c.t_start) / c.duration_s
        for c in built if c.valid and c.ipsi_toe_off_t is not None
    ]
    assert phases, "no cycle captured its own toe-off"
    assert 0.5 < float(np.median(phases)) < 0.7


def test_limbs_alternate(cfg):
    series, _, segments = _prepare(cfg, fps=60.0, n_strides=10, stride_time_cv=0.0)
    passes = direction_module.find_passes(series, cfg, segments)
    events = detect_events(series, passes[0], cfg)
    built = cycles_module.build_cycles(series, passes[0], 0, events, cfg)

    phases = [
        (c.contra_heel_strike_t - c.t_start) / c.duration_s
        for c in built if c.valid and c.contra_heel_strike_t is not None
    ]
    assert 0.4 < float(np.median(phases)) < 0.6


def test_subframe_refinement_improves_event_timing(cfg):
    """At 25 fps the frame grid is as coarse as the signal being measured."""
    errors = {}
    for refine in (False, True):
        tuned = cfg.with_overrides({"segmentation": {"subframe_refine": refine}})
        series, truth, segments = _prepare(tuned, fps=25.0, n_strides=12,
                                           stride_time_cv=0.03)
        passes = direction_module.find_passes(series, tuned, segments)
        events = detect_events(series, passes[0], tuned)

        detected = sorted(e.t for e in events.of("heel_strike", "left"))
        expected = [t for t in truth.heel_strikes_s["left"] if 0 < t < series.t[-1]]
        errors[refine] = float(np.mean([
            min(abs(d - t) for d in detected) for t in expected[1:-1]
        ])) if detected else float("inf")

    assert errors[True] < errors[False], (
        f"refined {errors[True] * 1000:.1f} ms vs integer {errors[False] * 1000:.1f} ms"
    )


# --------------------------------------------------------------------------
# cycles
# --------------------------------------------------------------------------
def test_stride_times_recovered(cfg):
    series, truth, segments = _prepare(cfg, fps=60.0, n_strides=14,
                                       stride_time_s=1.10, stride_time_cv=0.04)
    passes = direction_module.find_passes(series, cfg, segments)
    events = detect_events(series, passes[0], cfg)
    built = cycles_module.build_cycles(series, passes[0], 0, events, cfg)

    durations = np.array([c.duration_s for c in built if c.valid])
    assert durations.size >= 8
    assert abs(durations.mean() - 1.10) < 0.05


def test_stride_time_variability_is_measured_not_manufactured(cfg):
    """The central claim of the tool: a consistent walker must read as consistent."""
    from gaitscreen.features.variability import stride_time_variability

    measured = {}
    for prescribed_cv in (0.01, 0.06):
        series, truth, segments = _prepare(
            cfg, fps=60.0, n_strides=20, stride_time_cv=prescribed_cv, seed=11
        )
        passes = direction_module.find_passes(series, cfg, segments)
        events = detect_events(series, passes[0], cfg)
        built = cycles_module.build_cycles(series, passes[0], 0, events, cfg)
        result = stride_time_variability(built, cfg, series.fps)
        assert result.cv_pct is not None
        measured[prescribed_cv] = result.cv_pct

    assert measured[0.01] < 2.5, (
        f"a near-constant walker read as {measured[0.01]:.1f}% variable -- that is "
        "measurement noise being reported as gait variability"
    )
    assert measured[0.06] > measured[0.01] * 2, (
        f"variability did not track the prescribed increase: {measured}"
    )


def test_implausible_strides_are_excluded_with_a_reason(cfg):
    from gaitscreen.types import GaitCycle

    series, _, segments = _prepare(cfg, fps=60.0, n_strides=10)
    passes = direction_module.find_passes(series, cfg, segments)
    events = detect_events(series, passes[0], cfg)
    built = cycles_module.build_cycles(series, passes[0], 0, events, cfg)

    assert all(
        (c.exclusion_reason is not None) == (not c.valid) for c in built
    ), "every excluded stride must carry a reason"
    assert isinstance(built[0], GaitCycle)


def test_pass_edge_strides_are_excluded(cfg):
    """Acceleration and deceleration strides inflate variability."""
    series, _, segments = _prepare(cfg, fps=60.0, n_strides=12)
    passes = direction_module.find_passes(series, cfg, segments)
    events = detect_events(series, passes[0], cfg)
    built = cycles_module.build_cycles(series, passes[0], 0, events, cfg)

    reasons = " ".join(c.exclusion_reason or "" for c in built)
    assert "acceleration phase" in reasons
    assert "deceleration phase" in reasons


# --------------------------------------------------------------------------
# independent cross-check
# --------------------------------------------------------------------------
def test_independent_cadence_agrees_with_ground_truth(cfg):
    series, truth, _ = _prepare(cfg, fps=60.0, n_strides=14, stride_time_s=1.10,
                                stride_time_cv=0.0)
    reference = independent_cadence_spm(series, cfg)

    assert reference is not None
    assert abs(reference - 120.0 / 1.10) < 10.0


def test_independent_cadence_shares_no_machinery_with_zeni(cfg):
    """It must still work when the pelvis reference is destroyed."""
    from gaitscreen.pose.schema import HIPS

    series, _, _ = _prepare(cfg, fps=60.0, n_strides=14, stride_time_s=1.10,
                            stride_time_cv=0.0)
    for hip in HIPS:
        series.xy[:, int(hip), 0] += np.linspace(0, 400, series.n_frames)

    reference = independent_cadence_spm(series, cfg)
    assert reference is not None
    assert abs(reference - 120.0 / 1.10) < 10.0
