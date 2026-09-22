"""Towards-camera recordings: what is refused, and what is measured.

Two things are being tested here, and the first matters more than the second.

The first is that a coronal clip is *recognised*. The failure this guards
against is not a crash -- the sagittal pipeline runs perfectly happily on a
towards-camera video. It finds peaks in a signal that is mostly projection
artefact and reports a step-length asymmetry with a plausible number attached.
On the pilot coronal clip the old code returned 7.8% asymmetry and a cadence
of 44 steps/min against a true 80. In a screening tool a confident wrong number
is worse than a missing one, so the view has to be identified before any
sagittal metric is computed.

The second is that the measurements a coronal view genuinely supports come back
right. The fixture projects a three-dimensional walk through a pinhole lens, so
the ground truth is exact: a lateral distance and the leg length used to
normalise it are foreshortened by the same factor, which cancels.
"""
import numpy as np
import pytest

from gaitscreen.features import coronal as coronal_module
from gaitscreen.features.session import analyse
from gaitscreen.pose.to_pixels import to_pixels
from gaitscreen.quality import view as view_module
from gaitscreen.signal import filters, resample
from fixtures.synthetic import synthetic_coronal_walk, synthetic_walk


class _Extraction:
    """The parts of an Extraction that the feature stage reads."""

    def __init__(self, series, segments, view):
        self.series = series
        self.segments = segments
        self.view = view
        self.camera_motion = None
        self.speed = None
        self.subject_px_height = 0.0


def _prepare(cfg, coronal: bool, **kwargs):
    build = synthetic_coronal_walk if coronal else synthetic_walk
    raw, truth = build(**kwargs)
    series = to_pixels(raw, cfg)
    series, _ = resample.fill_short_gaps(series, cfg)
    series = filters.smooth_series(series, cfg)
    segments = resample.analysis_segments(series, cfg)
    view = view_module.classify_view(series, cfg)
    return _Extraction(series, segments, view), truth


# --------------------------------------------------------------------------
# recognising the view
# --------------------------------------------------------------------------
def test_towards_camera_walk_is_classified_coronal(cfg):
    extraction, _ = _prepare(cfg, coronal=True)
    assert extraction.view.kind == view_module.CORONAL


def test_side_on_walk_is_still_classified_sagittal(cfg):
    """The gate must not cost a single ordinary recording."""
    for speed in (120.0, 180.0, 260.0):
        extraction, _ = _prepare(cfg, coronal=False, speed_px_s=speed,
                                 n_strides=12)
        assert extraction.view.kind == view_module.SAGITTAL


def test_walking_in_place_is_not_mistaken_for_towards_camera(cfg):
    """A treadmill clip barely translates either, but it is still side-on.

    It has to reach the sagittal path: timing metrics from a treadmill walk are
    perfectly good, and only speed and step length are refused.
    """
    extraction, _ = _prepare(cfg, coronal=False, in_place=True, n_strides=12)
    assert extraction.view.is_sagittal


def test_classification_does_not_depend_on_body_proportions(cfg):
    """The discriminator is geometric, and must stay that way.

    The camera-angle check this replaces compared shoulder separation against
    the width implied by trunk height, which needs an assumed ratio between the
    two. On the pilot footage that assumption cost it most of its range: a
    subject walking straight at the camera measured 51 degrees off side-on when
    the truth was nearer 90, purely because their build did not match the
    constant. Under-reading in the one case that matters is what made it unfit
    as a gate, so this checks the replacement across builds that would have
    broken it.
    """
    for leg_length_m in (0.70, 0.85, 1.00):
        extraction, _ = _prepare(cfg, coronal=True, leg_length_m=leg_length_m)
        assert extraction.view.kind == view_module.CORONAL


# --------------------------------------------------------------------------
# refusing what the view cannot support
# --------------------------------------------------------------------------
def test_every_sagittal_metric_is_refused_with_a_reason(cfg):
    extraction, _ = _prepare(cfg, coronal=True)
    metrics = analyse(extraction, cfg).metrics

    for metric in coronal_module.SAGITTAL_ONLY_METRICS:
        assert getattr(metrics, metric) is None, metric
        assert metric in metrics.unavailable, metric
        assert "side-on" in metrics.unavailable[metric]


def test_no_gait_cycles_are_invented(cfg):
    """Nothing downstream should be handed cycles from a view with no events."""
    extraction, _ = _prepare(cfg, coronal=True)
    analysis = analyse(extraction, cfg)

    assert analysis.cycles == []
    assert analysis.events == []


def test_the_result_says_the_numbers_are_not_comparable(cfg):
    """Mixing views within one person's history is the trend-break failure.

    Same hazard as mixing algorithm versions: the numbers step, the person has
    not, and the two are indistinguishable after the fact.
    """
    extraction, _ = _prepare(cfg, coronal=True)
    notes = " ".join(analyse(extraction, cfg).notes)
    assert "not comparable" in notes


# --------------------------------------------------------------------------
# measuring what it does support
# --------------------------------------------------------------------------
def test_step_width_is_recovered(cfg):
    for prescribed in (0.12, 0.18, 0.26):
        extraction, truth = _prepare(cfg, coronal=True,
                                     step_width_ratio=prescribed)
        measured = analyse(extraction, cfg).metrics.step_width_norm
        assert measured is not None
        assert abs(measured - prescribed) < 0.02, prescribed


def test_step_width_does_not_depend_on_distance_from_the_camera(cfg):
    """The whole point of normalising per frame rather than per clip.

    The subject's apparent size changes by a factor of three across a coronal
    walk. A step width normalised by one whole-clip scale would read high at
    the near end and low at the far end, and its average would then depend on
    where the person happened to start.
    """
    near, _ = _prepare(cfg, coronal=True, start_depth_m=4.0, end_depth_m=2.0)
    far, _ = _prepare(cfg, coronal=True, start_depth_m=12.0, end_depth_m=6.0)

    a = analyse(near, cfg).metrics.step_width_norm
    b = analyse(far, cfg).metrics.step_width_norm
    assert a is not None and b is not None
    assert abs(a - b) < 0.02


def test_lateral_sway_is_recovered(cfg):
    extraction, truth = _prepare(cfg, coronal=True, sway_ratio=0.030)
    measured = analyse(extraction, cfg).metrics.trunk_lateral_sway_norm
    expected = truth.extras["trunk_lateral_sway_norm"]

    assert measured is not None
    assert abs(measured - expected) < 0.01


def test_cadence_is_recovered(cfg):
    for stride_time in (1.00, 1.50):
        extraction, _ = _prepare(cfg, coronal=True, stride_time_s=stride_time,
                                 n_strides=12)
        cadence = analyse(extraction, cfg).metrics.cadence_spm
        assert cadence is not None
        assert abs(cadence - 120.0 / stride_time) < 8.0, stride_time


def test_stride_time_variability_is_never_reported(cfg):
    """Deliberately withheld, not merely absent.

    The period is stable enough to average over a pass, but individual heel
    strikes cannot be located precisely enough in this view to time one stride
    against the next. A CV built from imprecise event times measures the
    detector, not the person -- and it would be read as the fall-risk signal
    that stride-time variability is.
    """
    extraction, _ = _prepare(cfg, coronal=True, n_strides=14)
    metrics = analyse(extraction, cfg).metrics
    assert metrics.stride_time_cv_pct is None


# --------------------------------------------------------------------------
# turns
# --------------------------------------------------------------------------
def test_a_turn_splits_the_walk_into_separate_passes(cfg):
    extraction, _ = _prepare(cfg, coronal=True, turn_around=True, n_strides=20)
    passes = coronal_module.find_passes(extraction.series, cfg)

    assert len(passes) == 2
    assert {p.towards_camera for p in passes} == {True, False}


def test_step_width_survives_a_turn(cfg):
    """Left and right trade sides of the image when the subject turns round.

    A signed ankle separation would cancel between the approach and the
    retreat, and report a step width near zero for someone walking normally.
    """
    there, _ = _prepare(cfg, coronal=True, step_width_ratio=0.18, n_strides=10)
    and_back, _ = _prepare(cfg, coronal=True, step_width_ratio=0.18,
                           n_strides=20, turn_around=True)

    one_way = analyse(there, cfg).metrics.step_width_norm
    both_ways = analyse(and_back, cfg).metrics.step_width_norm
    assert one_way is not None and both_ways is not None
    assert abs(one_way - both_ways) < 0.02
