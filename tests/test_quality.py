"""Quality scoring and assistive-device handling."""
import numpy as np

from gaitscreen.pose.schema import PL
from gaitscreen.pose.to_pixels import to_pixels
from gaitscreen.quality import assistive_device, score
from gaitscreen.types import GaitCycle
from fixtures.synthetic import occlude, synthetic_walk


def _cycles(n: int, valid: int) -> list[GaitCycle]:
    return [
        GaitCycle(
            side="left" if i % 2 else "right", t_start=float(i), t_end=float(i) + 1.0,
            start_frame=i * 60, end_frame=(i + 1) * 60, pass_index=0, valid=i < valid,
        )
        for i in range(n)
    ]


def test_good_session_scores_high(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    series = to_pixels(raw, cfg)
    report = score.score_session(series, raw, cfg, cycles=_cycles(12, 12))

    assert report.score > 0.85
    assert not report.low_confidence


def test_low_frame_rate_lowers_the_score_and_says_why(cfg):
    raw, _ = synthetic_walk(fps=25.0, n_strides=12)
    series = to_pixels(raw, cfg)
    report = score.score_session(series, raw, cfg, cycles=_cycles(12, 12))

    assert report.components["frame_rate"] == 0.5  # 25 / 50
    assert any("fps limits timing precision" in note for note in report.notes)


def test_few_cycles_lowers_the_score(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    series = to_pixels(raw, cfg)
    good = score.score_session(series, raw, cfg, cycles=_cycles(12, 12))
    poor = score.score_session(series, raw, cfg, cycles=_cycles(12, 2))

    assert poor.score < good.score
    assert any("only 2 valid gait cycles" in note for note in poor.notes)


def test_poor_visibility_lowers_the_score(cfg):
    from gaitscreen.pose.schema import GAIT_CRITICAL

    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    raw.visibility[:, [int(lm) for lm in GAIT_CRITICAL]] = 0.55
    series = to_pixels(raw, cfg)
    report = score.score_session(series, raw, cfg, cycles=_cycles(12, 12))

    assert report.components["landmark_visibility"] < 0.75
    assert any("mean visibility" in note for note in report.notes)


def test_declared_device_marks_the_session_low_confidence(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    series = to_pixels(raw, cfg)
    report = score.score_session(
        series, raw, cfg, cycles=_cycles(12, 12), declared_device="cane"
    )

    assert report.assistive_device == "cane"
    assert report.assistive_device_source == "metadata"
    assert report.components["assistive_device"] == 0.0
    assert any("does not model" in note for note in report.notes)


def test_declared_none_is_respected(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    series = to_pixels(raw, cfg)
    report = score.score_session(
        series, raw, cfg, cycles=_cycles(12, 12), declared_device="none"
    )
    assert report.assistive_device is None
    assert report.components["assistive_device"] == 1.0


def test_undetected_frames_lower_the_score(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    raw.detected[:40] = False
    series = to_pixels(raw, cfg)
    report = score.score_session(series, raw, cfg, cycles=_cycles(12, 12))

    assert report.components["detection_rate"] < 1.0
    assert any("pose was found in only" in note for note in report.notes)


# --------------------------------------------------------------------------
# arm swing heuristic
# --------------------------------------------------------------------------
def test_normal_arm_swing_is_not_flagged(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    series = to_pixels(raw, cfg)
    observation = assistive_device.observe_arm_swing(series, cfg)

    assert not observation.suspected
    assert observation.left_swing_norm > 0.08


def test_bilaterally_suppressed_arm_swing_is_flagged(cfg):
    """A walker or frame holds both arms still."""
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    series = to_pixels(raw, cfg)
    hip_x = series.midpoint(PL.LEFT_HIP, PL.RIGHT_HIP)[:, 0]
    for side in (PL.LEFT_WRIST, PL.RIGHT_WRIST):
        series.xy[:, int(side), 0] = hip_x  # pinned to the pelvis

    observation = assistive_device.observe_arm_swing(series, cfg)
    assert observation.suppressed
    assert "walker" in observation.note


def test_heuristic_never_asserts_absence_of_a_device(cfg):
    """The model cannot see a cane, so 'no device detected' is not a claim it makes."""
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    series = to_pixels(raw, cfg)
    device, source, _ = assistive_device.resolve(series, cfg, declared_device=None)

    assert device is None and source is None, (
        "with nothing declared and nothing suspicious, the heuristic must stay silent"
    )


def test_heuristic_only_suspects_never_confirms(cfg):
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    series = to_pixels(raw, cfg)
    hip_x = series.midpoint(PL.LEFT_HIP, PL.RIGHT_HIP)[:, 0]
    series.xy[:, int(PL.LEFT_WRIST), 0] = hip_x
    series.xy[:, int(PL.RIGHT_WRIST), 0] = hip_x

    device, source, note = assistive_device.resolve(series, cfg, declared_device=None)
    assert device == "suspected" and source == "heuristic"

    report = score.score_session(series, raw, cfg, cycles=_cycles(12, 12))
    assert report.components["assistive_device"] == 0.5, (
        "an unconfirmed suspicion should soften confidence, not condemn the session"
    )
    assert any("not confirmed" in note for note in report.notes)


def test_occluded_far_arm_does_not_produce_a_false_suspicion(cfg):
    """In a sagittal view the far arm is usually hidden -- that is not a cane."""
    raw, _ = synthetic_walk(fps=60.0, n_strides=12)
    occlude(raw, PL.LEFT_WRIST, 0, raw.n_frames)
    series = to_pixels(raw, cfg)

    observation = assistive_device.observe_arm_swing(series, cfg)
    assert np.isnan(observation.asymmetry_ratio)
    assert not observation.asymmetric
    assert not observation.suspected
    assert "cannot be assessed" in observation.note
