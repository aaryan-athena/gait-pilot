"""Recording diagnostics: why a session failed, and what to change next time.

The metrics layer already refuses to report what it cannot measure, and says so
-- "only 3 valid strides were recovered; at least 10 are needed". That is honest
but it states the *symptom*. Whoever recorded the video is left guessing which
part of their setup caused it, and guessing wrong means the next recording fails
the same way.

This module inspects the recording itself and names causes. Each check measures
a property of the video that a person can actually change -- how big the subject
is in frame, whether the camera is side-on, whether clothing is hiding the lower
legs -- and pairs it with the specific action that fixes it.

Two rules keep this useful rather than noisy:

* **Every diagnostic must be actionable.** "Low landmark confidence" is not a
  diagnostic, it is a restatement of the failure. "The subject fills 31% of the
  frame height; record from about half this distance" is.
* **Ranked, not listed.** A recording with eight problems usually has one that
  dominates. Diagnostics carry a severity and are ordered, so the app can lead
  with the single change that will help most.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from ..config import Config
from ..pose.schema import GAIT_CRITICAL, HIPS, PL, SHOULDERS, SIDE_LANDMARKS
from ..pose.to_pixels import leg_length_px
from . import view as view_module

Severity = Literal["blocker", "major", "minor"]

_SEVERITY_ORDER = {"blocker": 0, "major": 1, "minor": 2}

#: Ratio of shoulder width to trunk height in an adult, from standard
#: anthropometry (biacromial breadth ~0.23 of stature, shoulder-to-hip ~0.29).
#: Used to turn the observed horizontal shoulder separation into an estimate of
#: how far the camera sits from a true side-on view.
SHOULDER_TRUNK_RATIO = 0.79


@dataclass
class Diagnostic:
    """One identified problem with the recording."""

    code: str
    severity: Severity
    title: str
    detail: str  # what was measured, in plain language
    fix: str  # what to do differently
    measured: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return _SEVERITY_ORDER[self.severity]


@dataclass
class RecordingReport:
    """All diagnostics for one recording, most important first."""

    diagnostics: list[Diagnostic] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)

    @property
    def blockers(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "blocker"]

    @property
    def headline(self) -> Optional[Diagnostic]:
        """The single change most likely to fix the recording."""
        return self.diagnostics[0] if self.diagnostics else None

    @property
    def is_clean(self) -> bool:
        return not self.diagnostics

    def codes(self) -> list[str]:
        return [d.code for d in self.diagnostics]


def diagnose(extraction, analysis, cfg: Config) -> RecordingReport:
    """Inspect a recording and report actionable problems with it."""
    section = cfg.section("diagnostics")
    series = extraction.series
    raw = extraction.raw
    leg_px = leg_length_px(series)

    measurements: dict = {
        "subject_height_px": extraction.subject_px_height,
        "subject_height_frac": _safe_div(extraction.subject_px_height,
                                         series.video.height),
        "leg_length_px": leg_px,
        "frame_height": series.video.height,
        "detection_rate": raw.detection_rate,
    }

    checks = (
        _check_coronal_view,
        _check_subject_size,
        _check_framing,
        _check_camera_angle,
        _check_clothing_occlusion,
        _check_foot_tracking_noise,
        _check_identity_switches,
        _check_camera_shake,
        _check_walk_length,
        _check_frame_rate,
    )

    found: list[Diagnostic] = []
    for check in checks:
        result = check(extraction, analysis, series, raw, leg_px, section, measurements)
        if result is None:
            continue
        found.extend(result if isinstance(result, list) else [result])

    found.sort(key=lambda d: d.rank)
    return RecordingReport(diagnostics=found, measurements=measurements)


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------

def _check_coronal_view(extraction, analysis, series, raw, leg_px, cfg, out):
    """The camera was facing the walk rather than standing side-on to it.

    This is the one framing mistake that costs most of the measurement rather
    than degrading it. The walking direction ends up pointing along the camera
    axis, where the image cannot resolve it, so step length, gait speed and
    everything built on heel strike and toe-off are gone -- not foreshortened,
    gone. It is reported as a blocker for that reason, even though the clip
    still yields step width and side-to-side sway, which a side-on recording
    could not have given at all.

    The classification is made in features/session.py, which needs it before it
    can decide which pipeline to run; this check reports what was decided
    rather than deciding it again.
    """
    view = getattr(analysis, "view", None)
    if view is None:
        return None
    out.update(view.measurements)
    if view.kind != view_module.CORONAL:
        return None

    return Diagnostic(
        code="coronal_view",
        severity="blocker",
        title="Camera was facing the walk instead of side-on to it",
        detail=(
            "The person walks towards and away from the camera, so their "
            "direction of travel points straight at the lens. Distances along "
            "that direction cannot be recovered from the image, which rules "
            "out walking speed, step length, step-length asymmetry and time on "
            "both feet. Step width and side-to-side body sway were measured "
            "instead -- a side-on recording cannot see either."
        ),
        fix=(
            "Stand to one side of the walking path, level with its middle, and "
            "film the person crossing the frame from one edge to the other. "
            "Keep the whole body in shot for the full walk."
        ),
        measured=view.measurements,
    )


def _check_subject_size(extraction, analysis, series, raw, leg_px, cfg, out):
    """How many pixels tall the subject is, which caps all downstream precision.

    Everything the tool measures comes from where the foot landmarks sit, and
    their error in pixels is roughly fixed regardless of subject size. So the
    *relative* error -- which is what matters for a step length or a heel-strike
    time -- scales inversely with how big the subject is. A subject 150 px tall
    has a leg about 75 px long and a heel excursion of maybe 25 px, at which
    point a 2 px landmark wobble is 8% of the measurement.
    """
    height_frac = out["subject_height_frac"]
    min_frac = float(cfg["min_subject_height_frac"])
    min_leg = float(cfg["min_leg_length_px"])

    if not np.isfinite(height_frac) or height_frac <= 0:
        return None
    leg_ok = (not np.isfinite(leg_px)) or leg_px >= min_leg
    if height_frac >= min_frac and leg_ok:
        return None

    factor = min_frac / height_frac if height_frac > 0 else 2.0
    severity: Severity = "blocker" if height_frac < 0.6 * min_frac else "major"

    return Diagnostic(
        code="subject_too_small",
        severity=severity,
        title="Subject is too small in the frame",
        detail=(
            f"The person fills {height_frac:.0%} of the frame height "
            f"({out['subject_height_px']:.0f} px of {out['frame_height']} px), "
            f"giving a leg roughly {leg_px:.0f} px long. Foot position can only be "
            "located to within a pixel or two, so at this size that error is a "
            "large share of every step measurement."
        ),
        fix=(
            f"Fill at least {min_frac:.0%} of the frame height with the person -- "
            f"roughly {factor:.1f}x closer than this recording, or zoomed in that "
            "much. Frame them head to floor with a little room to spare, and keep "
            "the whole walk inside that framing."
        ),
        measured={"subject_height_frac": height_frac, "leg_length_px": leg_px},
    )


def _check_framing(extraction, analysis, series, raw, leg_px, cfg, out):
    """Whether the subject is fully inside the frame, and for how much of the clip."""
    height, width = series.video.height, series.video.width
    critical = list(GAIT_CRITICAL)
    diagnostics: list[Diagnostic] = []

    inside = (
        (series.xy[:, critical, 0] > 2)
        & (series.xy[:, critical, 0] < width - 2)
        & (series.xy[:, critical, 1] > 2)
        & (series.xy[:, critical, 1] < height - 2)
    )
    present = np.isfinite(series.xy[:, critical, :]).all(axis=2)
    fully_visible = (inside & present).all(axis=1)
    visible_frac = float(np.mean(fully_visible)) if fully_visible.size else 0.0
    out["fully_visible_frac"] = visible_frac

    min_frac = float(cfg["min_fully_visible_frac"])
    if visible_frac < min_frac:
        usable_s = visible_frac * series.video.duration_s
        severity: Severity = "blocker" if visible_frac < 0.35 else "major"
        diagnostics.append(Diagnostic(
            code="subject_not_in_frame",
            severity=severity,
            title="Subject is only in frame for part of the recording",
            detail=(
                "All the joints needed for gait measurement are visible in only "
                f"{visible_frac:.0%} of the {series.video.duration_s:.1f}s "
                f"recording -- about {usable_s:.1f}s of usable walking. The rest "
                "is the person entering, leaving, or partly out of shot."
            ),
            fix=(
                "Start recording once the person is already fully in shot and "
                "walking, and stop only after they have finished. Either trim the "
                "entering and leaving from the clip, or widen the framing so the "
                "whole walk stays inside it."
            ),
            measured={"fully_visible_frac": visible_frac, "usable_seconds": usable_s},
        ))

    # Feet specifically: clipped at the bottom edge is common when the camera is
    # aimed at the torso rather than at the whole body.
    feet = [int(SIDE_LANDMARKS[s][j]) for s in ("left", "right")
            for j in ("heel", "foot_index")]
    foot_y = series.xy[:, feet, 1]
    # At or past the frame edge, in pixels -- not "within a few percent of the
    # height". A tightly framed subject legitimately has their feet close to the
    # bottom edge, and that is good framing rather than a fault.
    near_bottom = np.isfinite(foot_y) & (foot_y < 2.0)
    clipped_frac = float(np.mean(near_bottom.any(axis=1))) if foot_y.size else 0.0
    out["feet_clipped_frac"] = clipped_frac

    if clipped_frac > 0.10:
        diagnostics.append(Diagnostic(
            code="feet_clipped",
            severity="major",
            title="The feet leave the bottom of the frame",
            detail=(
                f"A foot is at or past the bottom edge in {clipped_frac:.0%} of "
                "frames. Heel strike and toe-off are both detected from foot "
                "position, so neither can be found while the feet are cut off."
            ),
            fix=(
                "Tilt the camera down, or step back, so the floor under the person "
                "stays visible for the whole walk."
            ),
            measured={"feet_clipped_frac": clipped_frac},
        ))

    return diagnostics


def _check_camera_angle(extraction, analysis, series, raw, leg_px, cfg, out):
    """How far the camera is from a true side-on view.

    In a side-on view one shoulder hides the other, so their horizontal
    separation collapses towards zero; as the camera swings round to the front,
    the separation opens up towards the person's real shoulder width. Comparing
    the observed separation against the width implied by their trunk height
    estimates the viewing angle.

    This matters because every spatial measure is a projection onto the image
    plane. At 30 degrees off side-on, horizontal distances read about 13% short;
    at 45 degrees, 29% short. Timing is barely affected, which is why an oblique
    recording can produce a sensible cadence next to a badly understated step
    length -- and why that combination is worth naming rather than leaving the
    operator to wonder.
    """
    shoulder_sep = np.abs(
        series.point(SHOULDERS[0])[:, 0] - series.point(SHOULDERS[1])[:, 0]
    )
    trunk = np.abs(
        series.midpoint(*SHOULDERS)[:, 1] - series.midpoint(*HIPS)[:, 1]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = shoulder_sep / (SHOULDER_TRUNK_RATIO * trunk)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 10:
        return None

    sine = float(np.clip(np.median(ratio), 0.0, 1.0))
    angle_deg = float(np.degrees(np.arcsin(sine)))
    out["camera_angle_deg"] = angle_deg

    limit = float(cfg["max_sagittal_deviation_deg"])
    if angle_deg <= limit:
        return None

    shortening = 1.0 - float(np.cos(np.radians(angle_deg)))
    severity: Severity = "blocker" if angle_deg > 55 else "major"
    return Diagnostic(
        code="oblique_camera_angle",
        severity=severity,
        title="Camera is not side-on to the walking direction",
        detail=(
            f"The view is roughly {angle_deg:.0f} degrees away from a true side-on "
            "angle, where it would read near zero. At this angle horizontal "
            f"distances are foreshortened by about {shortening:.0%}, so step "
            "length and gait speed read short even when the timing is right."
        ),
        fix=(
            "Stand level with the middle of the walking path and aim the camera "
            "square at it, so the person crosses the frame from one side to the "
            "other rather than moving towards or away from you."
        ),
        measured={"camera_angle_deg": angle_deg, "foreshortening": shortening},
    )


def _check_clothing_occlusion(extraction, analysis, series, raw, leg_px, cfg, out):
    """Lower-leg landmarks much less visible than the torso.

    This is the signature of loose or flowing clothing. Pose estimation still
    reports a position for a hidden knee or ankle -- it infers one from the rest
    of the body -- so the failure is not a gap in the data but a confidently
    wrong number, which is considerably worse. The visibility score is what
    exposes it.
    """
    upper = [int(PL.LEFT_SHOULDER), int(PL.RIGHT_SHOULDER),
             int(PL.LEFT_HIP), int(PL.RIGHT_HIP)]
    lower = [int(SIDE_LANDMARKS[s][j]) for s in ("left", "right")
             for j in ("knee", "ankle", "heel", "foot_index")]

    seen = raw.detected
    if seen.sum() < 10:
        return None
    upper_vis = float(np.mean(raw.visibility[seen][:, upper]))
    lower_vis = float(np.mean(raw.visibility[seen][:, lower]))
    ratio = _safe_div(lower_vis, upper_vis)
    out["lower_upper_visibility_ratio"] = ratio
    out["lower_body_visibility"] = lower_vis

    min_ratio = float(cfg["min_lower_upper_visibility_ratio"])
    if not np.isfinite(ratio) or ratio >= min_ratio:
        return None

    severity: Severity = "blocker" if lower_vis < 0.5 else "major"
    return Diagnostic(
        code="lower_body_occluded",
        severity=severity,
        title="Legs and feet are poorly visible",
        detail=(
            f"The tracker is {lower_vis:.2f} confident about knees, ankles and "
            f"feet against {upper_vis:.2f} for the torso, a ratio of {ratio:.2f}. "
            "Something is hiding the lower legs -- long flowing clothing is the "
            "usual cause, and a dim or cluttered floor does it too. Hidden joints "
            "are estimated rather than seen, so they yield confident but wrong "
            "positions."
        ),
        fix=(
            "Record with the lower legs and ankles visible: fitted trousers, "
            "leggings, shorts, or loose trousers rolled up. Make sure the floor is "
            "lit and the feet are not against a similarly coloured background."
        ),
        measured={"lower_body_visibility": lower_vis,
                  "upper_body_visibility": upper_vis, "ratio": ratio},
    )


def _check_foot_tracking_noise(extraction, analysis, series, raw, leg_px, cfg, out):
    """Frame-to-frame jitter in the heel trajectory, relative to leg length.

    A real heel moves smoothly. High-frequency jitter is the tracker guessing,
    and since heel strike is detected from the peak of this signal, jitter
    becomes error in every stride time. Measured as deviation from a lightly
    smoothed copy of the same trajectory, so a genuinely fast swing does not
    count as noise.
    """
    if not np.isfinite(leg_px) or leg_px <= 0:
        return None

    residuals = []
    for side in ("left", "right"):
        heel = series.joint(side, "heel")[:, 0]
        finite = np.isfinite(heel)
        if finite.sum() < 20:
            continue
        values = heel[finite]
        # Three-point moving average: removes single-frame spikes only.
        smoothed = np.convolve(values, np.ones(3) / 3, mode="same")
        residuals.append(np.abs(values - smoothed)[1:-1])

    if not residuals:
        return None
    jitter = float(np.median(np.concatenate(residuals)) / leg_px)
    out["foot_jitter_norm"] = jitter

    limit = float(cfg["max_foot_jitter_norm"])
    if jitter <= limit:
        return None

    return Diagnostic(
        code="noisy_foot_tracking",
        severity="major",
        title="Foot tracking is unsteady",
        detail=(
            f"The heel position jumps about by {jitter:.1%} of leg length between "
            f"consecutive frames, where a steady recording sits under {limit:.1%}. "
            "Heel strike is found from the peak of this signal, so the noise goes "
            "straight into the stride timings."
        ),
        fix=(
            "Usually more light, or a larger subject in frame. Motion blur from a "
            "slow shutter in dim conditions is the most common cause -- record in "
            "brighter light, or outdoors in shade."
        ),
        measured={"foot_jitter_norm": jitter},
    )


def _check_identity_switches(extraction, analysis, series, raw, leg_px, cfg, out):
    """Implausible jumps in body position, meaning the tracker changed person.

    Only one person is tracked. With bystanders in shot the tracker can move to
    a different one mid-recording, and the resulting trajectory is a splice of
    two people's walks -- which produces plausible-looking numbers describing
    nobody.
    """
    if not np.isfinite(leg_px) or leg_px <= 0:
        return None

    hip = series.midpoint(*HIPS)
    finite = np.isfinite(hip).all(axis=1)
    if finite.sum() < 20:
        return None
    steps = np.linalg.norm(np.diff(hip[finite], axis=0), axis=1) / leg_px
    limit = float(cfg["max_identity_jump_norm"])
    jumps = int(np.sum(steps > limit))
    out["identity_jumps"] = jumps

    if jumps == 0:
        return None

    return Diagnostic(
        code="tracking_switched_person",
        severity="blocker" if jumps > 3 else "major",
        title="Tracking jumped to a different person",
        detail=(
            f"The tracked body moved more than {limit:.0%} of a leg length within "
            f"a single frame on {jumps} occasion(s), which a walking person cannot "
            "do. The tracker has switched between people, so the measurements mix "
            "more than one person's walk."
        ),
        fix=(
            "Record with only the person being assessed in shot. If others must be "
            "present, keep them out of the frame rather than in the background of "
            "the walking path."
        ),
        measured={"identity_jumps": jumps},
    )


def _check_camera_shake(extraction, analysis, series, raw, leg_px, cfg, out):
    """Handheld wobble, as distinct from a deliberate pan."""
    motion = extraction.camera_motion
    if not motion.determinate or not np.isfinite(motion.path_px):
        return None

    width = series.video.width
    shake = float(motion.path_px - abs(motion.net_px)) / max(width, 1)
    out["camera_shake_frac"] = shake

    limit = float(cfg["max_camera_shake_frac"])
    if shake <= limit:
        return None

    return Diagnostic(
        code="camera_shake",
        severity="major",
        title="Camera was not held still",
        detail=(
            f"The background drifts back and forth across {shake:.0%} of the frame "
            "width over the recording, beyond any steady pan. Camera movement is "
            "indistinguishable from subject movement in the image, so it corrupts "
            "the distance measurements."
        ),
        fix=(
            "Put the camera on a tripod, or prop the phone against something "
            "solid. Do not hold it, and do not follow the person as they walk."
        ),
        measured={"camera_shake_frac": shake},
    )


def _check_walk_length(extraction, analysis, series, raw, leg_px, cfg, out):
    """Whether enough strides were captured, expressed as what to add."""
    view = getattr(analysis, "view", None)
    if view is not None and view.kind == view_module.CORONAL:
        # No strides were segmented, because a coronal recording has no
        # anterior axis to segment on. Reporting "not enough strides" here
        # would send the operator off to record a longer walk when the fix is
        # to move the camera -- which the coronal_view blocker already says.
        return None
    summary = analysis.cycle_summary or {}
    n_valid = int(summary.get("n_valid", 0))
    out["strides_valid"] = n_valid

    needed = int(cfg["min_strides_target"])
    if n_valid >= needed:
        return None

    stride_s = analysis.metrics.stride_time_mean_s
    shortfall = needed - n_valid
    if stride_s:
        extra = f"roughly {shortfall * stride_s:.0f}s more walking"
    else:
        extra = "a longer walk"

    severity: Severity = "blocker" if n_valid < 4 else "major"
    return Diagnostic(
        code="walk_too_short",
        severity=severity,
        title="Not enough strides captured",
        detail=(
            f"{n_valid} usable stride(s) were recovered from this recording; "
            f"{needed} are needed before stride-to-stride variability means "
            "anything. Variability is the most fall-risk-predictive measure here, "
            "and it is also the one that needs the most strides."
        ),
        fix=(
            f"Capture {extra} -- either a longer walking path, or two to three "
            "passes back and forth within the same recording. Several passes also "
            "let each leg be the near one once, which is what makes the left/right "
            "comparison trustworthy."
        ),
        measured={"strides_valid": n_valid, "strides_needed": needed},
    )


def _check_frame_rate(extraction, analysis, series, raw, leg_px, cfg, out):
    """Frame rate, surfaced here so every recording problem is in one place."""
    fps = series.video.fps
    out["fps"] = fps
    target = float(cfg["target_fps"])
    if fps >= target:
        return None

    return Diagnostic(
        code="frame_rate_too_low",
        severity="minor",
        title="Frame rate is low for variability measurement",
        detail=(
            f"Recorded at {fps:g} fps, so gait events land on a "
            f"{1000 / fps:.0f}ms grid. Elderly stride-time variation is around "
            "25-35ms, so the grid is as coarse as the quantity being measured. "
            "Sub-frame interpolation recovers much of this, but not all."
        ),
        fix=(
            "Set the camera to 60 fps. On a phone this is in the video resolution "
            "settings, often labelled '1080p60'."
        ),
        measured={"fps": fps},
    )


# --------------------------------------------------------------------------
def _safe_div(numerator: float, denominator: float) -> float:
    if not denominator:
        return float("nan")
    return float(numerator) / float(denominator)
