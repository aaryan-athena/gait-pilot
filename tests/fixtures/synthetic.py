"""Synthetic gait with known ground truth.

Without a motion-capture reference there is no way to check whether the pipeline
*measures* stride-time variability or merely *manufactures* it from sampling
noise. So the tests generate landmark trajectories from prescribed stride times
and then ask the pipeline to recover them.

The model is deliberately shaped to match the detector's assumption rather than
to be anatomically rich: each foot's anterior offset relative to the pelvis is
``cos(2*pi*phase)``, so it is at a maximum exactly at heel strike -- which is the
definition the Zeni coordinate method keys on. That makes recovered event times
directly comparable to the prescribed ones.

Phase advances piecewise-linearly between prescribed heel strikes, so
*non-uniform* stride times are represented exactly. A generator that used a fixed
frequency could not test variability at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaitscreen.pose.schema import N_LANDMARKS, PL
from gaitscreen.types import RawLandmarks, VideoInfo


#: Share of the gait cycle the foot spends on the ground, from initial contact
#: to toe-off. 60% is the textbook value at a comfortable pace.
STANCE_FRACTION = 0.60

#: Double-support fraction implied by a 60% stance and a 50% contralateral
#: offset: the opposite foot lifts at 10% and lands at 50%, so the two
#: double-support phases are 0-10% and 50-60% of the cycle.
EXPECTED_DOUBLE_SUPPORT_PCT = 20.0


@dataclass
class GroundTruth:
    """What the generator was told to produce."""

    fps: float
    stride_times_s: dict[str, list[float]]
    heel_strikes_s: dict[str, list[float]]
    speed_px_s: float
    step_length_px: float
    leg_length_px: float
    scale_m_per_px: float | None = None
    extras: dict = field(default_factory=dict)

    @property
    def all_stride_times(self) -> np.ndarray:
        return np.array(self.stride_times_s["left"] + self.stride_times_s["right"])

    @property
    def stride_time_cv_pct(self) -> float:
        values = self.all_stride_times
        return float(100.0 * values.std(ddof=1) / values.mean())

    @property
    def speed_m_s(self) -> float | None:
        if self.scale_m_per_px is None:
            return None
        return self.speed_px_s * self.scale_m_per_px


def synthetic_walk(
    *,
    fps: float = 60.0,
    n_strides: int = 12,
    stride_time_s: float = 1.10,
    stride_time_cv: float = 0.03,
    speed_px_s: float = 180.0,
    leg_length_px: float = 160.0,
    step_amplitude_px: float = 55.0,
    width: int = 1280,
    height: int = 720,
    noise_px: float = 0.0,
    seed: int = 7,
    in_place: bool = False,
    scale_m_per_px: float | None = 0.005,
) -> tuple[RawLandmarks, GroundTruth]:
    """Generate a sagittal walk with prescribed stride-to-stride variability.

    ``in_place`` reproduces treadmill or camera-tracked footage (no net
    translation), which the feasibility check must refuse to measure speed from.
    """
    rng = np.random.default_rng(seed)

    # Prescribe left-foot stride times, then right offset by half a stride.
    left_strides = _draw_strides(rng, n_strides, stride_time_s, stride_time_cv)
    right_strides = _draw_strides(rng, n_strides, stride_time_s, stride_time_cv)

    left_hs = np.concatenate([[0.0], np.cumsum(left_strides)])
    right_hs = 0.5 * stride_time_s + np.concatenate([[0.0], np.cumsum(right_strides)])

    duration = min(left_hs[-1], right_hs[-1])
    n_frames = int(np.floor(duration * fps)) + 1
    t = np.arange(n_frames) / fps

    phase = {
        "left": _phase_from_events(t, left_hs),
        "right": _phase_from_events(t, right_hs),
    }

    hip_y = leg_length_px
    hip_x = np.zeros_like(t) if in_place else speed_px_s * t
    # A little vertical bounce at twice the stride frequency (two steps/stride).
    bounce = 0.01 * leg_length_px * np.cos(4 * np.pi * phase["left"])

    xy = np.zeros((n_frames, N_LANDMARKS, 2))

    def place(index: PL, x: np.ndarray, y: np.ndarray) -> None:
        xy[:, int(index), 0] = x
        xy[:, int(index), 1] = y

    hip_mid_y = hip_y + bounce
    place(PL.LEFT_HIP, hip_x, hip_mid_y)
    place(PL.RIGHT_HIP, hip_x, hip_mid_y)

    # Trunk: shoulders above the hips with a small anterior-posterior lean, the
    # sagittal substitute for lateral trunk sway when no frontal view exists.
    trunk_length = 0.9 * leg_length_px
    lean = 0.02 * trunk_length * np.cos(2 * np.pi * phase["left"])
    place(PL.LEFT_SHOULDER, hip_x + lean, hip_mid_y + trunk_length)
    place(PL.RIGHT_SHOULDER, hip_x + lean, hip_mid_y + trunk_length)
    place(PL.NOSE, hip_x + lean, hip_mid_y + trunk_length + 0.25 * leg_length_px)

    # Stride length must equal speed x stride time, or the foot would not stay
    # put under a pelvis travelling at ``speed_px_s``. With no translation
    # (treadmill), the belt supplies an equivalent stride length instead.
    stride_length_px = (
        speed_px_s * stride_time_s if not in_place else step_amplitude_px / 0.3
    )

    for side, sign in (("left", 1.0), ("right", -1.0)):
        p = phase[side]
        anterior = _foot_anterior(p, stride_length_px)
        lift = _swing_lift(p) * 0.10 * leg_length_px

        ankle_x = hip_x + anterior
        # A small mid-stance dip, so the ankle's vertical minimum is a real
        # feature rather than a flat line. This is what the ankle-velocity
        # cross-check detector keys on, and it deliberately lands at ~15% of the
        # cycle -- foot-flat, not initial contact -- which is exactly why that
        # method is a cross-check here and not the primary detector.
        stance_dip = 0.02 * leg_length_px * np.exp(-(((np.mod(p, 1.0) - 0.15) / 0.10) ** 2))
        ankle_y = 0.06 * leg_length_px + lift - stance_dip
        place(_pl(side, "ANKLE"), ankle_x, ankle_y)
        # Heel behind the ankle, toe ahead of it; the whole foot shares one
        # ground-contact interval, so toe-off is the end of stance for both.
        place(_pl(side, "HEEL"), ankle_x - 0.04 * leg_length_px, ankle_y)
        place(_pl(side, "FOOT_INDEX"),
              ankle_x + 0.12 * leg_length_px, ankle_y * 0.7)

        # Knee by two-link inverse kinematics from the hip and ankle, rather
        # than at their midpoint. The midpoint constrains the limb to a shallow
        # bend and caps knee flexion near 15 degrees, which is far too little to
        # test the angle computation against; solving the linkage produces the
        # real ~0 degrees at contact and ~65 degrees at mid-swing.
        knee_x, knee_y = _knee_by_ik(
            hip_x, hip_mid_y, ankle_x, ankle_y,
            segment=0.505 * leg_length_px, direction=1.0,
        )
        place(_pl(side, "KNEE"), knee_x, knee_y)

        # Arms swing in antiphase with the ipsilateral leg.
        arm_swing = 0.35 * step_amplitude_px * np.cos(2 * np.pi * p + np.pi)
        place(_pl(side, "WRIST"), hip_x + arm_swing, hip_mid_y + 0.25 * trunk_length)
        place(_pl(side, "ELBOW"), hip_x + 0.5 * arm_swing, hip_mid_y + 0.5 * trunk_length)
        _ = sign

    # Fill remaining (head/hand detail) landmarks so nothing is left at origin.
    for index in range(N_LANDMARKS):
        if not xy[:, index, :].any():
            xy[:, index, 0] = xy[:, int(PL.NOSE), 0]
            xy[:, index, 1] = xy[:, int(PL.NOSE), 1]

    # Centre horizontally so the subject starts at the left of frame.
    xy[:, :, 0] += 0.15 * width
    if noise_px:
        xy += rng.normal(0.0, noise_px, size=xy.shape)

    info = VideoInfo(
        path=Path("synthetic.mp4"), width=width, height=height, fps=fps,
        n_frames=n_frames,
    )
    raw = RawLandmarks(
        t=t,
        xy=_to_normalised(xy, width, height),
        z=np.zeros((n_frames, N_LANDMARKS)),
        visibility=np.full((n_frames, N_LANDMARKS), 0.95),
        detected=np.ones(n_frames, dtype=bool),
        video=info,
    )

    truth = GroundTruth(
        fps=fps,
        stride_times_s={"left": left_strides.tolist(), "right": right_strides.tolist()},
        heel_strikes_s={
            "left": [v for v in left_hs if v <= t[-1]],
            "right": [v for v in right_hs if 0 <= v <= t[-1]],
        },
        speed_px_s=0.0 if in_place else speed_px_s,
        step_length_px=step_amplitude_px,
        leg_length_px=leg_length_px,
        scale_m_per_px=scale_m_per_px,
        extras={"in_place": in_place, "noise_px": noise_px},
    )
    return raw, truth


def occlude(
    raw: RawLandmarks, landmark: PL, start: int, stop: int, *, visibility: float = 0.1
) -> RawLandmarks:
    """Drop a landmark's visibility over a frame range, to test gap handling."""
    raw.visibility[start:stop, int(landmark)] = visibility
    return raw


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------
def _pl(side: str, joint: str) -> PL:
    return PL[f"{side.upper()}_{joint}"]


def _draw_strides(
    rng: np.random.Generator, n: int, mean_s: float, cv: float
) -> np.ndarray:
    """Stride times with an exact realised mean, so ground truth is unambiguous."""
    values = rng.normal(mean_s, mean_s * cv, size=n)
    return np.clip(values, 0.4 * mean_s, 2.0 * mean_s)


def _phase_from_events(t: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Cycle phase in cycles, advancing linearly between successive events.

    Phase is integer-valued exactly at each heel strike, so a detector keying on
    ``cos(2*pi*phase)`` maxima should recover the prescribed event times.
    """
    cycles = np.arange(events.size, dtype=float)
    return np.interp(t, events, cycles)


def _knee_by_ik(
    hip_x: np.ndarray, hip_y: np.ndarray,
    ankle_x: np.ndarray, ankle_y: np.ndarray,
    *, segment: float, direction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Knee position for equal-length thigh and shank, bending anteriorly.

    Standard two-circle intersection. Where the hip-ankle distance exceeds the
    limb's reach the leg is treated as fully extended, which is the physical
    limit rather than an error.
    """
    dx, dy = ankle_x - hip_x, ankle_y - hip_y
    distance = np.hypot(dx, dy)
    distance = np.clip(distance, 1e-6, 2.0 * segment)

    along = distance / 2.0  # equal segments put the foot of the perpendicular midway
    offset = np.sqrt(np.maximum(segment**2 - along**2, 0.0))

    ux, uy = dx / distance, dy / distance
    # Rotate the unit vector to get the perpendicular, then bend forwards.
    px, py = -uy, ux
    sign = np.where(px * direction >= 0, 1.0, -1.0)

    return (hip_x + along * ux + sign * offset * px,
            hip_y + along * uy + sign * offset * py)


def _foot_anterior(phase: np.ndarray, stride_length: float) -> np.ndarray:
    """Anterior offset of a foot that is planted through stance, then swings.

    This is what a real foot does, and modelling it matters for more than
    realism. A foot on a cosine path never stops moving, so a fixture built
    that way cannot exercise any detector that works from ground contact -- and
    it quietly misrepresents the signal the pipeline actually sees.

    Through stance the foot holds still in the world while the pelvis travels
    over it, so the pelvis-relative offset falls linearly from +0.3 to -0.3 of a
    stride length. Through swing it travels one stride length forward. The
    offset is therefore at its maximum at heel strike and its minimum at
    toe-off, which keeps both the Zeni rules valid on this fixture as well.
    """
    fractional = np.mod(phase, 1.0)
    pelvis = fractional * stride_length

    swing = (fractional - STANCE_FRACTION) / (1.0 - STANCE_FRACTION)
    foot = np.where(
        fractional <= STANCE_FRACTION,
        0.0,
        stride_length * _swing_travel(np.clip(swing, 0.0, 1.0)),
    )
    return foot - pelvis + 0.5 * STANCE_FRACTION * stride_length


#: Share of swing spent accelerating the foot off the ground, and share spent
#: arresting it before contact. Both are short, and the arrest is the shorter
#: of the two: a foot is placed rather than eased down, which is why heel
#: strike is an impact. They are not zero, because a foot cannot change speed
#: instantaneously and a fixture that pretends otherwise puts a corner in the
#: signal that no real -- and no filtered -- recording contains.
SWING_RAMP_UP = 0.04
SWING_RAMP_DOWN = 0.02


def _swing_travel(s: np.ndarray) -> np.ndarray:
    """Fraction of a stride travelled by ``s`` through swing, ``s`` in [0, 1].

    Swing speed follows a trapezoid with raised-cosine ramps, rather than the
    half-sine a cosine path implies. The distinction is not cosmetic. A sine
    profile leaves and reaches the ground at zero speed, so a foot on it spends
    an eighth of its swing below any sane ground-contact threshold and the
    fixture reads back ~65% stance when it was told to produce 60%. It also
    puts the anterior maximum several frames *before* the prescribed heel
    strike, which silently biases every event-timing test on this fixture.
    """
    up, down = SWING_RAMP_UP, SWING_RAMP_DOWN
    # Normalise so the profile covers exactly one stride length: each ramp
    # contributes half the area a flat segment of the same length would.
    peak = 1.0 / (1.0 - 0.5 * (up + down))

    flat = peak * (0.5 * up + (s - up))

    rising = peak * 0.5 * (s - (up / np.pi) * np.sin(np.pi * s / up))

    tail = np.clip(1.0 - s, 0.0, None) / down
    falling = 1.0 - peak * 0.5 * down * (tail - np.sin(np.pi * tail) / np.pi)

    return np.where(s < up, rising, np.where(s > 1.0 - down, falling, flat))


def _swing_lift(phase: np.ndarray) -> np.ndarray:
    """Raised-cosine foot clearance, zero while the foot is on the ground."""
    fractional = np.mod(phase, 1.0)
    within = fractional > STANCE_FRACTION
    normalised = (fractional - STANCE_FRACTION) / (1.0 - STANCE_FRACTION)
    lift = np.zeros_like(phase)
    lift[within] = np.sin(np.pi * normalised[within])
    return lift


def _to_normalised(xy_up: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert pixel, y-up coordinates back to MediaPipe's normalised y-down form."""
    out = np.empty_like(xy_up)
    out[..., 0] = xy_up[..., 0] / width
    out[..., 1] = (height - xy_up[..., 1]) / height
    return out
