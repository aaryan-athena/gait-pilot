"""Advisory detection of assistive-device use (cane, walker, stick).

The brief suggested detecting a cane from a hand landmark's position relative to
the ground. That cannot work: MediaPipe Pose is a body-keypoint model, not an
object detector -- the cane is simply not in its output, and a wrist held low is
indistinguishable from an arm hanging at rest.

What *is* observable is the effect on the body. Normal walking has large,
reciprocal arm swing. A cane user's loaded arm is held quasi-statically and
abducted from the trunk, so its swing amplitude collapses and becomes markedly
asymmetric relative to the free arm. A walker suppresses both arms.

This remains a weak proxy, so it is treated as one:

* ``assistive_device`` in session metadata, entered by whoever recorded the walk,
  is the authoritative signal;
* this heuristic may only *lower* confidence, never assert a device and never
  raise confidence. Reporting "no cane detected" from a body-keypoint model would
  be a claim the model cannot support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import Config
from ..pose.schema import HIPS, SIDE_LANDMARKS
from ..pose.to_pixels import leg_length_px
from ..types import PixelSeries


@dataclass
class ArmSwingObservation:
    """Arm-swing amplitudes and what, if anything, they suggest."""

    left_swing_norm: float
    right_swing_norm: float
    asymmetry_ratio: float
    suppressed: bool
    asymmetric: bool
    note: Optional[str] = None

    @property
    def suspected(self) -> bool:
        return self.suppressed or self.asymmetric


def observe_arm_swing(series: PixelSeries, cfg: Config) -> ArmSwingObservation:
    """Measure normalised arm-swing amplitude per side.

    Amplitude is the peak-to-peak anterior-posterior wrist excursion relative to
    the pelvis, normalised by leg length so it is comparable across subjects and
    camera distances.
    """
    section = cfg.section("quality.assistive_device_heuristic")
    leg_length = leg_length_px(series)
    hip_x = series.midpoint(*HIPS)[:, 0]

    amplitudes: dict[str, float] = {}
    for side in ("left", "right"):
        wrist_x = series.point(SIDE_LANDMARKS[side]["wrist"])[:, 0]
        relative = wrist_x - hip_x
        relative = relative[np.isfinite(relative)]
        if relative.size < 5 or not np.isfinite(leg_length) or leg_length <= 0:
            amplitudes[side] = float("nan")
        else:
            # Robust peak-to-peak: 5th-95th percentile resists single-frame spikes.
            amplitudes[side] = float(
                (np.percentile(relative, 95) - np.percentile(relative, 5)) / leg_length
            )

    left, right = amplitudes["left"], amplitudes["right"]
    max_swing = float(section["max_wrist_swing_frac_of_leg_length"])
    min_ratio = float(section["min_arm_swing_asymmetry_ratio"])

    if not np.isfinite([left, right]).any():
        return ArmSwingObservation(
            left_swing_norm=left, right_swing_norm=right, asymmetry_ratio=float("nan"),
            suppressed=False, asymmetric=False,
            note="arm swing could not be measured (wrist landmarks unavailable)",
        )

    if not np.isfinite([left, right]).all():
        # In a sagittal view the far arm is usually hidden behind the torso for
        # most of the recording, so the asymmetry comparison is unavailable most
        # of the time. The near arm alone still detects a walker or frame, which
        # suppresses swing bilaterally.
        near = left if np.isfinite(left) else right
        suppressed = near < max_swing
        return ArmSwingObservation(
            left_swing_norm=left, right_swing_norm=right, asymmetry_ratio=float("nan"),
            suppressed=suppressed, asymmetric=False,
            note=(
                f"near arm swings only {near:.02f} of leg length; consistent with "
                "holding a walker or frame, or with a guarded gait"
                if suppressed else
                "far arm is occluded, so arm-swing asymmetry cannot be assessed "
                "from this single sagittal view"
            ),
        )

    smaller, larger = sorted((left, right))
    ratio = larger / smaller if smaller > 1e-6 else float("inf")
    suppressed = larger < max_swing

    # The asymmetry test only means anything when both arms were actually seen
    # well. In a sagittal view the far arm is foreshortened and intermittently
    # hidden behind the torso, which shrinks its measured swing for reasons that
    # have nothing to do with a walking aid -- on ordinary side-on footage of an
    # unaided walker this alone produces ratios above 2.5. Rather than emit a
    # suspicion on every recording, the test is skipped unless both wrists are
    # confidently tracked.
    wrist_visibility = min(
        float(np.nanmean(series.visibility[:, int(SIDE_LANDMARKS[s]["wrist"])]))
        for s in ("left", "right")
    )
    comparable = wrist_visibility >= 0.8
    asymmetric = comparable and ratio > min_ratio

    note = None
    if suppressed:
        note = (
            f"both arms swing very little ({left:.02f}/{right:.02f} of leg length); "
            "consistent with holding a walker or frame, or with a very guarded gait"
        )
    elif asymmetric:
        note = (
            f"arm swing is {ratio:.1f}x larger on one side "
            f"({left:.02f} vs {right:.02f} of leg length); consistent with a cane or "
            "stick, or with a one-sided impairment"
        )
    elif ratio > min_ratio and not comparable:
        note = (
            "arm swing looks uneven, but the far arm is not tracked well enough "
            f"(visibility {wrist_visibility:.2f}) for the comparison to mean "
            "anything from a single side-on view"
        )

    return ArmSwingObservation(
        left_swing_norm=left, right_swing_norm=right, asymmetry_ratio=ratio,
        suppressed=suppressed, asymmetric=asymmetric, note=note,
    )


def resolve(
    series: PixelSeries,
    cfg: Config,
    *,
    declared_device: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(device, source, note)`` for this session.

    Metadata wins. The heuristic can only add a suspicion note when nothing was
    declared -- it never overrides an operator, and never returns "none".
    """
    if declared_device:
        normalised = declared_device.strip().lower()
        if normalised in {"none", "no", "nil", ""}:
            return None, "metadata", None
        return normalised, "metadata", None

    if not bool(cfg["quality.assistive_device_heuristic.enabled"]):
        return None, None, None

    observation = observe_arm_swing(series, cfg)
    if observation.suspected:
        return "suspected", "heuristic", observation.note
    return None, None, None
