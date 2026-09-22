"""Which way the camera is pointing relative to the walk.

Everything else in this pipeline assumes a *sagittal* view: the person crosses
the frame side-on, so the walking direction lies in the image plane and
anterior-posterior distances are measurable. Heel strike, toe-off, step length
and gait speed all rest on that assumption.

A *coronal* recording -- the person walking towards or away from the camera --
breaks it completely. The walking direction now points along the camera axis,
where the image carries almost no information about it: a 70 cm step projects
to a few pixels of foreshortened motion, indistinguishable from noise. The
danger is that the pipeline does not crash on such a clip. It finds peaks in a
signal that is mostly depth artefact and reports a step-length asymmetry with
a plausible-looking number attached. In a screening tool, a confident wrong
number is worse than no number, so the view has to be identified before any
sagittal metric is computed.

The test used here is deliberately not anthropometric. The obvious approach --
comparing observed shoulder separation against the width implied by trunk
height -- needs an assumed shoulder-to-trunk ratio, and on the pilot footage
that assumption cost it most of its range: a subject walking directly at the
camera measured 51 degrees off side-on where the truth is nearer 90, because
their build did not match the constant. Under-reading in exactly the case that
matters makes it unfit as a gate.

Instead this compares two things the geometry guarantees, with no constants
about human bodies in either:

* how far the subject travels **across** the image, and
* how much their apparent **size** changes.

A person walking across the frame stays at a near-constant distance, so they
translate a long way and barely change size. A person walking towards the
camera does the opposite. On the pilot clips the ratio of the two separates
the cases by a factor of roughly forty -- coronal 3.4 against sagittal 0.016
to 0.085 -- which leaves room for thresholds that are nowhere near either
group.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..pose.schema import HIPS
from ..types import PixelSeries

#: Classification results. ``indeterminate`` means the subject moved too little
#: either way to tell -- walking in place, or a clip too short to judge -- and
#: is treated as sagittal, because that is what the recording instructions ask
#: for and no evidence contradicts it.
SAGITTAL = "sagittal"
CORONAL = "coronal"
OBLIQUE = "oblique"
INDETERMINATE = "indeterminate"


@dataclass
class ViewClassification:
    """Which plane the walk was recorded in, and the evidence for it."""

    kind: str
    image_travel: float  # how far the subject crossed the frame, in leg lengths
    size_change: float  # peak-to-peak apparent size, as a share of median size
    depth_ratio: float  # size_change / image_travel; high means towards-camera
    reason: str = ""

    @property
    def is_sagittal(self) -> bool:
        """True when the sagittal metrics may be trusted at all.

        Oblique counts as sagittal here: it is foreshortened rather than
        unmeasurable, the existing camera-angle diagnostic already reports the
        foreshortening, and refusing it outright would throw away usable
        recordings.
        """
        return self.kind in (SAGITTAL, OBLIQUE, INDETERMINATE)

    @property
    def measurements(self) -> dict:
        return {
            "view_kind": self.kind,
            "view_image_travel": self.image_travel,
            "view_size_change": self.size_change,
            "view_depth_ratio": self.depth_ratio,
        }


@contextmanager
def _quiet_all_nan():
    """Frames where the subject is not detected are all-NaN by design.

    numpy warns on those, once per call, which would bury the notes that a user
    is meant to read under noise about an expected condition.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"(Mean|All-NaN)", RuntimeWarning)
        yield


def frame_scale(series: PixelSeries, smooth_s: float = 1.0) -> np.ndarray:
    """Per-frame subject size in pixels: hip-to-ankle length, both legs.

    This is the conversion factor between pixels and body-relative units at
    each instant, and in a coronal clip it changes by a factor of three from
    one end of the walk to the other. Normalising by a single whole-clip value,
    as the sagittal path does, would leave every foot trajectory dominated by
    the subject's approach rather than by their gait.

    Smoothed over a window of about a second: the raw distance dips whenever a
    knee flexes, which is gait rather than depth, and a scale that breathes
    with the gait cycle would divide that signal back out of everything
    normalised by it.
    """
    lengths = []
    for side in ("left", "right"):
        d = np.linalg.norm(series.joint(side, "hip") - series.joint(side, "ankle"),
                           axis=1)
        lengths.append(d)
    with _quiet_all_nan():
        scale = np.nanmean(np.vstack(lengths), axis=0)

    half = max(1, int(round(0.5 * smooth_s * series.fps)))
    padded = np.pad(scale, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    with _quiet_all_nan():
        smoothed = np.nanmedian(windows, axis=1)
    return smoothed[: series.n_frames]


def classify_view(series: PixelSeries, cfg: Config) -> ViewClassification:
    """Decide whether this recording is side-on, towards-camera, or between."""
    section = cfg.section("view")
    scale = frame_scale(series)
    finite = np.isfinite(scale) & (scale > 0)
    hip = series.midpoint(*HIPS)[:, 0]
    usable = finite & np.isfinite(hip)

    if int(usable.sum()) < max(10, int(series.fps)):
        return ViewClassification(
            kind=INDETERMINATE, image_travel=0.0, size_change=0.0, depth_ratio=0.0,
            reason="too few frames with a usable body scale to judge the camera angle",
        )

    typical = float(np.nanmedian(scale[usable]))
    lo, hi = np.nanpercentile(scale[usable], [5, 95])
    size_change = float((hi - lo) / typical)
    image_travel = float(np.ptp(hip[usable]) / typical)

    # A person who never moves has no direction of travel to judge, so neither
    # ratio means anything. Guard before dividing rather than after.
    min_size = float(section["min_size_change"])
    if size_change < min_size:
        return ViewClassification(
            kind=SAGITTAL if image_travel > 0 else INDETERMINATE,
            image_travel=image_travel, size_change=size_change, depth_ratio=0.0,
            reason="apparent size held steady, so the walk stayed in the image plane",
        )

    depth_ratio = size_change / max(image_travel, 1e-6)
    coronal_at = float(section["coronal_depth_ratio"])
    sagittal_at = float(section["sagittal_depth_ratio"])

    if depth_ratio >= coronal_at:
        kind, reason = CORONAL, (
            "the subject's apparent size changed far more than their position "
            "across the frame, so they walked towards or away from the camera"
        )
    elif depth_ratio <= sagittal_at:
        kind, reason = SAGITTAL, (
            "the subject crossed the frame at a near-constant size, so the walk "
            "was side-on to the camera"
        )
    else:
        kind, reason = OBLIQUE, (
            "the subject both crossed the frame and changed size, so the walk "
            "ran diagonally to the camera and distances read short"
        )

    return ViewClassification(
        kind=kind, image_travel=image_travel, size_change=size_change,
        depth_ratio=depth_ratio, reason=reason,
    )
