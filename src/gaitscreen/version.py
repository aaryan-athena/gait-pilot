"""Algorithm versioning.

This is deliberately separate from the package version. It identifies the
*measurement* produced by the pipeline.

Why it matters: in a longitudinal tool, changing the segmentation or filtering
code changes the numbers, so mixing algorithm versions inside one user's
history creates a step change that looks exactly like clinical decline. Every
stored session records the ALGO_VERSION that produced it, and the CLI can
reprocess a user's whole history under a single version from the retained raw
landmarks.

Bump MINOR for changes that alter output values; bump PATCH for changes that
cannot (logging, reporting, docstrings).
"""

ALGO_VERSION = "0.5.0"

# Human-readable note on what defines this version's measurement, surfaced in
# reports so a caregiver can see when the yardstick changed.
ALGO_NOTES = (
    "Heel strike from the Zeni coordinate rule; toe-off from the end of "
    "measured ground contact; sub-frame refinement throughout; zero-phase "
    "4th-order Butterworth @ 6 Hz. Sagittal single-view, with a separate and "
    "much smaller measurement for towards-camera recordings."
)

#: What changed, and why a stored session from an earlier version is not
#: comparable to one from this version without reprocessing.
ALGO_CHANGELOG = {
    "0.5.0": (
        "Steps are paired only within one walking pass. Pooling every heel "
        "strike across the recording paired the last strike before a turn or "
        "stop with the first one after it, and that 4-7 s gap entered one "
        "side's mean step time: the garage pilot walks stored step-time "
        "asymmetry of 22-99% for healthy walkers, against 0.5-3% now. Step-"
        "length asymmetry moves by up to 2 points for the same reason. "
        "Cadence and stride-time variability are unchanged."
    ),
    "0.4.0": (
        "The clock is rebuilt from the camera's own frame timestamps. Phones "
        "skip frames in dim light and the old clock counted the frames that "
        "were kept as if evenly spaced, which mis-timed individual strides by "
        "5-20% on the garage pilot clips and reported a stride-time CV of "
        "11.8% for a metronome-paced walk whose true value is near 2%. Any "
        "session from a clip that skipped frames has different timing metrics "
        "under this version and must be reprocessed from the original video; "
        "clips that skipped nothing are unchanged."
    ),
    "0.3.0": (
        "Recordings filmed towards the camera are now identified and measured "
        "separately, instead of being run through the sagittal pipeline as if "
        "they were side-on. That path produced numbers -- a step-length "
        "asymmetry and a cadence off by a factor of two on the pilot clip -- "
        "from a signal that was mostly projection artefact. Such a session now "
        "reports step width and lateral trunk sway, which a side view cannot "
        "measure at all, and refuses the six metrics that need a side view. "
        "Sagittal sessions are unaffected and their values are unchanged; the "
        "version is bumped because the stored schema and the set of reported "
        "metrics both changed. Sessions are never pooled across camera angles."
    ),
    "0.2.0": (
        "Toe-off is now measured as the end of the foot's ground contact "
        "rather than inferred from the toe's anterior minimum. The old rule "
        "was badly conditioned -- the pelvis travels over a planted foot, so "
        "the anterior position slides throughout stance with no real minimum "
        "-- and it overstated stance and double support. Stance now measures "
        "58-59% of the cycle at a normal pace against a textbook 60%, where "
        "the old rule gave 66-73%. Double-support values from 0.1.x are not "
        "comparable with these and must be reprocessed."
    ),
    "0.1.0": "Initial pipeline.",
}
