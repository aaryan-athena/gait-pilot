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

ALGO_VERSION = "0.2.0"

# Human-readable note on what defines this version's measurement, surfaced in
# reports so a caregiver can see when the yardstick changed.
ALGO_NOTES = (
    "Heel strike from the Zeni coordinate rule; toe-off from the end of "
    "measured ground contact; sub-frame refinement throughout; zero-phase "
    "4th-order Butterworth @ 6 Hz; sagittal single-view."
)

#: What changed, and why a stored session from an earlier version is not
#: comparable to one from this version without reprocessing.
ALGO_CHANGELOG = {
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
