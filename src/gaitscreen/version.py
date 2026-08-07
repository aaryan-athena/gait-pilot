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

ALGO_VERSION = "0.1.0"

# Human-readable note on what defines this version's measurement, surfaced in
# reports so a caregiver can see when the yardstick changed.
ALGO_NOTES = (
    "Zeni coordinate-based event detection with sub-frame refinement; "
    "zero-phase 4th-order Butterworth @ 6 Hz; sagittal single-view."
)
