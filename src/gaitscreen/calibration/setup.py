"""The camera setup: how much floor the camera's view covers.

This is the simplest calibration that still gives walking speed in metres per
second. Two tape marks go on the floor along the line the person walks, one at
the left edge of the picture and one at the right; the distance between them is
the *coverage length*. Divided by the frame width it is the size of one pixel
at the walking line, and that is all a side-on speed measurement needs.

It is stored once per camera setup rather than per person, because it describes
where the camera stands, not who walks past it. Three consequences follow, and
each is handled here rather than left to the caller:

* **Resolution.** A pixel scale does not survive a change of video size, and
  the stored-calibration check refuses one made at a different resolution. The
  coverage length does survive -- it is a physical distance across the whole
  frame -- so a calibration is built for each video at that video's own width.
* **Camera drift.** The per-person setup could notice a moved camera from the
  subject's height in pixels. A shared setup cannot, because different people
  are different heights, so that check is not attempted.
* **Storage.** Calibrations are keyed by person in the database, so the setup
  is stored under a reserved id that the app keeps out of the person lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import Calibration, CalibrationError

#: Database id the camera setup is stored under. Not a person; filtered out of
#: every person list by :func:`is_reserved`.
CAMERA_SETUP_ID = "__camera_setup__"

#: Plausible coverage, in metres. Below the lower bound the person would cross
#: the frame in under a stride; above the upper one they would be a few pixels
#: tall. Either is almost certainly a typing error (centimetres, or feet).
MIN_COVERAGE_M = 1.0
MAX_COVERAGE_M = 25.0

#: Reference width the setup is stored at. Arbitrary: only the ratio of the
#: coverage length to the frame width is used, and that is resolution-free.
_REFERENCE_WIDTH = 1000


@dataclass
class CameraSetup:
    """The stored setup, in the terms it was entered in."""

    coverage_m: float
    calibration_id: str
    created_at: str
    notes: Optional[str] = None

    def calibration_for(self, width: int, height: int) -> Calibration:
        """A calibration for one video, at that video's own resolution.

        It keeps the setup's id, so a saved session records which setup its
        speed came from -- and so the database's link from session to
        calibration points at a row that exists.
        """
        return Calibration.from_two_points(
            CAMERA_SETUP_ID, (0.0, 0.0), (float(width), 0.0), self.coverage_m,
            int(width), int(height),
            calibration_id=self.calibration_id, created_at=self.created_at,
            notes=self.notes,
        )

    def metres_per_pixel(self, width: int) -> float:
        return self.coverage_m / max(int(width), 1)


def validate_coverage(coverage_m: float) -> None:
    """Refuse a coverage length that is almost certainly a unit mistake."""
    if not MIN_COVERAGE_M <= float(coverage_m) <= MAX_COVERAGE_M:
        raise CalibrationError(
            f"a camera view covering {coverage_m:g} m of floor is outside the "
            f"plausible {MIN_COVERAGE_M:g}-{MAX_COVERAGE_M:g} m range for a "
            "walking recording; check the units (metres, not centimetres or feet)"
        )


def save_setup(repository, coverage_m: float, *, notes: Optional[str] = None) -> CameraSetup:
    """Store a new camera setup, replacing the active one."""
    validate_coverage(coverage_m)
    calibration = Calibration.from_two_points(
        CAMERA_SETUP_ID, (0.0, 0.0), (float(_REFERENCE_WIDTH), 0.0),
        float(coverage_m), _REFERENCE_WIDTH, _REFERENCE_WIDTH,
        notes=notes,
    )
    repository.save_calibration(calibration)
    return _from_calibration(calibration)


def load_setup(repository) -> Optional[CameraSetup]:
    """The active camera setup, or None if none has been entered yet."""
    calibration = repository.active_calibration(CAMERA_SETUP_ID)
    return _from_calibration(calibration) if calibration is not None else None


def setup_from_calibration(calibration: Calibration) -> CameraSetup:
    """Express any single-scale calibration as a coverage length.

    Used when the scale comes from marked points or a known walk instead of
    being typed in: the scale times the frame width is the coverage length.
    """
    if calibration.method != "two_point" or not calibration.scale_m_per_px:
        raise CalibrationError("only a single-scale calibration has a coverage length")
    return CameraSetup(
        coverage_m=float(calibration.scale_m_per_px * calibration.frame_width),
        calibration_id=calibration.calibration_id,
        created_at=calibration.created_at,
        notes=calibration.notes,
    )


#: Database id the built-in sample videos' setups are stored under.
SAMPLE_SETUP_ID = "__sample_setup__"


def is_reserved(user_id: str) -> bool:
    """True for database ids that are not people.

    Reserved ids are wrapped in double underscores, which a person's name as
    typed into the app never is: names pass through ``safe_user_id``, which
    keeps underscores but nobody types a name that both starts and ends with two.
    """
    return bool(user_id) and user_id.startswith("__") and user_id.endswith("__")


def ensure_fixed_setup(repository, calibration_id: str, coverage_m: float,
                       *, notes: Optional[str] = None) -> CameraSetup:
    """A setup with a fixed id, stored once: for videos that bring their own.

    A sample video was filmed somewhere specific, so it must be measured with
    that place's coverage rather than whatever setup the viewer has saved. It
    still needs a row in the database, because a saved session records the id
    of the calibration that measured it and the database enforces that link.
    Stored without becoming the active camera setup.
    """
    validate_coverage(coverage_m)
    if repository.get_calibration(calibration_id) is None:
        calibration = Calibration.from_two_points(
            SAMPLE_SETUP_ID, (0.0, 0.0), (float(_REFERENCE_WIDTH), 0.0),
            float(coverage_m), _REFERENCE_WIDTH, _REFERENCE_WIDTH,
            calibration_id=calibration_id, notes=notes,
        )
        repository.save_calibration(calibration, make_active=False)
    return CameraSetup(coverage_m=float(coverage_m), calibration_id=calibration_id,
                       created_at="", notes=notes)


def _from_calibration(calibration: Calibration) -> CameraSetup:
    return setup_from_calibration(calibration)
