"""Pixel-to-metre calibration, stored once per user/camera setup.

Two methods, with an honest account of what each can and cannot do.

``two_point``
    The user marks two points a known real-world distance apart and we store a
    single scalar, metres per pixel. This is only strictly valid in a plane
    parallel to the sensor at the depth of those points. Two practical
    consequences: the reference points should lie **along the walking path**
    (so the scale applies along the axis actually being measured) rather than
    across it, and the scale is derived on the floor plane while gait speed is
    measured from hip motion roughly a metre above it, which introduces a
    systematic error that grows the closer the camera is to the subject. Good
    enough for a trend tool -- consistency matters more than absolute accuracy
    -- but not a clinical-grade distance.

``homography``
    The user marks four points on the floor with known relative positions. This
    recovers the full floor plane, removing the perspective error along the walk
    and giving distances that stay consistent as the subject moves nearer or
    further from the camera. Preferred where the operator can manage four points.

Both are stored with the frame dimensions they were derived on. A calibration is
invalid for a recording of a different resolution, and only conditionally valid
if the camera was moved -- see :mod:`gaitscreen.calibration.verify`.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np


class CalibrationError(ValueError):
    pass


@dataclass
class Calibration:
    """A stored pixel-to-metre mapping for one user/camera setup."""

    user_id: str
    method: str  # "two_point" | "homography"
    frame_width: int
    frame_height: int
    reference_points: list[list[float]]  # image pixels, y DOWN (as clicked)
    calibration_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    reference_distance_m: Optional[float] = None
    scale_m_per_px: Optional[float] = None
    homography: Optional[np.ndarray] = None  # 3x3, image -> floor metres
    expected_subject_px_height: Optional[float] = None
    is_active: bool = True
    notes: Optional[str] = None

    # -- construction ----------------------------------------------------
    @classmethod
    def from_two_points(
        cls,
        user_id: str,
        p0: Sequence[float],
        p1: Sequence[float],
        distance_m: float,
        frame_width: int,
        frame_height: int,
        **kwargs,
    ) -> "Calibration":
        p0 = [float(p0[0]), float(p0[1])]
        p1 = [float(p1[0]), float(p1[1])]
        pixel_distance = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if pixel_distance < 1.0:
            raise CalibrationError(
                "the two reference points are less than a pixel apart; "
                "mark points that span a good fraction of the walking path"
            )
        if distance_m <= 0:
            raise CalibrationError("reference distance must be positive")

        # A reference spanning only a small part of the frame amplifies clicking
        # error into the scale factor, and thus into every speed measurement.
        span_frac = pixel_distance / max(frame_width, 1)
        notes = kwargs.pop("notes", None)
        if span_frac < 0.3:
            warning = (
                f"reference spans only {span_frac:.0%} of frame width; a 2px "
                "clicking error changes the scale factor by "
                f"{2 / pixel_distance:.1%}"
            )
            notes = f"{notes}; {warning}" if notes else warning

        return cls(
            user_id=user_id,
            method="two_point",
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            reference_points=[p0, p1],
            reference_distance_m=float(distance_m),
            scale_m_per_px=float(distance_m) / pixel_distance,
            notes=notes,
            **kwargs,
        )

    @classmethod
    def from_floor_points(
        cls,
        user_id: str,
        image_points: Sequence[Sequence[float]],
        floor_points_m: Sequence[Sequence[float]],
        frame_width: int,
        frame_height: int,
        **kwargs,
    ) -> "Calibration":
        import cv2

        if len(image_points) != 4 or len(floor_points_m) != 4:
            raise CalibrationError("homography calibration needs exactly 4 point pairs")
        src = np.asarray(image_points, dtype=np.float64)
        dst = np.asarray(floor_points_m, dtype=np.float64)
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise CalibrationError(
                "could not fit a homography; the four points must not be collinear"
            )
        return cls(
            user_id=user_id,
            method="homography",
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            reference_points=[[float(x), float(y)] for x, y in src],
            homography=np.asarray(matrix, dtype=float),
            notes=kwargs.pop("notes", None),
            **kwargs,
        )

    # -- use -------------------------------------------------------------
    def applies_to(self, width: int, height: int) -> bool:
        return width == self.frame_width and height == self.frame_height

    def image_to_floor(self, points: np.ndarray) -> np.ndarray:
        """Map image points (y DOWN) to floor-plane metres. Homography only."""
        if self.method != "homography" or self.homography is None:
            raise CalibrationError("image_to_floor requires a homography calibration")
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        homogeneous = np.column_stack([pts, np.ones(len(pts))])
        projected = homogeneous @ self.homography.T
        return projected[:, :2] / projected[:, 2:3]

    def distance_m(self, p0: Sequence[float], p1: Sequence[float]) -> float:
        """Real-world distance between two image points (y DOWN)."""
        if self.method == "homography":
            floor = self.image_to_floor(np.array([p0, p1], dtype=float))
            return float(np.hypot(*(floor[1] - floor[0])))
        if self.scale_m_per_px is None:
            raise CalibrationError("calibration has no scale factor")
        return float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) * self.scale_m_per_px

    def px_to_m(self, pixels: float) -> float:
        """Convert a pixel length to metres. Two-point only (scale is uniform)."""
        if self.method == "homography":
            raise CalibrationError(
                "a homography has no single pixel scale; use distance_m() with "
                "the actual endpoints"
            )
        if self.scale_m_per_px is None:
            raise CalibrationError("calibration has no scale factor")
        return float(pixels) * self.scale_m_per_px

    # -- serialisation ---------------------------------------------------
    def to_row(self) -> dict:
        return {
            "calibration_id": self.calibration_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "method": self.method,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "reference_distance_m": self.reference_distance_m,
            "scale_m_per_px": self.scale_m_per_px,
            "homography_json": (
                json.dumps(self.homography.tolist()) if self.homography is not None else None
            ),
            "reference_points_json": json.dumps(self.reference_points),
            "expected_subject_px_height": self.expected_subject_px_height,
            "is_active": int(self.is_active),
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row) -> "Calibration":
        homography_json = row["homography_json"]
        return cls(
            calibration_id=row["calibration_id"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            method=row["method"],
            frame_width=row["frame_width"],
            frame_height=row["frame_height"],
            reference_points=json.loads(row["reference_points_json"]),
            reference_distance_m=row["reference_distance_m"],
            scale_m_per_px=row["scale_m_per_px"],
            homography=(
                np.asarray(json.loads(homography_json), dtype=float)
                if homography_json else None
            ),
            expected_subject_px_height=row["expected_subject_px_height"],
            is_active=bool(row["is_active"]),
            notes=row["notes"],
        )
