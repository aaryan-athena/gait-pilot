"""Per-session artifact directory: raw landmarks, angle curves, reports.

Layout::

    <artifacts_root>/<user_id>/<session_id>/
        meta.json          video info, algo version, ingestion warnings
        landmarks.parquet  raw MediaPipe output, one row per frame
        angles.npz         joint angle curves, 0-100% of cycle
        report.html        generated trend/session report

Raw landmarks are retained deliberately. When the segmentation or filtering
changes, every historical session must be re-derivable under the new algorithm;
otherwise the upgrade puts a step change into each user's trend line that is
indistinguishable from a real change in their walking. Re-running MediaPipe over
archived video would also work but costs orders of magnitude more time, and the
video may not be kept.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..pose.schema import LANDMARK_NAMES
from ..types import AngleCurves, RawLandmarks, VideoInfo

_LANDMARK_COLUMNS = [
    f"{name}_{suffix}" for name in LANDMARK_NAMES for suffix in ("x", "y", "z", "v")
]


class SessionArtifacts:
    """Filesystem accessor for one session's array data."""

    def __init__(self, root: str | Path, user_id: str, session_id: str):
        self.dir = Path(root) / user_id / session_id
        self.user_id = user_id
        self.session_id = session_id

    # -- paths -----------------------------------------------------------
    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def landmarks_path(self) -> Path:
        return self.dir / "landmarks.parquet"

    @property
    def angles_path(self) -> Path:
        return self.dir / "angles.npz"

    @property
    def report_path(self) -> Path:
        return self.dir / "report.html"

    def ensure(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    # -- meta ------------------------------------------------------------
    def save_meta(self, meta: dict) -> Path:
        self.ensure()
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return self.meta_path

    def load_meta(self) -> dict:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    # -- raw landmarks ---------------------------------------------------
    def save_raw_landmarks(self, raw: RawLandmarks) -> Path:
        """Write the raw archive as one wide row per frame."""
        self.ensure()
        n = raw.n_frames
        block = np.concatenate(
            [raw.xy, raw.z[..., None], raw.visibility[..., None]], axis=2
        )  # (n, 33, 4)
        frame = pd.DataFrame(block.reshape(n, -1), columns=_LANDMARK_COLUMNS)
        frame.insert(0, "detected", raw.detected)
        frame.insert(0, "t", raw.t)
        frame.insert(0, "frame", np.arange(n))
        frame.to_parquet(self.landmarks_path, index=False)
        return self.landmarks_path

    def load_raw_landmarks(self, video: Optional[VideoInfo] = None) -> RawLandmarks:
        """Read the raw archive back for reprocessing under a new algorithm."""
        frame = pd.read_parquet(self.landmarks_path)
        n = len(frame)
        block = frame[_LANDMARK_COLUMNS].to_numpy(dtype=float).reshape(n, len(LANDMARK_NAMES), 4)

        if video is None:
            meta = self.load_meta().get("video", {})
            video = VideoInfo(
                path=Path(meta.get("path", "unknown")),
                width=int(meta["width"]),
                height=int(meta["height"]),
                fps=float(meta["fps"]),
                n_frames=n,
                warnings=list(meta.get("warnings", [])),
            )

        return RawLandmarks(
            t=frame["t"].to_numpy(dtype=float),
            xy=block[:, :, 0:2],
            z=block[:, :, 2],
            visibility=block[:, :, 3],
            detected=frame["detected"].to_numpy(dtype=bool),
            video=video,
        )

    # -- angle curves ----------------------------------------------------
    def save_angle_curves(self, angles: AngleCurves) -> Path:
        self.ensure()
        np.savez_compressed(self.angles_path, percent=angles.percent, **angles.curves)
        return self.angles_path

    def load_angle_curves(self) -> AngleCurves:
        with np.load(self.angles_path) as data:
            percent = data["percent"]
            curves = {key: data[key] for key in data.files if key != "percent"}
        return AngleCurves(percent=percent, curves=curves)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SessionArtifacts({self.dir})"
