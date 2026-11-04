"""Face embeddings from a walk video, and from enrolment photos.

Uses the face detector (YuNet) and recogniser (SFace) built into OpenCV, so
nothing beyond ``opencv-python`` is needed and the same code runs on the Pi.
Both models are openly licensed (MIT and Apache-2.0); the better-known
InsightFace weights are not licensed for commercial use, which rules them out
for a product.

Each face becomes a 128-number embedding. Two embeddings of the same person
point in nearly the same direction, so their cosine similarity is high;
OpenCV documents 0.363 as the same-person threshold for SFace. Embeddings
cannot be turned back into a picture, and no image is kept.

On a walk, faces are small: a person 4 m from a 720p camera has a face about
40 pixels tall, often in profile. So the pose landmarks already found are used
to choose frames where the face is towards the camera and to zoom into the
head before detecting, rather than searching whole frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..pose.schema import PL
from ..types import RawLandmarks

DETECTOR_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx")
RECOGNISER_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_recognition_sface/face_recognition_sface_2021dec.onnx")

_HEAD = (PL.NOSE, PL.LEFT_EYE, PL.RIGHT_EYE, PL.LEFT_EAR, PL.RIGHT_EAR,
         PL.MOUTH_LEFT, PL.MOUTH_RIGHT)


class PhotoError(ValueError):
    """An enrolment photo that cannot be used, with a reason staff can act on."""


@dataclass
class FaceModels:
    detector: object
    recogniser: object
    score_threshold: float

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Faces in ``image`` as YuNet rows (box, 5 landmarks, score); may be empty."""
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return np.zeros((0, 15), np.float32) if faces is None else faces

    def embed(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recogniser.alignCrop(image, face)
        vector = np.asarray(self.recogniser.feature(aligned), np.float64).ravel()
        return vector / (np.linalg.norm(vector) or 1.0)


def load_models(cfg: Config, project_root: str | Path = ".") -> FaceModels:
    """Load (downloading on first use) the face detector and recogniser."""
    from ..pose.landmarker import _download_model

    paths = []
    for key, url, label, min_bytes in (
            ("identity.face.detector_path", DETECTOR_URL, "face detector (0.2 MB)", 100_000),
            ("identity.face.recogniser_path", RECOGNISER_URL,
             "face recogniser (~37 MB)", 10_000_000)):
        path = cfg.resolve_path(key, project_root)
        if not path.exists() or path.stat().st_size < min_bytes:
            path.unlink(missing_ok=True)
            if not bool(cfg.get("pose.auto_download", True)):
                raise FileNotFoundError(f"{label} not found at {path}; download {url}")
            _download_model(url, path, label=label, min_bytes=min_bytes)
        paths.append(str(path))
    return _load(paths[0], paths[1], float(cfg.get("identity.face.min_score", 0.8)))


@lru_cache(maxsize=2)
def _load(detector_path: str, recogniser_path: str, score: float) -> FaceModels:
    import cv2

    detector = cv2.FaceDetectorYN.create(detector_path, "", (320, 320), score, 0.3, 50)
    recogniser = cv2.FaceRecognizerSF.create(recogniser_path, "")
    return FaceModels(detector, recogniser, score)


# -- enrolment -----------------------------------------------------------------
def embed_photo(image: np.ndarray, models: FaceModels, cfg: Config) -> dict:
    """The embedding of the one face in an enrolment photo.

    Refuses a photo with no face, several faces, or a face too small to
    describe reliably: a bad enrolment would quietly mislabel walks for months.
    """
    import cv2

    longest = max(image.shape[:2])
    if longest > 1600:
        scale = 1600 / longest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    faces = models.detect(image)
    if len(faces) == 0:
        raise PhotoError("no face found; use a clear, well-lit photo of the face")
    if len(faces) > 1:
        raise PhotoError(f"{len(faces)} faces found; crop the photo to this person only")
    face = faces[0]
    size = float(min(face[2], face[3]))
    min_px = float(cfg.get("identity.face.min_photo_face_px", 60))
    if size < min_px:
        raise PhotoError(f"the face is only {size:.0f} pixels across; use a closer photo")
    return {"embedding": _pack(models.embed(image, face)), "score": round(float(face[-1]), 3),
            "size_px": round(size)}


# -- walks ---------------------------------------------------------------------
def face_embeddings(video_path: str | Path, raw: RawLandmarks, models: FaceModels,
                    cfg: Config) -> list[dict]:
    """The clearest faces of the walker in a video, best first.

    Each item: ``embedding`` (128 floats), ``score`` (detector confidence),
    ``size_px`` (face width in the original frame), ``t`` (seconds).
    """
    from ..io.video import frames

    wanted = _candidate_rows(raw, cfg)
    if not wanted:
        return []
    rows_of_decoded = raw.frame_rows
    found: list[dict] = []
    last_row = max(wanted)
    for index, image in frames(video_path):
        if index >= len(rows_of_decoded):
            break
        row = int(rows_of_decoded[index])
        if row > last_row:
            break
        if row not in wanted:
            continue
        item = _face_in_frame(image, raw, row, models, cfg)
        if item is not None:
            item["t"] = round(float(raw.t[row]), 2)
            found.append(item)
    found.sort(key=lambda f: f["score"] * min(1.0, f["size_px"] / 80.0), reverse=True)
    return found[: int(cfg.get("identity.face.keep", 8))]


def _candidate_rows(raw: RawLandmarks, cfg: Config) -> set[int]:
    """Frames where the face is plausibly visible, spread through the walk."""
    min_vis = float(cfg.get("identity.face.min_landmark_visibility", 0.8))
    n_sample = int(cfg.get("identity.face.sample_frames", 24))
    vis = raw.visibility
    eyes = np.maximum(vis[:, PL.LEFT_EYE], vis[:, PL.RIGHT_EYE])
    ok = raw.detected & (vis[:, PL.NOSE] >= min_vis) & (eyes >= min_vis)
    rows = np.flatnonzero(ok)
    if rows.size == 0:
        return set()
    # Prefer frames where the person is largest in the picture (closest),
    # then spread the choice over the walk, so one moment cannot dominate.
    size = _head_size_norm(raw)[rows]
    keep = rows[size >= np.nanpercentile(size, 50)] if rows.size > n_sample else rows
    picks = np.linspace(0, keep.size - 1, min(n_sample, keep.size)).round().astype(int)
    return {int(r) for r in keep[picks]}


def _head_size_norm(raw: RawLandmarks) -> np.ndarray:
    """Nose to mid-shoulder distance in frame heights, per row."""
    w, h = raw.video.width, raw.video.height
    nose = raw.xy[:, PL.NOSE] * (w, h)
    shoulders = 0.5 * (raw.xy[:, PL.LEFT_SHOULDER] + raw.xy[:, PL.RIGHT_SHOULDER]) * (w, h)
    return np.linalg.norm(nose - shoulders, axis=1) / h


def _face_in_frame(image: np.ndarray, raw: RawLandmarks, row: int, models: FaceModels,
                   cfg: Config) -> Optional[dict]:
    import cv2

    h, w = image.shape[:2]
    head = raw.xy[row, list(_HEAD)] * (w, h)
    head = head[np.isfinite(head).all(axis=1)]
    if head.size == 0:
        return None
    centre = head.mean(axis=0)
    shoulders = 0.5 * (raw.xy[row, PL.LEFT_SHOULDER] + raw.xy[row, PL.RIGHT_SHOULDER]) * (w, h)
    reach = float(np.linalg.norm(raw.xy[row, PL.NOSE] * (w, h) - shoulders))
    half = max(1.3 * reach, 32.0) if np.isfinite(reach) else 64.0
    x0, y0 = int(max(0, centre[0] - half)), int(max(0, centre[1] - half))
    x1, y1 = int(min(w, centre[0] + half)), int(min(h, centre[1] + half))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    crop = image[y0:y1, x0:x1]
    # Small faces are found far more reliably once enlarged.
    zoom = min(4.0, max(1.0, 320.0 / max(crop.shape[:2])))
    if zoom > 1.0:
        crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    faces = models.detect(crop)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])
    size = float(face[2]) / zoom
    if size < float(cfg.get("identity.face.min_face_px", 24)):
        return None
    return {"embedding": _pack(models.embed(crop, face)), "score": round(float(face[-1]), 3),
            "size_px": round(size)}


def _pack(vector: np.ndarray) -> list[float]:
    return [round(float(v), 5) for v in vector]
