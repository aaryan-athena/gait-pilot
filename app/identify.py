"""Glue between the app and :mod:`gaitscreen.identity`.

Kept out of the page modules so the analyse page and the people page share one
idea of what "identifying a walk" means, and so neither grows further.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from app.shared import PROJECT_ROOT, list_people, open_repository
from gaitscreen.config import Config
from gaitscreen.identity import (face_embeddings, load_models, rank,
                                 signature_from_result, walk_signature)


def enabled(cfg: Config) -> bool:
    return bool(cfg.get("identity.enabled", True))


def identify(cfg: Config, result, video_path) -> dict:
    """Signature, faces and suggestions for an analysed walk.

    The face pass reads the video again, so it only runs when someone has
    consented to face recognition: without that there is nobody to match a
    face against, and no one whose walk faces could be kept. A failure here
    costs the suggestion, never the analysis.
    """
    out: dict = {"signature": None, "faces": [], "candidates": [], "error": None}
    try:
        repository = open_repository(cfg)
        out["signature"] = signature_from_result(result, cfg)
        people = repository.people()
        if not people.empty and people["face_consent_at"].notna().any() and video_path:
            models = load_models(cfg, PROJECT_ROOT)
            out["faces"] = face_embeddings(video_path, result.extraction.raw, models, cfg)
        out["candidates"] = rank(out["signature"], out["faces"], repository.gallery(), cfg)
    except Exception as exc:  # noqa: BLE001
        print("IDENTIFICATION FAILED\n" + traceback.format_exc(), flush=True)
        out["error"] = str(exc)
    return out


def remember(cfg: Config, session_id: str, user_id: str, ident: Optional[dict]) -> None:
    """Store a saved walk's signature, confirmed, so it teaches the matcher."""
    if not ident or not ident.get("signature"):
        return
    try:
        open_repository(cfg).save_signature(session_id, user_id, ident["signature"],
                                            ident.get("faces") or [])
    except Exception:  # noqa: BLE001 - never lose a save over its signature
        print("SIGNATURE NOT SAVED\n" + traceback.format_exc(), flush=True)


def reason(candidate) -> str:
    """Why a person is suggested, e.g. "strong match on face and walk"."""
    how = {"face": "face", "walk": "build and walk", "face + walk": "face and walk"}
    sure = "strong match" if candidate.band == "strong" else "possible match"
    return f"{sure} on {how.get(candidate.evidence, candidate.evidence)}"


def candidate_label(candidate) -> str:
    return f"{candidate.user_id} ({reason(candidate)})"


def person_choices(cfg: Config, candidates) -> list[str]:
    """Suggested people first, then everyone else."""
    suggested = [c.user_id for c in candidates]
    others = [p for p in list_people(open_repository(cfg)) if p not in suggested]
    return suggested + others


def learn_from_saved_walks(cfg: Config, progress=None) -> tuple[int, int]:
    """Signatures for saved walks that have none, from their stored landmarks.

    Every saved walk was named by a person, so each counts as confirmed. No
    video is read: the raw landmarks kept for reprocessing are enough for
    the body proportions, and cadence and speed are already in the database.
    Faces cannot be recovered this way.
    """
    from gaitscreen.pipeline import extraction_from_raw
    from gaitscreen.storage.artifacts import SessionArtifacts
    from gaitscreen.types import SessionMetrics

    repository = open_repository(cfg)
    todo = repository.sessions_without_signature()
    done = skipped = 0
    for i, row in enumerate(todo.itertuples()):
        if progress:
            progress(i, len(todo))
        try:
            folder = Path(row.artifacts_dir or "")
            artifacts = SessionArtifacts(folder.parent.parent, row.user_id, row.session_id)
            if not artifacts.landmarks_path.exists():
                skipped += 1
                continue
            extraction = extraction_from_raw(artifacts.load_raw_landmarks(), cfg)
            metrics = SessionMetrics(cadence_spm=row.cadence_spm,
                                     gait_speed_mps=row.gait_speed_mps)
            view = row.view_kind or (extraction.view.kind if extraction.view else None)
            repository.save_signature(row.session_id, row.user_id,
                                      walk_signature(extraction.series, metrics, view, cfg))
            done += 1
        except Exception:  # noqa: BLE001
            print("BACKFILL FAILED for " + str(row.session_id) + "\n" + traceback.format_exc(),
                  flush=True)
            skipped += 1
    return done, skipped
