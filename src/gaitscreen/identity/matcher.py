"""Rank the people a walk could belong to.

The output is a *suggestion* for a person to confirm, never an assignment:
a walk filed under the wrong person corrupts two trends at once, and their
flags with them. So the matcher is free to say "not sure" (an empty list),
and says how strong each suggestion is and what it rests on.

* **Face** evidence is strong when it is there. A person's face gallery holds
  their enrolment photos plus the faces from walks already confirmed as theirs.
* **Walk** evidence compares body proportions (and, weakly, cadence and speed)
  with the person's confirmed walks from the same camera angle. It is never
  reported as strong on its own: how well it separates people has not been
  measured yet.
* When both point at the same person, the suggestion is strong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..config import Config
from .signature import BODY_TRAITS, GAIT_TRAITS

STRONG, POSSIBLE = "strong", "possible"

#: Fallback spread of each trait between two walks of the same person, used
#: until enough people have repeat walks to estimate it from the data. Body
#: ratios are typically repeatable to a few percent; cadence and speed vary
#: far more from day to day.
_DEFAULT_SPREAD = {
    "torso_to_leg": 0.03, "thigh_to_shank": 0.05, "upper_arm_to_forearm": 0.06,
    "head_to_torso": 0.04, "shoulder_to_hip_width": 0.06,
    "cadence_spm": 6.0, "gait_speed_mps": 0.10,
}


@dataclass
class PersonGallery:
    """What is known about one person, for matching."""

    user_id: str
    faces: list[np.ndarray] = field(default_factory=list)   # unit vectors
    walks: list[dict] = field(default_factory=list)         # confirmed signatures


@dataclass
class Candidate:
    user_id: str
    band: str              # "strong" | "possible"
    evidence: str          # "face" | "walk" | "face + walk"
    face_similarity: Optional[float] = None
    walk_distance: Optional[float] = None

    def describe(self) -> str:
        parts = []
        if self.face_similarity is not None:
            parts.append(f"face similarity {self.face_similarity:.2f}")
        if self.walk_distance is not None:
            parts.append(f"walk distance {self.walk_distance:.1f}")
        return f"{self.evidence} match, {self.band} ({', '.join(parts)})"


def rank(signature: Optional[dict], faces: Sequence[dict], gallery: Sequence[PersonGallery],
         cfg: Config) -> list[Candidate]:
    """Suggestions for who walked, most likely first; empty means not sure."""
    face_scores = _face_scores(faces, gallery)
    walk_scores = _walk_scores(signature, gallery, cfg)

    strong_face = float(cfg.get("identity.face.strong", 0.50))
    possible_face = float(cfg.get("identity.face.possible", 0.363))
    margin = float(cfg.get("identity.face.margin", 0.08))
    max_walk = float(cfg.get("identity.walk.max_distance", 2.0))

    ordered_faces = sorted(face_scores.items(), key=lambda kv: kv[1], reverse=True)
    runner_up = ordered_faces[1][1] if len(ordered_faces) > 1 else -1.0
    ordered_walks = sorted(walk_scores.items(), key=lambda kv: kv[1])
    best_walk = ordered_walks[0][0] if ordered_walks and ordered_walks[0][1] <= max_walk else None

    out: dict[str, Candidate] = {}
    for i, (user, sim) in enumerate(ordered_faces):
        if sim < possible_face:
            break
        clear = i == 0 and sim >= strong_face and sim - runner_up >= margin
        out[user] = Candidate(user, STRONG if clear else POSSIBLE, "face",
                              face_similarity=round(sim, 3))
    for user, dist in ordered_walks:
        if dist > max_walk:
            break
        if user in out:
            out[user].walk_distance = round(dist, 2)
            if user == best_walk:
                # Two independent kinds of evidence agreeing on the same
                # person is worth more than either alone.
                out[user].evidence = "face + walk"
                out[user].band = STRONG
        else:
            out[user] = Candidate(user, POSSIBLE, "walk", walk_distance=round(dist, 2))

    def order(c: Candidate):
        face = c.face_similarity if c.face_similarity is not None else -1.0
        walk = c.walk_distance if c.walk_distance is not None else 99.0
        return (c.band != STRONG, -face, walk)

    ranked = sorted(out.values(), key=order)
    return ranked[: int(cfg.get("identity.max_candidates", 3))]


def _face_scores(faces: Sequence[dict], gallery: Sequence[PersonGallery]) -> dict[str, float]:
    probes = [np.asarray(f["embedding"], np.float64) for f in faces if f.get("embedding")]
    if not probes:
        return {}
    probe = np.vstack(probes)
    scores = {}
    for person in gallery:
        if not person.faces:
            continue
        sims = probe @ np.vstack(person.faces).T          # (probe faces, gallery faces)
        best = sims.max(axis=1)                           # each probe's closest match
        # Upper quartile, not the maximum: one lucky frame should not decide,
        # and not the median either, since profile frames match poorly even
        # for the right person.
        scores[person.user_id] = float(np.percentile(best, 75))
    return scores


def _walk_scores(signature: Optional[dict], gallery: Sequence[PersonGallery],
                 cfg: Config) -> dict[str, float]:
    if not signature:
        return {}
    min_walks = int(cfg.get("identity.walk.min_confirmed", 2))
    gait_weight = float(cfg.get("identity.walk.gait_weight", 0.25))
    view = signature.get("view")
    probe = _vector(signature)
    same_view = {p.user_id: [w for w in p.walks if w.get("view") == view] for p in gallery}
    spread = _spread(same_view.values())

    scores = {}
    for user, walks in same_view.items():
        if len(walks) < min_walks:
            continue
        ref = {}
        for trait in BODY_TRAITS + GAIT_TRAITS:
            values = [_vector(w).get(trait) for w in walks]
            values = [v for v in values if v is not None]
            if values:
                ref[trait] = float(np.median(values))
        total = weight = 0.0
        n_body = 0
        for trait, value in probe.items():
            if value is None or trait not in ref:
                continue
            w = 1.0 if trait in BODY_TRAITS else gait_weight
            n_body += trait in BODY_TRAITS
            total += w * ((value - ref[trait]) / spread[trait]) ** 2
            weight += w
        if n_body >= 2 and weight:
            scores[user] = float(np.sqrt(total / weight))
    return scores


def _spread(walk_groups) -> dict[str, float]:
    """Within-person spread per trait: pooled from people with repeat walks."""
    spread = dict(_DEFAULT_SPREAD)
    for trait in spread:
        variances, counts = [], []
        for walks in walk_groups:
            values = [v for v in (_vector(w).get(trait) for w in walks) if v is not None]
            if len(values) >= 2:
                variances.append(np.var(values, ddof=1))
                counts.append(len(values) - 1)
        if sum(counts) >= 6:
            pooled = float(np.sqrt(np.average(variances, weights=counts)))
            # Never tighter than half the default: a few near-identical walks
            # would otherwise make every new walk look like a stranger.
            spread[trait] = max(pooled, 0.5 * _DEFAULT_SPREAD[trait])
    return spread


def _vector(signature: dict) -> dict[str, Optional[float]]:
    out = dict(signature.get("body") or {})
    out.update(signature.get("gait") or {})
    return out


def unit(vector) -> np.ndarray:
    v = np.asarray(vector, np.float64).ravel()
    return v / (np.linalg.norm(v) or 1.0)
