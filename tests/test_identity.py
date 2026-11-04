"""Identification: suggestions only, consent before faces, and nothing lost on deletion."""
from pathlib import Path

import numpy as np
import pytest

from gaitscreen.identity import PersonGallery, rank, signature_from_result, unit
from gaitscreen.identity.matcher import POSSIBLE, STRONG
from gaitscreen.pipeline import analyse_extraction
from gaitscreen.storage.repository import SessionRepository
from fixtures.extraction import build_extraction
from test_storage import _record

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "app" / "samples" / "vishal_normal_95spm.mp4"


def _face(seed: int, noise: float = 0.0, base=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base if base is not None else rng.normal(size=128)
    # Noise relative to a unit vector: 0.3 leaves a cosine similarity near 0.95.
    return unit(v + noise * rng.normal(size=128) / np.sqrt(128))


def _walk(view="sagittal", **body):
    traits = {"torso_to_leg": 0.71, "thigh_to_shank": 0.93, "upper_arm_to_forearm": 1.18,
              "head_to_torso": 0.43, "shoulder_to_hip_width": None}
    traits.update(body)
    return {"version": 1, "view": view, "body": traits,
            "gait": {"cadence_spm": 100.0, "gait_speed_mps": None}}


ALICE, BOB = _face(1), _face(2)


def _probe_faces(base, n=4, noise=0.3):
    return [{"embedding": list(_face(10 + i, noise, base))} for i in range(n)]


def test_clear_face_match_is_strong(cfg):
    gallery = [PersonGallery("alice", faces=[ALICE]), PersonGallery("bob", faces=[BOB])]
    ranked = rank(None, _probe_faces(ALICE), gallery, cfg)
    assert ranked[0].user_id == "alice" and ranked[0].band == STRONG
    assert all(c.user_id != "bob" for c in ranked), "a stranger's face is not suggested"


def test_no_evidence_means_not_sure(cfg):
    gallery = [PersonGallery("alice", faces=[ALICE])]
    assert rank(None, _probe_faces(_face(99)), gallery, cfg) == []
    assert rank(None, [], [], cfg) == []


def test_walk_alone_is_never_strong(cfg):
    gallery = [PersonGallery("alice", walks=[_walk(), _walk(torso_to_leg=0.715)])]
    ranked = rank(_walk(torso_to_leg=0.712), [], gallery, cfg)
    assert [(c.user_id, c.band, c.evidence) for c in ranked] == [("alice", POSSIBLE, "walk")]


def test_face_and_walk_agreeing_is_strong(cfg):
    # Two people with similar faces: the face alone is ambiguous...
    near = _face(3, 0.15, ALICE)
    gallery = [PersonGallery("alice", faces=[ALICE], walks=[_walk(), _walk()]),
               PersonGallery("carol", faces=[near],
                             walks=[_walk(thigh_to_shank=1.05, torso_to_leg=0.78)] * 2)]
    probe = [{"embedding": list(_face(4, 0.25, ALICE))}] * 3
    ranked = rank(_walk(), probe, gallery, cfg)
    # ...but the body agrees with only one of them.
    assert ranked[0].user_id == "alice" and ranked[0].band == STRONG
    assert ranked[0].evidence == "face + walk"


def test_walk_matching_needs_confirmed_walks_from_the_same_view(cfg):
    one_walk = [PersonGallery("alice", walks=[_walk()])]
    assert rank(_walk(), [], one_walk, cfg) == []
    side_on_only = [PersonGallery("alice", walks=[_walk(), _walk()])]
    assert rank(_walk(view="coronal"), [], side_on_only, cfg) == []


def test_closest_body_ranks_first(cfg):
    gallery = [PersonGallery("alice", walks=[_walk()] * 2),
               PersonGallery("bob", walks=[_walk(thigh_to_shank=0.99)] * 2)]
    assert rank(_walk(thigh_to_shank=0.985), [], gallery, cfg)[0].user_id == "bob"


def test_signature_of_an_analysed_walk(cfg):
    extraction, _ = build_extraction(cfg)
    result = analyse_extraction(extraction, cfg, user_id="u")
    sig = signature_from_result(result, cfg)
    assert sig["view"] == result.analysis.view.kind
    assert sig["body"]["thigh_to_shank"] and np.isfinite(sig["body"]["thigh_to_shank"])
    assert sig["gait"]["cadence_spm"] == pytest.approx(result.analysis.metrics.cadence_spm, abs=1e-3)
    assert sig["body"]["shoulder_to_hip_width"] is None, "not measurable side-on"


# -- storage -------------------------------------------------------------------
@pytest.fixture()
def repository(tmp_path):
    with SessionRepository(tmp_path / "gait.db") as repo:
        yield repo


def test_faces_need_recorded_consent(repository):
    repository.ensure_user("alice")
    with pytest.raises(PermissionError):
        repository.add_face_enrolment("alice", list(ALICE))
    repository.record_face_consent("alice", "daughter, by phone")
    repository.add_face_enrolment("alice", list(ALICE), source="photo.jpg")
    people = repository.people().set_index("user_id")
    assert people.loc["alice", "n_faces"] == 1
    assert people.loc["alice", "face_consent_by"] == "daughter, by phone"


def test_walk_faces_are_dropped_without_consent(repository):
    sid = repository.upsert_session(_record(user="bob"))
    repository.save_signature(sid, "bob", _walk(), [{"embedding": list(BOB)}])
    (bob,) = repository.gallery()
    assert bob.faces == [] and len(bob.walks) == 1


def test_gallery_and_deleting_face_data_keeps_walks(repository):
    sid = repository.upsert_session(_record(user="alice"))
    repository.record_face_consent("alice", "self")
    repository.add_face_enrolment("alice", list(ALICE))
    repository.save_signature(sid, "alice", _walk(), [{"embedding": list(ALICE)}])
    (alice,) = repository.gallery()
    assert len(alice.faces) == 2 and len(alice.walks) == 1

    repository.delete_biometrics("alice")
    (alice,) = repository.gallery()
    assert alice.faces == [] and len(alice.walks) == 1
    assert not repository.has_face_consent("alice")
    assert len(repository.sessions_for_user("alice")) == 1, "the trend is untouched"


def test_backfill_lists_sessions_without_a_signature(repository):
    a = repository.upsert_session(_record(user="alice", video="a.mp4"))
    b = repository.upsert_session(_record(user="alice", video="b.mp4"))
    repository.save_signature(a, "alice", _walk())
    assert list(repository.sessions_without_signature()["session_id"]) == [b]


def test_an_older_database_gains_the_new_tables(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE users (user_id TEXT PRIMARY KEY, display_name TEXT, "
                "created_at TEXT NOT NULL, notes TEXT)")
    old.execute("INSERT INTO users VALUES ('alice', NULL, '2026-01-01', NULL)")
    old.commit()
    old.close()
    with SessionRepository(path) as repo:
        repo.record_face_consent("alice", "self")
        repo.add_face_enrolment("alice", list(ALICE))
        assert repo.people().iloc[0]["n_faces"] == 1


# -- faces (needs the face models, downloaded on first use) ----------------------
@pytest.fixture(scope="module")
def face_models():
    from gaitscreen.config import Config
    from gaitscreen.identity import load_models

    cfg = Config.load()
    if not cfg.resolve_path("identity.face.recogniser_path", REPO).exists():
        pytest.skip("face models not downloaded")
    return load_models(cfg, REPO)


def _sample_face_photo():
    import cv2

    cap = cv2.VideoCapture(str(SAMPLE))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 134)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        pytest.skip("sample video not available")
    # The walker's head at this moment, enlarged like a portrait photo.
    crop = frame[120:340, 40:240]
    return cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def test_enrolment_photo_needs_exactly_one_face(face_models, cfg):
    from gaitscreen.identity import PhotoError, embed_photo

    photo = _sample_face_photo()
    item = embed_photo(photo, face_models, cfg)
    assert len(item["embedding"]) == 128
    with pytest.raises(PhotoError, match="no face"):
        embed_photo(np.full((400, 400, 3), 128, np.uint8), face_models, cfg)
    with pytest.raises(PhotoError, match="2 faces"):
        embed_photo(np.hstack([photo, photo]), face_models, cfg)
