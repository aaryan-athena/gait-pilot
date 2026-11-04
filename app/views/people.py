"""Page: the people the tool can recognise, and the consent behind it.

Recognition only ever *suggests* who walked; a person confirms before a walk
joins anyone's trend (see the analyse page). It learns from two things:

* **Face photos**, enrolled here, only for people whose consent is recorded.
  A face embedding is biometric data, so only the embedding is kept (never
  the photo), and it can be deleted in one step without touching the walks.
* **Confirmed walks.** Every walk saved to a person teaches the tool how
  they are built and how they walk. No consent step: these are the gait
  measurements the tool already keeps.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from app import identify, ui
from app.shared import PROJECT_ROOT, list_people, open_repository, safe_user_id
from gaitscreen.calibration.setup import is_reserved
from gaitscreen.config import Config
from gaitscreen.identity import PhotoError, embed_photo, load_models

NEW = "Add a new person…"


def render(cfg: Config) -> None:
    ui.hero(
        "People",
        "Who the tool can recognise, from a face photo or from walks already "
        "saved to them. It only ever suggests a name; you confirm it.",
    )
    repository = open_repository(cfg)

    ui.section("Everyone")
    people = repository.people()
    people = people[~people["user_id"].map(is_reserved)] if not people.empty else people
    if people.empty:
        ui.empty_state("No one yet", "Add a person below, or save a walk to a name.")
    else:
        st.dataframe(
            people.assign(
                face=lambda f: np.where(f["n_faces"] > 0, f["n_faces"].astype(str) + " photo(s)",
                                        "none"),
                consent=lambda f: f["face_consent_by"].fillna("not recorded"),
            )[["user_id", "face", "consent", "n_walks"]].rename(columns={
                "user_id": "Person", "face": "Face enrolled", "consent": "Face consent from",
                "n_walks": "Walks it has learned from",
            }),
            hide_index=True, width="stretch",
        )
        st.caption("A person is recognised by walk once two of their side-on (or two "
                   "facing-camera) walks are saved. A face can be recognised from the "
                   "first walk, once a photo is enrolled.")

    ui.section("Enrol a face", "Optional, and only with consent.")
    _enrol(cfg, repository)

    ui.section("Learn from walks saved before this feature")
    _backfill(cfg, repository)

    ui.section("Delete face data")
    _delete(repository)


def _enrol(cfg: Config, repository) -> None:
    with st.container(border=True):
        names = list_people(repository)
        left, right = st.columns(2)
        with left:
            chosen = st.selectbox("Person", names + [NEW], key="enrol_person")
            new_name = st.text_input("Name", key="enrol_new_name",
                                     disabled=chosen != NEW)
        person = safe_user_id(new_name) if chosen == NEW and new_name.strip() else (
            chosen if chosen != NEW else "")
        consented = bool(person) and repository.has_face_consent(person)
        with right:
            if consented:
                st.markdown(":material/verified_user: Consent to face recognition is "
                            "recorded for this person.")
                given_by = None
            else:
                given_by = st.text_input(
                    "Consent given by", key="enrol_consent_by",
                    placeholder="e.g. the person themselves, or their daughter",
                    help="Who agreed to this person's face being used to recognise "
                         "them. Recorded with the date.")
                st.checkbox("I have their consent to use face recognition",
                            key="enrol_consent")
        photos = st.file_uploader(
            "Photos of the face (1 to 5): clear, well lit, only this person in shot",
            type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="enrol_photos")
        can_enrol = bool(person and photos) and (
            consented or (st.session_state.get("enrol_consent") and (given_by or "").strip()))
        if st.button("Enrol", type="primary", icon=":material/face:", disabled=not can_enrol):
            if not consented:
                repository.record_face_consent(person, given_by.strip())
            _enrol_photos(cfg, repository, person, photos[:5])


def _enrol_photos(cfg: Config, repository, person: str, photos) -> None:
    import cv2

    models = load_models(cfg, PROJECT_ROOT)
    added = 0
    for photo in photos:
        image = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error(f"{photo.name}: not an image this tool can read.")
            continue
        try:
            item = embed_photo(image, models, cfg)
        except PhotoError as exc:
            st.error(f"{photo.name}: {exc}.")
            continue
        repository.add_face_enrolment(person, item["embedding"], source=photo.name)
        added += 1
    if added:
        st.success(f"Enrolled {added} photo{'s' if added != 1 else ''} for **{person}**. "
                   "The photos themselves were not kept.")


def _backfill(cfg: Config, repository) -> None:
    with st.container(border=True):
        todo = len(repository.sessions_without_signature())
        if not todo:
            st.markdown("Every saved walk has been learned from.")
            return
        st.markdown(f"**{todo} saved walk{'s' if todo != 1 else ''}** can teach the tool "
                    "how their person is built and walks. This uses the tracking kept "
                    "with each walk; no video is needed. Faces cannot be learned this way.")
        if st.button("Learn from saved walks", icon=":material/school:"):
            bar = st.progress(0.0)
            done, skipped = identify.learn_from_saved_walks(
                cfg, progress=lambda i, n: bar.progress(i / max(1, n)))
            bar.empty()
            st.success(f"Learned from {done} walk{'s' if done != 1 else ''}."
                       + (f" {skipped} had no stored tracking and were skipped."
                          if skipped else ""))


def _delete(repository) -> None:
    with st.container(border=True):
        enrolled = [p for p in list_people(repository) if repository.has_face_consent(p)]
        if not enrolled:
            st.markdown("No one has face data stored.")
            return
        st.markdown("Removes the person's enrolled faces, the faces kept from their "
                    "walks, and their consent record. Their walks and trends stay.")
        cols = st.columns([2, 2, 1.3], vertical_alignment="bottom")
        with cols[0]:
            person = st.selectbox("Person", enrolled, key="delete_face_person")
        with cols[1]:
            sure = st.checkbox("Yes, delete this person's face data", key="delete_face_sure")
        with cols[2]:
            if st.button("Delete", icon=":material/delete:", disabled=not sure):
                repository.delete_biometrics(person)
                st.success(f"Face data for **{person}** deleted.")
