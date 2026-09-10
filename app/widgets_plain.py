"""Widgets for the plain-language results view.

Separate from :mod:`app.shared` because these are a distinct concern: rendering
results for someone with no background in gait analysis. The technical widgets
(charts, diagnostic tables, raw notes) stay in ``shared`` and are reached through
expanders, so the two audiences do not have to read each other's version.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

#: Status chip per metric state. Symbol and colour both carry the meaning, so it
#: still reads for a colour-blind viewer or in greyscale.
STATUS_CHIP: dict[str, str] = {
    "good": ":green[**✓ Typical**]",
    "watch": ":orange[**● Keep an eye on**]",
    "attention": ":red[**▲ Worth discussing**]",
    "unmeasured": ":grey[**— Not measured**]",
}

TONE_RENDERER = {
    "good": st.success,
    "watch": st.warning,
    "attention": st.error,
    "insufficient": st.info,
}

VIDEO_CAPTION = (
    "The skeleton is what the tool actually tracked, and the circles mark the "
    "moments it decided each foot landed and lifted. If the skeleton follows the "
    "joints and the red circles appear as the heel touches down, the "
    "measurements rest on solid ground. If they do not, they do not — and this "
    "is the quickest way to tell."
)


def render_verdict(summary) -> None:
    """The single line that leads the results page."""
    renderer = TONE_RENDERER.get(summary.tone, st.info)
    renderer("### " + summary.headline + "\n\n" + summary.sub_headline)
    if summary.walk_description:
        st.caption(summary.walk_description)


def render_plain_cards(summary) -> None:
    """Metric cards written for a caregiver rather than a gait lab.

    Measured metrics come first, and the unmeasured ones are collected into
    their own section rather than interleaved: someone scanning for results
    should not have to step over six blanks to find the two numbers that exist.
    """
    measured = summary.measured_cards
    if measured:
        columns = st.columns(min(3, len(measured)))
        for index, card in enumerate(measured):
            with columns[index % len(columns)]:
                _plain_card(card)

    unmeasured = summary.unmeasured_cards
    if not unmeasured:
        return

    label = (
        f"Why {len(unmeasured)} other measure"
        f"{'s' if len(unmeasured) > 1 else ''} could not be taken"
    )
    with st.expander(label, expanded=False):
        st.caption(
            "These are missing because of what the video could show, not because "
            "of anything about the person's walking."
        )
        for card in unmeasured:
            st.markdown(f"**{card.name}** — {card.note}")
            st.caption(card.what)


def _plain_card(card) -> None:
    with st.container(border=True):
        st.markdown(STATUS_CHIP[card.status])
        st.markdown(f"##### {card.name}")
        if card.measured:
            st.markdown(f"## {card.value_text}")
            if card.everyday:
                st.caption(card.everyday)
            st.caption(f"_{card.direction}_")
        st.caption(card.what)
        if card.note:
            st.caption(f":orange[⚠ {card.note}]")


def render_annotated_video(overlay) -> None:
    """The source video with the tracking and detected events drawn on it."""
    if overlay is None:
        st.info(
            "The annotated video was not generated for this session. Turn it on "
            "in the sidebar and analyse again."
        )
        return

    path = Path(overlay["path"])
    if not path.exists():
        st.info("The annotated video is no longer available on disk.")
        return

    data = path.read_bytes()
    if overlay.get("playable", True):
        st.video(data)
    else:
        st.warning(overlay.get("note") or "This video may not play in the browser.")
    st.caption(VIDEO_CAPTION)
    st.download_button(
        "Download the annotated video", data, file_name=path.name, mime="video/mp4",
    )
