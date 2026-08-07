"""Page: the tool's known limitations, rendered from the canonical document.

Read from ``docs/limitations.md`` rather than duplicated here, so the text a
pilot tester reads in the app cannot drift from the text in the repository.
"""
from __future__ import annotations

import streamlit as st

from app.shared import PROJECT_ROOT
from gaitscreen.config import Config

LIMITATIONS_PATH = PROJECT_ROOT / "docs" / "limitations.md"


def render(cfg: Config) -> None:  # noqa: ARG001 - uniform signature across views
    st.header("What this tool can and cannot do")
    if LIMITATIONS_PATH.exists():
        st.markdown(LIMITATIONS_PATH.read_text(encoding="utf-8"))
    else:
        st.warning(f"{LIMITATIONS_PATH} not found.")
