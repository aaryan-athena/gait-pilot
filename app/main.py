"""Entry point for the gaitscreen pilot platform.

Run with::

    streamlit run app/main.py

Design intent for a pilot: the point of putting this in front of people is to
find out where it breaks on real footage, so the interface shows its working
rather than just its answers. Every unmeasurable metric states why, every
unreliable one carries its caveat inline, and detected gait events are plotted
over the raw signal so a tester can see whether the segmentation actually locked
on -- which is the failure a plausible-looking number would otherwise hide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit executes this file as a script, so the project root is not on the
# path and `import app.…` would fail without this.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

st.set_page_config(page_title="Gait screening pilot", page_icon="🚶", layout="wide")

from app.shared import get_config  # noqa: E402
from app.views import analyse, calibration, limitations, trends  # noqa: E402
from gaitscreen.version import ALGO_VERSION  # noqa: E402

PAGES = {
    "Analyse a walk": analyse.render,
    "Trends": trends.render,
    "Calibration": calibration.render,
    "Limitations": limitations.render,
}


def main() -> None:
    cfg = get_config()

    st.sidebar.title("🚶 Gait screening")
    st.sidebar.caption(f"pilot build · algorithm {ALGO_VERSION}")
    choice = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()

    PAGES[choice](cfg)


main()
