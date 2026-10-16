"""Entry point for the gaitscreen pilot platform.

Run with::

    streamlit run app/main.py

Design intent for a pilot: the point of putting this in front of people is to
find out where it breaks on real footage, so the interface shows its working
rather than just its answers. Every unmeasurable metric states why, every
unreliable one carries its caveat inline, and detected gait events are plotted
over the raw signal so a tester can see whether the segmentation actually locked
on -- which is the failure a plausible-looking number would otherwise hide.

The trends page is the landing view: the tool exists to follow one person over
time, so the first thing a returning user sees is where each person stands.
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

st.set_page_config(page_title="CadenceCare", page_icon="🚶", layout="wide",
                   initial_sidebar_state="auto")

from app import ui  # noqa: E402
from app.shared import get_config  # noqa: E402
from app.views import analyse, calibration, limitations, trends  # noqa: E402

ui.inject_css()

# Rendered at the top so it is visible even while an analysis blocks the page.
st.markdown(
    '<div class="sticky-footer">CadenceCare is a screening and trend-monitoring aid'
    ', not a medical device. Gait analysis uses pose estimation (MediaPipe).</div>',
    unsafe_allow_html=True,
)

PAGES = {
    "Trends": trends.render,
    "Analyse a walk": analyse.render,
    "Camera setup": calibration.render,
    "Limitations": limitations.render,
}
assert list(PAGES) == ui.PAGES

#: Widget keys whose values should survive switching pages. Streamlit forgets
#: a widget's state when a run does not draw it, which would wipe the person's
#: name every time someone glanced at the trends page mid-session.
PERSISTENT_KEYS = ("person_name", "session_date", "trends_person",
                   "video_source", "sample_key")

_THRESHOLD_LABELS = {
    "gait_speed_mps": "Speed",
    "stride_time_cv_pct": "Stride-time variability",
    "step_length_asymmetry_pct": "Step-length asymmetry",
}


def _render_thresholds(cfg) -> None:
    """Show the active flag thresholds. Illustrative only, not clinical."""
    for key, label in _THRESHOLD_LABELS.items():
        thresh = cfg.get(f"flagging.absolute.{key}", {})
        parts = []
        for name, val in thresh.items():
            direction = ">" if "above" in name else "<"
            level = "high" if "high_risk" in name else "moderate"
            parts.append(f"{direction}{val} {level}")
        if parts:
            st.sidebar.caption(f"{label}: {', '.join(parts)}")


def _keep_widget_state() -> None:
    for key in PERSISTENT_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def main() -> None:
    cfg = get_config()
    _keep_widget_state()

    locked = (st.session_state.get("analysing", False)
              or st.session_state.get("analysis_phase") in (2, 3))

    with st.sidebar:
        ui.brand()
        ui.nav_label("Menu")
        choice = st.radio(
            "Page", ui.PAGES, key=ui.NAV_KEY, label_visibility="collapsed",
            format_func=lambda page: f"{ui.PAGE_ICONS[page]}  {page}",
            disabled=locked,
        )
        if locked:
            st.caption("Navigation is paused while a video is being analysed.")

    show_technical = st.session_state.get("show_technical", False)
    PAGES[choice](cfg)

    with st.sidebar:
        st.divider()
        new_technical = st.checkbox(
            "Technical details",
            value=show_technical,
            help="Show pipeline internals: algorithm version, technical caveats, diagnostic detail.",
            disabled=locked,
        )
        if new_technical != show_technical:
            st.session_state["show_technical"] = new_technical
            st.rerun()
        if new_technical:
            st.caption("Flag thresholds (illustrative):")
            _render_thresholds(cfg)



main()
