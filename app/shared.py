"""Paths, resources and widgets shared across the app's views."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gaitscreen.config import Config  # noqa: E402
from gaitscreen.storage.repository import SessionRepository  # noqa: E402
from gaitscreen.types import CORE_METRICS  # noqa: E402

#: Uploaded videos, kept under their original filename -- see :func:`save_upload`.
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

#: Label, unit and number format for each core metric.
METRIC_DISPLAY: dict[str, tuple[str, str, str]] = {
    "gait_speed_mps": ("Gait speed", "m/s", "{:.2f}"),
    "stride_time_cv_pct": ("Stride-time variability", "%", "{:.1f}"),
    "step_length_asymmetry_pct": ("Step-length asymmetry", "%", "{:.1f}"),
    "cadence_spm": ("Cadence", "steps/min", "{:.0f}"),
    "double_support_pct": ("Double support", "% of cycle", "{:.1f}"),
    "trunk_ap_sway_norm": ("Trunk lean excursion", "x leg length", "{:.3f}"),
}

VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]


# --------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------
@st.cache_resource
def get_config() -> Config:
    return Config.load()


def open_repository(cfg: Config) -> SessionRepository:
    """Open the session database. Callers are responsible for closing it."""
    return SessionRepository(cfg.resolve_path("storage.db_path", PROJECT_ROOT))


def save_upload(uploaded, user_id: str) -> Path:
    """Persist an upload under its original filename.

    Session IDs are derived from the filename, so keeping it stable means
    re-analysing the same video updates that session rather than creating a
    duplicate visit on the same date.
    """
    target_dir = UPLOAD_DIR / (user_id or "_unassigned")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path


# --------------------------------------------------------------------------
# widgets
# --------------------------------------------------------------------------
def disclaimer() -> None:
    st.caption(
        "Screening and trend-monitoring tool — **not a diagnostic instrument**. "
        "It is built to flag *changes* in one person's walking over time. "
        "All clinical thresholds are illustrative and must be reviewed against "
        "current geriatric literature before real use."
    )


def render_flags(flags) -> None:
    if not flags:
        st.success("No flags raised for this session.")
        return
    for flag in flags:
        confirmed = " · confirmed across sessions" if flag.confirmed else ""
        body = f"**{flag.severity.upper()}** · {flag.trigger} check{confirmed}\n\n{flag.message}"
        (st.error if flag.severity == "high" else st.warning)(body)


def render_metrics(metrics) -> None:
    """Metric cards.

    A metric that could not be measured says so and gives the reason; one that
    was measured under a caveat carries the caveat next to the value. Neither is
    relegated to a footnote, because in a screening tool an unqualified number
    reads as a normal result.
    """
    columns = st.columns(3)
    for index, key in enumerate(CORE_METRICS):
        label, unit, fmt = METRIC_DISPLAY[key]
        value = metrics.value(key)
        with columns[index % 3]:
            if value is None:
                st.metric(label, "not measurable")
                reason = metrics.unavailable.get(key)
                if reason:
                    st.caption(f"⚠️ {reason}")
            else:
                st.metric(label, f"{fmt.format(value)} {unit}".strip())
                caveat = metrics.low_confidence_metrics.get(key)
                if caveat:
                    st.caption(f"⚠️ {caveat}")
            st.write("")


def bullet_list(items: Sequence[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")
