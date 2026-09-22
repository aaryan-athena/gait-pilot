"""Page: a person's history and per-metric trends."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.shared import METRIC_DISPLAY, disclaimer, open_repository
from gaitscreen.config import Config
from gaitscreen.flagging.baseline import compute as compute_baseline
from gaitscreen.reporting import charts
from gaitscreen.types import CORE_METRICS


def render(cfg: Config) -> None:
    st.header("Trends over time")
    disclaimer()

    repository = open_repository(cfg)
    try:
        users = repository.list_users()
        if not users:
            st.info("No sessions saved yet. Analyse a walk and save it first.")
            return

        user_id = st.selectbox("Person", users)
        history = repository.sessions_for_user(user_id)
        if history.empty:
            st.info(f"No saved sessions for {user_id}.")
            return

        _render_summary(repository, user_id)
        history = _one_camera_angle(history)
        flagged = _flagged_dates(repository, history)
        _render_trends(cfg, history, flagged)
        _render_tables(repository, history)
    finally:
        repository.close()


#: How each stored view kind is described to someone reading their own trends.
VIEW_LABELS = {
    "sagittal": "Filmed from the side",
    "oblique": "Filmed from the side",
    "indeterminate": "Filmed from the side",
    "coronal": "Filmed towards the person",
}


def _one_camera_angle(history: pd.DataFrame) -> pd.DataFrame:
    """Never draw two camera angles on the same trend line.

    A side-on and a towards-camera recording of the same walk on the same day
    give different cadences, because they measure it by different means. Drawn
    together they make a step in the chart that looks exactly like the thing
    this page exists to detect. The baseline already refuses to mix them; the
    chart must not either.
    """
    if "view_kind" not in history or history.empty:
        return history

    # Sessions stored before the view was recorded were all analysed by the
    # sagittal path, because it was the only one.
    kinds = history["view_kind"].fillna("sagittal").map(
        lambda k: "coronal" if k == "coronal" else "sagittal"
    )
    present = list(dict.fromkeys(kinds))
    if len(present) < 2:
        return history

    st.info(
        "This person has sessions filmed from two different camera angles. "
        "They measure walking in different ways and their numbers are not "
        "comparable, so only one angle is shown at a time -- plotting both "
        "together would show a step change that is the camera moving, not the "
        "person."
    )
    chosen = st.radio(
        "Camera angle", present, horizontal=True,
        format_func=lambda k: f"{VIEW_LABELS[k]} ({int((kinds == k).sum())} sessions)",
    )
    return history[kinds == chosen]


def _render_summary(repository, user_id: str) -> None:
    summary = repository.summary(user_id)
    columns = st.columns(4)
    columns[0].metric("Sessions", summary["n_sessions"])
    columns[1].metric("Low confidence", summary["n_low_confidence"])
    columns[2].metric(
        "Date range",
        f"{summary['date_range'][0]} → {summary['date_range'][1]}"
        if summary["date_range"] else "—",
    )
    columns[3].metric("Algorithm versions", len(summary["algo_versions"]))

    if len(summary["algo_versions"]) > 1:
        st.warning(
            f"This history spans {len(summary['algo_versions'])} algorithm versions "
            f"({', '.join(summary['algo_versions'])}). Changing the algorithm changes "
            "the measurement, so part of any apparent trend may be the tool rather "
            "than the person. Reprocess the history under a single version before "
            "reading it."
        )


def _render_trends(cfg: Config, history: pd.DataFrame, flagged: dict) -> None:
    st.subheader("Per-metric trends")
    st.caption(
        "Dashed line and shaded band are this person's own baseline (robust "
        "centre ± spread). Red rings mark flagged sessions; hollow points are "
        "low-confidence recordings, which are excluded from the baseline."
    )

    for metric in CORE_METRICS:
        if history[metric].notna().sum() == 0:
            continue
        label, unit, _ = METRIC_DISPLAY[metric]
        baseline = compute_baseline(history, metric, cfg)
        st.pyplot(
            charts.trend_figure(
                history, metric, f"{label} ({unit})",
                flagged_dates=flagged.get(metric, []),
                baseline_centre=baseline.centre if baseline.available else None,
                baseline_spread=baseline.spread if baseline.available else None,
            ),
            clear_figure=True,
        )
        if not baseline.available:
            st.caption(f"⚠️ {baseline.unavailable_reason}")


def _render_tables(repository, history: pd.DataFrame) -> None:
    with st.expander("Session table"):
        columns = [
            "session_date", "algo_version", *CORE_METRICS,
            "n_strides_valid", "quality_score", "low_confidence",
        ]
        st.dataframe(
            history[[c for c in columns if c in history]],
            hide_index=True, width="stretch",
        )

    _render_recording_history(history)

    with st.expander("All flags raised"):
        rows = []
        for _, row in history.iterrows():
            for flag in repository.flags_for_session(row["session_id"]):
                rows.append({
                    "date": row["session_date"].date(),
                    "severity": flag.severity,
                    "trigger": flag.trigger,
                    "metric": flag.metric,
                    "confirmed": flag.confirmed,
                    "message": flag.message,
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         width="stretch")
        else:
            st.write("No flags raised across this person's sessions.")


def _flagged_dates(repository, history: pd.DataFrame) -> dict[str, list]:
    out: dict[str, list] = {}
    for _, row in history.iterrows():
        for flag in repository.flags_for_session(row["session_id"]):
            out.setdefault(flag.metric, []).append(row["session_date"])
    return out


def _render_recording_history(history: pd.DataFrame) -> None:
    """Recording problems across the history.

    A trend that looks flat, or that has gaps in it, may be the camera rather
    than the person. Showing the recurring recording problems alongside the
    trends is what lets someone tell those apart -- and a problem that appears in
    most sessions is a setup to fix once, not a run of bad luck.
    """
    import collections
    import json

    counts: collections.Counter = collections.Counter()
    titles: dict[str, str] = {}
    for raw in history.get("recording_diagnostics_json", []):
        if not raw:
            continue
        for entry in json.loads(raw):
            counts[entry["code"]] += 1
            titles[entry["code"]] = entry.get("title", entry["code"])

    with st.expander("Recording problems across these sessions"):
        if not counts:
            st.write("No recording problems recorded for these sessions.")
            return
        total = len(history)
        st.caption(
            "A flat or gappy trend can be the camera rather than the person. "
            "Anything appearing in most sessions is one setup change away from "
            "being fixed for good."
        )
        st.dataframe(
            pd.DataFrame([
                {"problem": titles[code], "sessions affected": f"{n} of {total}"}
                for code, n in counts.most_common()
            ]),
            hide_index=True, width="stretch",
        )
