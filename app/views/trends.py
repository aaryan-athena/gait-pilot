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
        flagged = _flagged_dates(repository, history)
        _render_trends(cfg, history, flagged)
        _render_tables(repository, history)
    finally:
        repository.close()


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
            hide_index=True, use_container_width=True,
        )

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
                         use_container_width=True)
        else:
            st.write("No flags raised across this person's sessions.")


def _flagged_dates(repository, history: pd.DataFrame) -> dict[str, list]:
    out: dict[str, list] = {}
    for _, row in history.iterrows():
        for flag in repository.flags_for_session(row["session_id"]):
            out.setdefault(flag.metric, []).append(row["session_date"])
    return out
