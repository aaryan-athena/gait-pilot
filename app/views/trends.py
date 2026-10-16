"""Page: a person's history, concerns and per-metric trends. The landing view.

Ordered by the questions a returning user brings to it: is anything wrong
(the verdict and the concerns), where do things stand (the latest walk against
the person's usual range), and how did it get there (the charts and the session
log). The summary itself is built in :mod:`gaitscreen.reporting.history`, from
the flags stored with each session, so this page cannot disagree with what a
session reported when it was analysed.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import ui
from app.shared import METRIC_DISPLAY, current_setup, list_people, open_repository
from gaitscreen.config import Config
from gaitscreen.flagging.baseline import compute as compute_baseline
from gaitscreen.reporting import charts
from gaitscreen.reporting.history import summarise_history
from gaitscreen.reporting.plain import PLAIN
from gaitscreen.types import CORE_METRICS

MOVEMENT_STYLE = {
    "better": ("good", "Improving"),
    "worse": ("watch", "Less favourable"),
    "steady": ("neutral", "Steady"),
}


def render(cfg: Config) -> None:
    ui.hero(
        "Walking trends",
        "How each person's walking is changing over time, and anything worth a "
        "closer look.",
    )

    repository = open_repository(cfg)
    people = list_people(repository)
    if not people:
        _first_run(cfg)
        return

    # A remembered choice can outlive the person (deleted, or a different
    # database after a redeploy); a stale value would break the picker.
    if st.session_state.get("trends_person") not in people:
        st.session_state.pop("trends_person", None)

    picker, window, _ = st.columns([1.1, 1.1, 1])
    with picker:
        user_id = st.selectbox("Person", people, key="trends_person")

    history = repository.sessions_for_user(user_id)
    if history.empty:
        ui.empty_state(f"No saved walks for {user_id} yet",
                       "Analyse a walk and save it to start this person's trend.")
        ui.nav_button("Analyse a walk", "Analyse a walk", key="trends_empty_analyse",
                      primary=True, icon=":material/directions_walk:")
        return

    history = _one_camera_angle(history)
    if len(history) > 1:
        with window:
            n_choice = st.slider(
                "Walks to include", min_value=1, max_value=len(history),
                value=min(10, len(history)),
                help="The summary, charts and usual range use only the most recent "
                     "walks selected here.",
            )
        history = history.sort_values("session_date").tail(n_choice)

    flags = {sid: repository.flags_for_session(sid) for sid in history["session_id"]}
    summary = summarise_history(history, flags, cfg)

    ui.banner(summary.headline, summary.sub_headline, tone=summary.tone)
    _overview_tiles(summary)
    _latest_walk(summary)
    _concerns(summary)
    _charts(cfg, history, flags)
    _session_log(history, flags)
    if st.session_state.get("show_technical", False):
        _technical_tables(history, flags)
        _render_recording_history(history)


# --------------------------------------------------------------------------
# first run
# --------------------------------------------------------------------------
def _first_run(cfg: Config) -> None:
    ui.empty_state(
        "Welcome to CadenceCare",
        "No walks have been saved yet. Three steps get the first trend going.",
    )
    st.write("")
    setup_done = current_setup(cfg) is not None
    columns = st.columns(3, gap="medium")
    with columns[0], st.container(border=True):
        ui.step(1, "Set up the camera", state="done" if setup_done else "current",
                hint="Measure how much floor the camera sees, so speed can be shown.")
        ui.nav_button("Camera setup", "Camera setup", key="welcome_setup",
                      primary=not setup_done, icon=":material/straighten:")
    with columns[1], st.container(border=True):
        ui.step(2, "Analyse a walk", state="current" if setup_done else "todo",
                hint="Upload a side-on video of the person walking across the frame.")
        ui.nav_button("Analyse a walk", "Analyse a walk", key="welcome_analyse",
                      primary=setup_done, icon=":material/directions_walk:")
    with columns[2], st.container(border=True):
        ui.step(3, "Save it and come back", state="todo",
                hint="Each saved walk adds to the person's trend shown on this page.")


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def _overview_tiles(summary) -> None:
    active = summary.active_concerns
    attention = sum(1 for c in active if c.severity == "attention")
    concern_tone = "attention" if attention else ("watch" if active else "good")
    concern_sub = (
        f"{attention} worth discussing" if attention
        else ("to keep an eye on" if active else "in the latest walk")
    )
    span = _span_text(summary.first_date, summary.last_date)
    ui.tiles([
        ui.tile("Walks saved", str(summary.n_sessions), sub=span, tone="accent"),
        ui.tile("Latest walk", _pretty_date(summary.last_date),
                sub=f"first on {_pretty_date(summary.first_date)}"),
        ui.tile("Concerns now", str(len(active)), sub=concern_sub, tone=concern_tone),
    ] + ([ui.tile("Lower-quality recordings", str(summary.n_low_confidence),
                  sub="left out of the usual range")]
         if summary.n_low_confidence else []))
    for note in summary.notes:
        st.caption(f":material/info: {note}")


def _latest_walk(summary) -> None:
    if not summary.latest:
        return
    ui.section("Latest walk",
               "Each measure from the most recent walk, against this person's usual "
               "range (or the previous walk until there are enough for a range).")
    items = []
    for trend in summary.latest:
        spec = PLAIN[trend.metric]
        value, _, unit = trend.value_text.partition(" ")
        if trend.movement:
            tone, word = MOVEMENT_STYLE[trend.movement]
            sub = ui.pill(word, tone) + f"<div style='margin-top:.35rem'>{ui.esc(trend.change_text)}</div>"
        else:
            tone, sub = "", ""
        if trend.metric == "gait_speed_mps":
            sub += f"<div>{ui.esc(spec['everyday'].format(kmh=trend.latest * 3.6))}</div>"
        items.append(ui.tile(trend.name, value, unit=unit, sub_html=sub,
                             tone="accent" if trend.metric == "gait_speed_mps" else tone))
    ui.tiles(items)


def _concerns(summary) -> None:
    ui.section("Concerns",
               "Raised by the screening rules on saved walks. A concern is a prompt "
               "to look more closely, not a diagnosis.")
    active = summary.active_concerns
    if not active:
        ui.banner("Nothing flagged in the latest walk",
                  "No measure crossed a screening threshold or moved outside this "
                  "person's usual range.", tone="good")
    for concern in active:
        _concern_card(concern, summary.n_sessions, muted=False)

    if summary.recording_issues:
        problems = "; ".join(f"{title} ({n} of {summary.n_sessions} walks)"
                             for title, n in summary.recording_issues)
        ui.banner("A recurring recording problem may be affecting these numbers",
                  f"{problems}. A setup problem seen in most walks is worth fixing "
                  "once, so it stops colouring the trend.", tone="info")

    earlier = summary.earlier_concerns
    if earlier:
        with st.expander(f"Earlier concerns, not seen in the latest walk ({len(earlier)})",
                         icon=":material/history:"):
            for concern in earlier:
                _concern_card(concern, summary.n_sessions, muted=True)


def _concern_card(concern, n_sessions: int, *, muted: bool) -> None:
    label = "Worth discussing" if concern.severity == "attention" else "Keep an eye on"
    badges = [ui.pill(label, concern.severity)]
    if concern.confirmed:
        badges.append(ui.pill("Confirmed across walks", "info"))
    if "trend" in concern.triggers:
        badges.append(ui.pill("Change from usual", "neutral"))
    seen = (f"Flagged in {concern.sessions_flagged} of {n_sessions} walks"
            if concern.sessions_flagged > 1 else "Flagged in 1 walk")
    when = (f"on {_pretty_date(concern.first_seen)}" if concern.first_seen == concern.last_seen
            else f"from {_pretty_date(concern.first_seen)} to {_pretty_date(concern.last_seen)}")
    ui.concern_card(concern.name, concern.message, f"{seen}, {when}.",
                    tone="muted" if muted else concern.severity, badges=badges)


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------
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
        "This person has walks filmed from two different camera angles. They "
        "measure walking in different ways and their numbers are not comparable, "
        "so only one angle is shown at a time.",
        icon=":material/videocam:",
    )
    chosen = st.segmented_control(
        "Camera angle", present, default=present[0],
        format_func=lambda k: f"{VIEW_LABELS[k]} ({int((kinds == k).sum())})",
    ) or present[0]
    return history[kinds == chosen]


def _charts(cfg: Config, history: pd.DataFrame, flags: dict) -> None:
    metrics = [m for m in CORE_METRICS
               if m in history and history[m].notna().sum() > 0]
    if not metrics:
        return
    ui.section("Over time",
               "Dashed line and shaded band: this person's usual range. Red rings "
               "mark flagged walks; hollow points are lower-quality recordings, "
               "left out of the usual range.")
    flagged = _flagged_dates(history, flags)
    tabs = st.tabs([PLAIN[m]["name"] if m in PLAIN else METRIC_DISPLAY[m][0]
                    for m in metrics])
    for tab, metric in zip(tabs, metrics):
        with tab:
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
                st.caption(f":material/hourglass_empty: {baseline.unavailable_reason}")
            st.caption(PLAIN.get(metric, {}).get("what", ""))


def _session_log(history: pd.DataFrame, flags: dict) -> None:
    ui.section("Session log", "Every saved walk in the period shown, newest first.")
    shown = [m for m in ("gait_speed_mps", "cadence_spm", "stride_time_cv_pct",
                         "step_length_asymmetry_pct", "double_support_pct")
             if m in history]
    table = pd.DataFrame({"Date": pd.to_datetime(history["session_date"]).dt.date})
    for metric in shown:
        table[PLAIN[metric]["name"]] = history[metric].values
    table["Concerns"] = [
        sum(1 for f in flags.get(sid, ()) if f.trigger != "quality")
        for sid in history["session_id"]
    ]
    if "quality_score" in history:
        table["Recording quality"] = history["quality_score"].values
    if "n_strides_valid" in history:
        table["Usable strides"] = history["n_strides_valid"].values
    table = table.iloc[::-1]

    config = {"Date": st.column_config.DateColumn(format="D MMM YYYY")}
    for metric in shown:
        spec = PLAIN[metric]
        # printf-style: a literal "%" in a unit ("% variation") must be doubled
        # or the column renders as an error.
        unit = spec["unit"].replace("%", "%%")
        config[spec["name"]] = st.column_config.NumberColumn(
            format=spec["fmt"].replace("{:", "%").replace("}", "")
            + (f" {unit}" if unit else ""),
        )
    config["Concerns"] = st.column_config.NumberColumn(format="%d")
    config["Recording quality"] = st.column_config.ProgressColumn(
        min_value=0.0, max_value=1.0, format="%.2f")
    st.dataframe(table, hide_index=True, width="stretch", column_config=config)


def _technical_tables(history: pd.DataFrame, flags: dict) -> None:
    """Raw flag and session tables, for testers. Off the default view."""
    with st.expander("All flags raised", icon=":material/flag:"):
        rows = []
        for _, row in history.iterrows():
            for flag in flags.get(row["session_id"], ()):
                rows.append({
                    "date": pd.to_datetime(row["session_date"]).date(),
                    "severity": flag.severity,
                    "trigger": flag.trigger,
                    "metric": flag.metric,
                    "confirmed": flag.confirmed,
                    "message": flag.message,
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.write("No flags raised across these walks.")

    with st.expander("Full session table (technical)", icon=":material/table:"):
        columns = ["session_date", "algo_version", *CORE_METRICS,
                   "n_strides_valid", "quality_score", "low_confidence"]
        _tech = history[[c for c in columns if c in history]].copy()
        # Round all numeric columns to 2 decimals for display
        for _col in _tech.select_dtypes(include=["float64", "float32"]).columns:
            _tech[_col] = _tech[_col].round(2)
        st.dataframe(_tech, hide_index=True, width="stretch")


def _flagged_dates(history: pd.DataFrame, flags: dict) -> dict[str, list]:
    out: dict[str, list] = {}
    for _, row in history.iterrows():
        for flag in flags.get(row["session_id"], ()):
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
        # Walks saved without feedback come back as NaN, not None, once they
        # share a column with walks that have it -- and NaN is truthy.
        if not isinstance(raw, str) or not raw:
            continue
        for entry in json.loads(raw):
            counts[entry["code"]] += 1
            titles[entry["code"]] = entry.get("title", entry["code"])

    with st.expander("Recording problems across these walks",
                     icon=":material/videocam_off:"):
        if not counts:
            st.write("No recording problems recorded for these walks.")
            return
        total = len(history)
        st.caption(
            "A flat or gappy trend can be the camera rather than the person. "
            "Anything appearing in most walks is one setup change away from "
            "being fixed for good."
        )
        st.dataframe(
            pd.DataFrame([
                {"problem": titles[code], "walks affected": f"{n} of {total}"}
                for code, n in counts.most_common()
            ]),
            hide_index=True, width="stretch",
        )


# --------------------------------------------------------------------------
def _pretty_date(value) -> str:
    if not value:
        return "—"
    stamp = pd.Timestamp(value)
    return f"{stamp.day} {stamp.strftime('%b %Y')}"


def _span_text(first, last) -> str:
    if not first or not last or first == last:
        return "one day so far"
    days = (pd.Timestamp(last) - pd.Timestamp(first)).days
    if days < 60:
        return f"over {days} days"
    return f"over {days // 30} months"
