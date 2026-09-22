"""Page: analyse a single walking video.

The results are laid out for two different readers, in the order they need
things. A caregiver gets a plain verdict, six plainly-named measures, and the
annotated video that shows the tool working. A clinician or tester gets the
same session's exact metric names, event plots and diagnostic numbers, one
expander down. Neither has to read the other's version to find their own.
"""
from __future__ import annotations

import traceback
from datetime import date

import pandas as pd
import streamlit as st

from app.shared import (PROJECT_ROOT, VIDEO_TYPES, bullet_list, disclaimer,
                        open_repository, render_flags, render_metrics,
                        render_recording_feedback,
                        render_recording_measurements, save_upload)
from app.widgets_plain import (render_annotated_video, render_plain_cards,
                               render_verdict)
from gaitscreen.config import Config
from gaitscreen.pipeline import analyse_video, persist_session
from gaitscreen.reporting import charts
from gaitscreen.reporting.overlay import ffmpeg_available, render_overlay_video
from gaitscreen.reporting.plain import summarise
from gaitscreen.segmentation.events import independent_cadence_spm
from gaitscreen.storage.artifacts import SessionArtifacts
from gaitscreen.version import ALGO_NOTES, ALGO_VERSION

RECORDING_GUIDANCE = """
**In order of how much it matters.** These come from what actually went wrong on
real pilot recordings, not from theory.

1. **Two to three passes back and forth, in one recording.** This is the most
   common reason a session cannot be measured. There is a hard trade-off: framed
   well, a person filling half the frame covers most of it in two or three
   strides, so a single walk-past yields two to five strides where ten are
   needed for a variability figure. You cannot fix this by zooming out — that
   just makes the subject too small. Walk them up and back, two or three times,
   in the same clip.
2. **Fill at least half the frame height with the person.** Foot position can
   only be located to within a pixel or two, so how big they are in frame sets
   the precision of everything. Head to floor, with a little room to spare.
3. **Legs and ankles visible.** Loose or flowing trousers hide the knee and
   ankle, and the tracker then *infers* where they are rather than seeing them —
   producing confident, wrong numbers. Fitted trousers, leggings, shorts, or
   loose trousers rolled up.
4. **Fixed camera, square to the walking path.** On a tripod or propped against
   something solid — never handheld, and never following the person. Stand level
   with the middle of the path so they cross the frame sideways rather than
   moving towards or away from you.
5. **Whole walk inside the frame.** Start recording once the person is already
   fully in shot and walking, and stop only after they have finished. Feet must
   stay above the bottom edge throughout.
6. **60 fps.** At 25–30 fps the gap between frames is as large as the
   stride-to-stride variation being measured, which degrades the single most
   fall-risk-predictive metric to indicative only.
7. **Only the walker in shot.** Bystanders in the background can make the
   tracker jump to a different person mid-recording.
8. **Record assistive-device use in the sidebar.** Pose estimation cannot see a
   cane — it detects bodies, not objects.

After analysing, the **How to improve the recording** section tells you which of
these applied to your clip and what to change.
"""


def render(cfg: Config) -> None:
    st.header("Analyse a walk")
    disclaimer()

    settings = _sidebar()
    uploaded = st.file_uploader(
        "Upload a side-on (sagittal) walking video", type=VIDEO_TYPES
    )

    with st.expander("Recording guidance — read before your first pilot recording"):
        st.markdown(RECORDING_GUIDANCE)

    if uploaded is None:
        st.info("Upload a video to begin.")
        return

    if st.button("Analyse", type="primary"):
        _run(cfg, uploaded, settings)

    if st.session_state.get("result") is not None:
        _render_result(cfg, st.session_state["result"], settings["notes"])


def _sidebar() -> dict:
    with st.sidebar:
        st.subheader("Session details")
        settings = {
            "user_id": st.text_input(
                "Person ID", value="pilot01",
                help="Used to group sessions into a trend.",
            ),
            "session_date": st.date_input("Date of recording", value=date.today()),
            "device": st.selectbox(
                "Assistive device used",
                ["none", "cane", "walking stick", "walker / frame", "other"],
                help=(
                    "Pose estimation cannot see a cane — it detects bodies, not "
                    "objects. This entry is the authoritative record."
                ),
            ),
            "notes": st.text_area("Notes (optional)", height=70),
        }

        st.subheader("Processing options")
        settings["use_calibration"] = st.checkbox(
            "Use this person's saved calibration", value=True,
            help="Needed for walking speed in m/s. Everything else works without it.",
        )
        settings["camera_check"] = st.checkbox(
            "Check for camera movement", value=True,
            help=(
                "Tracks background features to detect a panning camera. Slower, "
                "but a camera that follows the subject makes speed meaningless."
            ),
        )
        settings["make_video"] = st.checkbox(
            "Create the annotated video", value=True,
            help=(
                "Replays the clip with the tracked skeleton and detected steps "
                "drawn on, so you can see what the tool saw. Adds a few seconds."
            ),
        )
        if settings["make_video"] and not ffmpeg_available():
            st.caption(
                ":orange[ffmpeg is not installed, so the annotated video may not "
                "play in the browser. It will still be downloadable.]"
            )
    return settings


def _run(cfg: Config, uploaded, settings: dict) -> None:
    path = save_upload(uploaded, settings["user_id"])
    session_date = settings["session_date"].isoformat()
    progress = st.progress(0.0, text="Tracking the person in the video…")
    repository = open_repository(cfg)

    try:
        calibration = (
            repository.active_calibration(settings["user_id"])
            if settings["use_calibration"] else None
        )
        history = repository.sessions_for_user(
            settings["user_id"], before_date=session_date
        )

        # Frame count is not reliably known before reading, so the bar runs off
        # an estimate from the file size and is simply clamped.
        estimated_frames = max(1, int(getattr(uploaded, "size", 0) / 20000))

        def on_frame(index: int) -> None:
            progress.progress(
                min(0.9, index / estimated_frames),
                text=f"Tracking the person in the video… frame {index}",
            )

        result = analyse_video(
            path, cfg,
            user_id=settings["user_id"],
            session_date=session_date,
            calibration=calibration,
            declared_device=None if settings["device"] == "none" else settings["device"],
            history=history,
            project_root=PROJECT_ROOT,
            progress=on_frame,
            check_camera_motion=settings["camera_check"],
        )
        st.session_state["result"] = result
        st.session_state["overlay"] = (
            _render_overlay(cfg, result, progress) if settings["make_video"] else None
        )
        progress.progress(1.0, text="Done")
    except Exception as exc:  # noqa: BLE001 - surface the error, don't kill the app
        progress.empty()
        st.error(f"Analysis failed: {exc}")
        with st.expander("Technical detail"):
            st.code(traceback.format_exc())
        st.session_state["result"] = None
        st.session_state["overlay"] = None
    finally:
        repository.close()


def _render_overlay(cfg: Config, result, progress) -> dict | None:
    """Render the annotated playback, without letting a failure lose the analysis.

    The video is a presentation aid; the measurements are the product. A missing
    encoder should cost the viewer the replay, not the session.
    """
    progress.progress(0.92, text="Drawing the tracking onto the video…")
    artifacts = SessionArtifacts(
        cfg.resolve_path("storage.artifacts_root", PROJECT_ROOT),
        result.user_id, result.session_id,
    )
    artifacts.ensure()
    try:
        overlay = render_overlay_video(
            result.extraction, result.analysis, artifacts.overlay_path, cfg,
            max_width=int(cfg["reporting.overlay.max_width"]),
        )
        return {
            "path": str(overlay.path),
            "playable": overlay.browser_playable,
            "note": overlay.note,
        }
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not create the annotated video: {exc}")
        return None


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
def _render_result(cfg: Config, result, notes: str) -> None:
    st.divider()
    summary = summarise(result, cfg)

    if _is_coronal(result):
        # Said before the numbers, not after. Someone who filmed from the wrong
        # place will otherwise read a short page of results as a complete one,
        # and the missing measures are the ones the tool is mostly for.
        st.warning(
            "**This video was filmed towards the person, not from the side.**\n\n"
            "Walking speed, step length, step-length asymmetry and time on both "
            "feet cannot be worked out from this angle, so they are not shown. "
            "Two measures a side-on video *cannot* give are shown instead: how "
            "wide they walk, and how much they sway side to side.\n\n"
            "These numbers are kept separate from side-on recordings when "
            "tracking change over time, because the two are not comparable. "
            "For a full result, film again from the side of the walking path."
        )

    render_verdict(summary)

    st.subheader("What the walk looked like")
    render_plain_cards(summary)

    st.subheader("See it for yourself")
    render_annotated_video(st.session_state.get("overlay"))

    st.subheader("How to improve the recording")
    render_recording_feedback(result.diagnostics, expanded=False)

    _render_technical(cfg, result)

    st.divider()
    if st.button("Save this session to the trend history", type="primary"):
        _save(cfg, result, notes)


def _is_coronal(result) -> bool:
    view = getattr(getattr(result, "analysis", None), "view", None)
    return view is not None and view.kind == "coronal"


def _render_technical(cfg: Config, result) -> None:
    """Everything a clinician or tester needs, kept out of the plain view."""
    analysis = result.analysis

    with st.expander("Technical detail — exact measures, event plots, diagnostics"):
        st.markdown("**Clinical flags**")
        st.caption(
            "The same findings as above, in the wording and thresholds the "
            "flagging rules actually use."
        )
        render_flags(result.flags.flags)

        st.markdown("---")
        st.markdown("**Metrics, as named in the pipeline and stored data**")
        render_metrics(result.metrics, coronal=_is_coronal(result))

        st.markdown("---")
        st.markdown("**Recording quality**")
        columns = st.columns(2)
        columns[0].metric(
            "Quality score", f"{result.quality.score:.2f}",
            delta="low confidence" if result.quality.low_confidence else "usable",
            delta_color="inverse" if result.quality.low_confidence else "normal",
        )
        columns[1].metric(
            "Usable strides",
            f"{analysis.cycle_summary.get('n_valid', 0)} of "
            f"{analysis.cycle_summary.get('n_total', 0)}",
        )

        st.markdown("---")
        st.markdown("**Did event detection lock on?**")
        st.caption(
            "Heel strikes (filled triangles) should sit on the peaks and toe-offs "
            "(hollow) in the troughs, with the two legs alternating."
        )
        if analysis.events:
            st.pyplot(
                charts.event_overlay_figure(result.extraction.series, analysis),
                clear_figure=True,
            )
            st.pyplot(charts.stride_time_figure(analysis), clear_figure=True)
        else:
            st.warning("No gait events were detected in this recording.")

        _render_angles(analysis)
        _render_diagnostics(cfg, result)


def _render_angles(analysis) -> None:
    st.markdown("---")
    st.markdown("**Joint angle curves**")
    st.caption(
        "Averaged over valid cycles; the shaded band is one standard deviation. "
        "These are 2D projections from one camera: out-of-plane rotation and "
        "occlusion of the far limb both add error, and ankle angle is the least "
        "reliable of the three."
    )
    st.pyplot(charts.angle_curves_figure(analysis.angles), clear_figure=True)
    if analysis.range_of_motion:
        st.dataframe(
            pd.DataFrame([
                {"joint": k.replace("_", " "), "range of motion (deg)": round(v, 1)}
                for k, v in analysis.range_of_motion.items()
            ]),
            hide_index=True, width="stretch",
        )


def _render_diagnostics(cfg: Config, result) -> None:
    analysis, extraction = result.analysis, result.extraction
    info = extraction.info

    st.markdown("---")
    st.markdown("**Caveats raised during analysis**")
    if result.all_notes:
        bullet_list(result.all_notes)
    else:
        st.write("None.")

    st.markdown("**Recording measurements**")
    render_recording_measurements(result.diagnostics)

    left, right = st.columns(2)
    with left:
        st.markdown("**Recording**")
        st.write({
            "resolution": f"{info.width}x{info.height}",
            "frame rate": f"{info.fps:g} fps ({1000 / info.fps:.0f} ms/frame)",
            "duration": f"{info.duration_s:.1f} s",
            "pose detected": f"{extraction.raw.detection_rate:.1%} of frames",
            "subject height": f"{extraction.subject_px_height:.0f} px",
            "subject travel": (
                f"{extraction.speed.subject_translation_frac:.0%} of frame width"
            ),
        })
    with right:
        st.markdown("**Segmentation**")
        reference = independent_cadence_spm(extraction.series, cfg)
        metrics = analysis.metrics
        st.write({
            "direction resolved from": analysis.direction_method,
            "side nearer camera": analysis.camera_side or "not determined",
            "strides valid / detected": (
                f"{analysis.cycle_summary.get('n_valid', 0)} / "
                f"{analysis.cycle_summary.get('n_total', 0)}"
            ),
            "cadence (pipeline)": (
                f"{metrics.cadence_spm:.0f} steps/min" if metrics.cadence_spm else "n/a"
            ),
            "cadence (independent check)": (
                f"{reference:.0f} steps/min" if reference else "n/a"
            ),
            "mean stride time": (
                f"{metrics.stride_time_mean_s:.2f} s"
                if metrics.stride_time_mean_s else "n/a"
            ),
        })
        st.caption(
            "The independent cadence check counts leg-split events and shares no "
            "code with the main detector. Close agreement is good evidence the "
            "segmentation locked onto the real rhythm."
        )

    exclusions = analysis.cycle_summary.get("exclusions")
    if exclusions:
        st.markdown("**Why strides were excluded**")
        st.dataframe(
            pd.DataFrame([
                {"reason": k, "strides": v} for k, v in exclusions.items()
            ]),
            hide_index=True, width="stretch",
        )

    st.caption(f"Algorithm version {ALGO_VERSION} — {ALGO_NOTES}")


def _save(cfg: Config, result, notes: str) -> None:
    repository = open_repository(cfg)
    try:
        session_id = persist_session(
            result, cfg, repository, project_root=PROJECT_ROOT, notes=notes or None
        )
        st.success(
            f"Saved as session `{session_id}` for **{result.user_id}**. "
            "Open the Trends page to see it in context."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not save: {exc}")
    finally:
        repository.close()
