"""Page: analyse a single walking video.

The results are laid out for two different readers, in the order they need
things. A caregiver gets a plain verdict, six plainly-named measures, and the
annotated video that shows the tool working. A clinician or tester gets the
same session's exact metric names, event plots and diagnostic numbers, one
expander down. Neither has to read the other's version to find their own.
"""
from __future__ import annotations

import traceback
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from app import ui
from app.samples import (Sample, available_samples, cached_extraction,
                         cached_overlay, get_sample, sample_setup)
from app.shared import (PROJECT_ROOT, VIDEO_TYPES, bullet_list, current_setup,
                        open_repository, render_flags, render_metrics,
                        render_recording_feedback,
                        render_recording_measurements, safe_user_id, save_upload)
from app.widgets_plain import (render_annotated_video, render_plain_cards,
                               render_verdict)
from gaitscreen.calibration.model import CalibrationError
from gaitscreen.calibration.setup import MAX_COVERAGE_M, MIN_COVERAGE_M, save_setup
from gaitscreen.config import Config
from gaitscreen.io import video as video_io
from gaitscreen.pipeline import analyse_extraction, analyse_video, persist_session
from gaitscreen.reporting import charts
from gaitscreen.reporting.overlay import ffmpeg_available, render_overlay_video
from gaitscreen.reporting.plain import _plain_unmeasured as plain_unmeasured
from gaitscreen.reporting.plain import summarise
from gaitscreen.segmentation.events import independent_cadence_spm
from gaitscreen.storage.artifacts import SessionArtifacts
from gaitscreen.version import ALGO_NOTES, ALGO_VERSION

def _local_today() -> date:
    """Best-effort local date for the recording default. Never raises.

    Server runs UTC; without adjustment, evening US users get "tomorrow".
    Shifts back 7h (PDT) as a pragmatic fix. Wrapped so no browser or
    environment quirk can break the date picker.
    """
    try:
        return (datetime.now(timezone.utc) - timedelta(hours=7)).date()
    except Exception:
        try:
            return date.today()
        except Exception:
            return date(1970, 1, 1)  # sentinel: should never happen


RECORDING_GUIDANCE = """
**In order of how much it matters.** These come from what actually went wrong on
real pilot recordings, not from theory.

1. **Two to three passes back and forth, in one recording.** This is the most
   common reason a session cannot be measured. There is a hard trade-off: framed
   well, a person filling half the frame covers most of it in two or three
   strides, so a single walk-past yields two to five strides where ten are
   needed for a variability figure. You cannot fix this by zooming out. That
   just makes the subject too small. Walk them up and back, two or three times,
   in the same clip.
2. **Fill at least half the frame height with the person.** Foot position can
   only be located to within a pixel or two, so how big they are in frame sets
   the precision of everything. Head to floor, with a little room to spare.
3. **Legs and ankles visible.** Loose or flowing trousers hide the knee and
   ankle, and the tracker then *infers* where they are rather than seeing them.
   Producing confident, wrong numbers. Fitted trousers, leggings, shorts, or
   loose trousers rolled up.
4. **Fixed camera, square to the walking path.** On a tripod or propped against
   something solid. Never handheld, and never following the person. Stand level
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
   cane. It detects bodies, not objects.

After analysing, the **How to improve the recording** section tells you which of
these applied to your clip and what to change.
"""


def render(cfg: Config) -> None:
    ui.hero(
        "Analyse a walk",
        "Upload a side-on video, or try a sample, to see walking speed, rhythm "
        "and evenness, with the tracking drawn over the video.",
    )

    # Lock the inputs during analysis: uploading mid-run orphans the progress
    # bar. The user waits for completion, then uploads.
    locked = (st.session_state.get("analysing", False)
              or st.session_state.get("analysis_phase") in (2, 3))
    setup = current_setup(cfg)

    sample_mode = _source_picker(locked) == SOURCE_SAMPLE
    if not sample_mode:
        _camera_step(cfg, setup, locked)
    settings, uploaded = _walk_step(setup, locked, sample_mode)

    # If the chosen video differs from the one that was analysed, the
    # displayed results are stale. Clear them immediately.
    uploaded_key = _source_key(uploaded)
    analyzed_key = st.session_state.get("_analyzed_file_key")
    if uploaded_key is not None and st.session_state.get("result") is not None:
        if analyzed_key is not None and uploaded_key != analyzed_key:
            st.session_state["result"] = None
            st.session_state["overlay"] = None

    # Phase 2: render cleared UI (no stale results), then trigger phase 3.
    if st.session_state.get("analysis_phase") == 2:
        st.session_state["analysis_phase"] = 3
        st.empty()
        st.info("Starting analysis…")
        # Brief pause so the frontend renders the cleared state before
        # Phase 3 blocks for 4 minutes. Without this, the two reruns batch
        # and the old results stay visible (greyed out).
        import time
        time.sleep(2)
        st.rerun()
        return

    # Phase 3: start the blocking analysis. Frontend already shows cleared UI.
    if st.session_state.get("analysis_phase") == 3:
        st.session_state["analysis_phase"] = None
        if uploaded is not None:
            _run(cfg, uploaded, settings, setup)
        st.rerun()
        return

    _err = st.session_state.get("last_error")
    if _err:
        st.error("The last analysis failed.")
        with st.expander("Technical detail (from the failed run)"):
            st.code(_err)
        if st.button("Dismiss error"):
            st.session_state.pop("last_error", None)
            st.rerun()

    if st.session_state.get("analysing", False):
        pct = st.session_state.get("progress_pct", 0.0)
        txt = st.session_state.get("progress_text", "Processing…")
        st.progress(pct, text=txt)
        st.caption("Analysis in progress. Results will appear here when done.")
        return

    existing = st.session_state.get("result")
    if existing is not None and uploaded is None:
        _render_result(cfg, existing, settings)
        if st.button("Analyse a new video", type="secondary",
                     icon=":material/restart_alt:"):
            st.session_state["result"] = None
            st.session_state["overlay"] = None
            st.rerun()
        return

    if uploaded is None:
        return

    if st.button("Analyse this walk", type="primary", icon=":material/play_arrow:",
                 disabled=setup is None and not sample_mode):
        # Clear old results immediately so the pane doesn't show stale metrics
        # while the new analysis runs.
        st.session_state["result"] = None
        st.session_state["overlay"] = None
        st.session_state["analysis_phase"] = 2
        st.rerun()

    if st.session_state.get("result") is not None:
        _render_result(cfg, st.session_state["result"], settings)


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------
def _camera_step(cfg: Config, setup, locked: bool) -> None:
    """Step 1: the camera setup has to exist before anything is analysed.

    Walking speed is the measure the pilot most wants to show, and it is the
    only one that needs a real distance. Analysing without it would quietly
    produce a result missing its headline number, so the upload stays closed
    until the setup is in. It can be entered right here, so the gate costs one
    field rather than a trip to another page.
    """
    with st.container(border=True):
        if setup is not None:
            top = st.columns([4, 1], vertical_alignment="center")
            with top[0]:
                ui.step(1, "Camera setup", state="done",
                        hint=f"The camera view covers {setup.coverage_m:.2f} m of floor "
                             "along the walking line, so walking speed will be shown.")
            with top[1]:
                ui.nav_button("Change", "Camera setup", key="analyse_change_setup",
                              icon=":material/edit:", disabled=locked)
            return

        ui.step(1, "Set up the camera first", state="current",
                hint="Enter how much floor the camera sees, measured along the line "
                     "the person walks: tape marks at the left and right edges of the "
                     "picture, and the distance between them.")
        row = st.columns([1.2, 1, 1.4], vertical_alignment="bottom")
        with row[0]:
            coverage = st.number_input(
                "Floor length the camera sees (metres)", min_value=MIN_COVERAGE_M,
                max_value=MAX_COVERAGE_M, value=4.0, step=0.05, format="%.2f",
                disabled=locked, key="quick_coverage",
            )
        with row[1]:
            if st.button("Save setup", type="primary", icon=":material/save:",
                         disabled=locked, key="quick_setup_save"):
                try:
                    save_setup(open_repository(cfg), coverage)
                    st.toast(f"Camera setup saved: {coverage:.2f} m", icon="✅")
                    st.rerun()
                except CalibrationError as exc:
                    st.error(str(exc))
        with row[2]:
            ui.nav_button("How to measure it", "Camera setup", key="analyse_setup_help",
                          icon=":material/help:", disabled=locked)


SOURCE_UPLOAD = "Your own video"
SOURCE_SAMPLE = "Try a sample"


def _source_picker(locked: bool) -> str:
    """Upload a video, or try a built-in sample without recording first."""
    if not available_samples():
        return SOURCE_UPLOAD
    st.session_state.setdefault("video_source", SOURCE_UPLOAD)
    choice = st.segmented_control(
        "Video", [SOURCE_UPLOAD, SOURCE_SAMPLE], key="video_source",
        label_visibility="collapsed", disabled=locked,
    )
    return choice or SOURCE_UPLOAD


def _sample_picker(locked: bool) -> Sample:
    samples = available_samples()
    st.session_state.setdefault("sample_key", samples[0].key)
    left, right = st.columns([1, 1.15], gap="medium")
    with left:
        key = st.radio(
            "Sample walk", [x.key for x in samples], key="sample_key",
            format_func=lambda k: get_sample(k).title,
            captions=[x.description for x in samples], disabled=locked,
        )
    sample = get_sample(key)
    with right:
        # A still, not the video: the clips are 20-30 MB, and streaming one just
        # to help choose between them makes the page slow to settle.
        if sample.poster.exists():
            st.image(str(sample.poster), width="stretch")
    return sample


def _source_key(source) -> str | None:
    """Identifies the chosen video, so a change can clear stale results.

    Name and size for an upload (Streamlit's file id is not stable across
    re-uploads of the same file); the sample's key for a sample.
    """
    if source is None:
        return None
    if isinstance(source, Sample):
        return f"sample:{source.key}"
    return f"{source.name}_{source.size}"


def _person_and_date(locked: bool):
    details = st.columns(2)
    with details[0]:
        person = st.text_input(
            "Person name", key="person_name", placeholder="e.g. Bob",
            disabled=locked,
            help="Used to group this person's walks into a trend.",
        )
    with details[1]:
        # Seeded through session state rather than value=: the key is kept
        # alive across pages (see main.py), and Streamlit warns when a widget
        # gets both.
        st.session_state.setdefault("session_date", _local_today())
        when = st.date_input("Date of recording", key="session_date",
                             disabled=locked)
    return person, when


def _walk_step(setup, locked: bool, sample_mode: bool) -> tuple[dict, object]:
    """Who is walking, when, and the video itself (an upload or a sample)."""
    ready = setup is not None or sample_mode
    with st.container(border=True):
        if sample_mode:
            ui.step(1, "Pick a sample walk", state="current",
                    hint="Filmed in a garage at a set pace. The name is only needed "
                         "if you want to save the result to a trend.")
        else:
            ui.step(2, "Choose the walk", state="current" if ready else "todo",
                    hint="Name the person so the walk can be added to their trend.")
        if sample_mode:
            # The sample is the point here; who and when only matter for saving.
            uploaded = _sample_picker(locked)
            person, when = _person_and_date(locked)
        else:
            person, when = _person_and_date(locked)
            uploaded = st.file_uploader(
                "Upload a side-on video (person walking across the frame)",
                type=VIDEO_TYPES, disabled=locked or not ready,
            )
            if locked:
                st.caption("Upload is disabled during analysis and will be "
                           "re-enabled when it finishes.")
            elif uploaded is None:
                st.info("Upload a video to begin." if ready else
                        "Upload a video to begin once the camera setup above is "
                        "saved.", icon=":material/upload:")

        camera_check, make_video = True, True
        if technical_mode():
            with st.expander("Processing options", icon=":material/tune:"):
                camera_check = st.checkbox(
                    "Check for camera movement", value=True, disabled=locked,
                    help="Tracks background features to detect a panning camera. "
                         "Slower, but a camera that follows the subject makes speed "
                         "meaningless.",
                )
                make_video = st.checkbox(
                    "Create the annotated video", value=True, disabled=locked,
                    help="Replays the clip with the tracked skeleton and detected "
                         "steps drawn on, so you can see what the tool saw.",
                )
                if make_video and not ffmpeg_available():
                    st.caption(":orange[ffmpeg is not installed, so the annotated "
                               "video may not play in the browser.]")
        if not sample_mode:
            with st.expander("How to record a walk", icon=":material/videocam:"):
                st.markdown(RECORDING_GUIDANCE)

    settings = {
        "user_id": (person or "").strip(),
        "session_date": when,
        # Assistive device hidden for pilot (all unaided walks).
        "device": "none",
        # Notes not used in the pilot; kept because _save() expects them.
        "notes": "",
        "camera_check": camera_check,
        "make_video": make_video,
    }
    return settings, uploaded


def _run(cfg: Config, uploaded, settings: dict, setup) -> None:
    st.session_state["analysing"] = True
    st.session_state["progress_pct"] = 0.0
    st.session_state["progress_text"] = "Starting…"
    try:
        _run_inner(cfg, uploaded, settings, setup)
    finally:
        st.session_state["analysing"] = False
        st.session_state.pop("progress_pct", None)
        st.session_state.pop("progress_text", None)


def _run_inner(cfg: Config, uploaded, settings: dict, setup) -> None:
    if isinstance(uploaded, Sample):
        _run_sample(cfg, uploaded, settings)
        return
    path = save_upload(uploaded, settings["user_id"])
    session_date = settings["session_date"].isoformat()
    progress = st.progress(0.0, text="Tracking the person in the video…")
    repository = open_repository(cfg)

    try:
        # The camera setup is a length across the whole frame, so it is turned
        # into a calibration at this video's own resolution.
        calibration = None
        if setup is not None:
            info = video_io.probe(path, cfg)
            calibration = setup.calibration_for(info.width, info.height)
        history = repository.sessions_for_user(
            safe_user_id(settings["user_id"]), before_date=session_date
        )

        # Frame count is not reliably known before reading, so the bar runs off
        # an estimate from the file size and is simply clamped.
        estimated_frames = max(1, int(getattr(uploaded, "size", 0) / 20000))

        def on_frame(index: int) -> None:
            pct = min(0.9, index / estimated_frames)
            txt = f"Tracking the person in the video… frame {index}"
            st.session_state["progress_pct"] = pct
            st.session_state["progress_text"] = txt
            progress.progress(pct, text=txt)

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
        st.session_state["result_saved"] = False
        # Remember which file was analyzed, so a new upload clears stale results.
        st.session_state["_analyzed_file_key"] = _source_key(uploaded)
        st.session_state["overlay"] = (
            _render_overlay(cfg, result, progress) if settings["make_video"] else None
        )
        progress.progress(1.0, text="Done")
    except Exception as exc:  # noqa: BLE001 - surface the error, don't kill the app
        progress.empty()
        detail = traceback.format_exc()
        print("ANALYSIS FAILED\n" + detail, flush=True)
        st.session_state["last_error"] = detail
        st.error(f"Analysis failed: {exc}")
        with st.expander("Technical detail"):
            st.code(detail)
        st.session_state["result"] = None
        st.session_state["overlay"] = None


def _run_sample(cfg: Config, sample: Sample, settings: dict) -> None:
    """Analyse a built-in sample: tracking cached, flags worked out fresh."""
    progress = st.progress(0.0, text="Tracking the person in the video…")
    session_date = settings["session_date"].isoformat()
    try:
        setup = sample_setup(cfg)
        info = video_io.probe(sample.path, cfg)
        calibration = setup.calibration_for(info.width, info.height)

        def on_frame(index: int) -> None:
            pct = min(0.9, index / max(1, info.n_frames))
            txt = f"Tracking the person in the video… frame {index}"
            st.session_state["progress_pct"] = pct
            st.session_state["progress_text"] = txt
            progress.progress(pct, text=txt)

        extraction = cached_extraction(
            sample.key, settings["camera_check"], ALGO_VERSION, cfg, calibration,
            on_frame,
        )
        history = open_repository(cfg).sessions_for_user(
            safe_user_id(settings["user_id"]), before_date=session_date
        ) if settings["user_id"] else None
        result = analyse_extraction(
            extraction, cfg, user_id=settings["user_id"], session_date=session_date,
            history=history,
        )
        st.session_state["result"] = result
        st.session_state["result_saved"] = False
        st.session_state["_analyzed_file_key"] = _source_key(sample)

        overlay = None
        if settings["make_video"]:
            progress.progress(0.92, text="Drawing the tracking onto the video…")
            try:
                overlay = cached_overlay(sample.key, settings["camera_check"],
                                         ALGO_VERSION, cfg, result)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not create the annotated video: {exc}")
        st.session_state["overlay"] = overlay
        progress.progress(1.0, text="Done")
    except Exception as exc:  # noqa: BLE001 - surface the error, don't kill the app
        progress.empty()
        detail = traceback.format_exc()
        print("SAMPLE ANALYSIS FAILED\n" + detail, flush=True)
        st.session_state["last_error"] = detail
        st.error(f"Analysis failed: {exc}")
        st.session_state["result"] = None
        st.session_state["overlay"] = None


def _render_overlay(cfg: Config, result, progress) -> dict | None:
    """Render the annotated playback, without letting a failure lose the analysis.

    The video is a presentation aid; the measurements are the product. A missing
    encoder should cost the viewer the replay, not the session.
    """
    # Update session_state so a rerun during overlay shows current progress,
    # not the stale "frame 775" from the tracking phase.
    st.session_state["progress_pct"] = 0.92
    st.session_state["progress_text"] = "Drawing the tracking onto the video…"
    progress.progress(0.92, text="Drawing the tracking onto the video…")

    n_frames = result.extraction.series.n_frames
    def on_overlay_frame(decoded: int) -> None:
        pct = 0.92 + 0.08 * min(1.0, decoded / max(1, n_frames))
        txt = f"Drawing the tracking onto the video… frame {decoded}/{n_frames}"
        st.session_state["progress_pct"] = pct
        st.session_state["progress_text"] = txt
        progress.progress(pct, text=txt)

    artifacts = SessionArtifacts(
        cfg.resolve_path("storage.artifacts_root", PROJECT_ROOT),
        result.user_id, result.session_id,
    )
    artifacts.ensure()
    try:
        overlay = render_overlay_video(
            result.extraction, result.analysis, artifacts.overlay_path, cfg,
            max_width=int(cfg["reporting.overlay.max_width"]),
            progress=on_overlay_frame,
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
def _render_result(cfg: Config, result, settings: dict) -> None:
    summary = summarise(
        result, cfg, include_technical=st.session_state.get("show_technical", False))

    ui.section("Results", getattr(result.extraction.info.path, "name", ""))
    _headline_tiles(result)
    _save_panel(cfg, result, settings)

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

    ui.section("What the walk looked like")
    render_plain_cards(summary)

    ui.section("See it for yourself",
               "The tracked skeleton and detected steps, drawn over the video.")
    render_annotated_video(st.session_state.get("overlay"))

    if not result.diagnostics.is_clean:
        ui.section("How to improve the recording")
        render_recording_feedback(result.diagnostics, expanded=False)

    if technical_mode():
        _render_technical(cfg, result)


def _headline_tiles(result) -> None:
    """Speed first: it is the measure the camera setup exists to produce."""
    metrics = result.metrics
    speed = metrics.gait_speed_mps
    if speed is not None:
        speed_tile = ui.tile("Walking speed", f"{speed:.2f}", unit="m/s",
                             sub=f"About {speed * 3.6:.1f} km/h", tone="accent")
    else:
        reason = metrics.unavailable.get("gait_speed_mps")
        speed_tile = ui.tile("Walking speed", "Not measured",
                             sub=plain_unmeasured(reason), tone="watch")
    cadence = metrics.cadence_spm
    quality = result.quality
    ui.tiles([
        speed_tile,
        ui.tile("Steps per minute", f"{cadence:.0f}" if cadence else "—",
                sub="how often a step is taken"),
        ui.tile("Recording quality", f"{quality.score:.0%}",
                sub=("lower quality: left out of the usual range"
                     if quality.low_confidence else "good enough to trend"),
                tone="watch" if quality.low_confidence else "good"),
    ])


def _save_panel(cfg: Config, result, settings: dict) -> None:
    """Adding the walk to the person's trend is a deliberate act, not automatic:
    a pilot user analyses plenty of test clips that should never enter anyone's
    history."""
    person = settings["user_id"]
    with st.container(border=True):
        cols = st.columns([3, 1.3], vertical_alignment="center")
        if st.session_state.get("result_saved"):
            with cols[0]:
                st.markdown(f":material/check_circle: **Saved to {result.user_id}'s "
                            "trend.**")
            with cols[1]:
                ui.nav_button("Open trends", "Trends", key="result_to_trends",
                              icon=":material/monitoring:")
            return
        with cols[0]:
            if person:
                st.markdown(f"**Add this walk to {person}'s trend?** Only saved walks "
                            "count towards their usual range.")
            else:
                st.markdown("**Add this walk to a trend?** Enter the person's name "
                            "above first.")
        with cols[1]:
            if st.button("Save to trend", type="primary", icon=":material/bookmark_add:",
                         disabled=not person, key="save_to_trend"):
                result.user_id = safe_user_id(person)
                _save(cfg, result, settings["notes"])
                st.session_state["result_saved"] = True
                st.rerun()


def technical_mode() -> bool:
    return bool(st.session_state.get("show_technical", False))


def _is_coronal(result) -> bool:
    view = getattr(getattr(result, "analysis", None), "view", None)
    return view is not None and view.kind == "coronal"


def _render_technical(cfg: Config, result) -> None:
    """Everything a clinician or tester needs, kept out of the plain view."""
    analysis = result.analysis

    with st.expander("Technical detail: exact measures, event plots, diagnostics"):
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

    st.caption(f"Algorithm version {ALGO_VERSION}: {ALGO_NOTES}")


def _save(cfg: Config, result, notes: str) -> None:
    repository = open_repository(cfg)
    try:
        # A session_id is person + date + file name, and the insert upserts.
        # So re-saving one clip on one day corrects that row instead of adding
        # a point. Counting either side of the save is the only way to tell
        # "updated" from "silently lost".
        n_before = len(repository.sessions_for_user(result.user_id))
        session_id = persist_session(
            result, cfg, repository, project_root=PROJECT_ROOT, notes=notes or None
        )
        n_after = len(repository.sessions_for_user(result.user_id))
        added = n_after > n_before
        print(f"SAVED session={session_id} user={result.user_id} "
              f"{'added' if added else 'updated'} total={n_after}", flush=True)
        st.success(
            f"{'Saved' if added else 'Updated'} session `{session_id}` for "
            f"**{result.user_id}**, who now has {n_after} saved "
            f"session{'s' if n_after != 1 else ''}."
        )
        if not added:
            st.info(
                "This updated an existing session rather than adding one. A "
                "session is identified by person, recording date and video "
                "file name, so re-analysing the same clip corrects its data "
                "point instead of duplicating the walk. To add a point, use a "
                "different clip or change the recording date."
            )
    except Exception as exc:  # noqa: BLE001
        detail = traceback.format_exc()
        print("SAVE FAILED\n" + detail, flush=True)
        st.session_state["last_error"] = detail
        st.error(f"Could not save: {exc}")
        with st.expander("Technical detail"):
            st.code(detail)
