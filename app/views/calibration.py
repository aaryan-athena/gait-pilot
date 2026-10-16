"""Page: the camera setup -- how much floor the camera's view covers.

Walking speed is the one measure that needs a real-world distance; everything
else is scale-free. The setup asks for it in the form a person can actually
measure with a tape: the length of floor the picture spans, along the line the
person walks. Analysis is locked until it is entered (see ``analyse.py``),
because speed is the measure the pilot most wants to show.

The two older ways of getting a scale -- from a walk over a known distance, and
from two marked points in a frame -- are kept as alternatives. Both now produce
the same thing: a coverage length, stored as the one camera setup.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app import ui
from app.shared import (PROJECT_ROOT, VIDEO_TYPES, current_setup, open_repository,
                        save_upload)
from gaitscreen.calibration.interactive import grab_frame
from gaitscreen.calibration.model import CalibrationError
from gaitscreen.calibration.setup import (MAX_COVERAGE_M, MIN_COVERAGE_M,
                                          save_setup)
from gaitscreen.config import Config
from gaitscreen.pipeline import extract_session

#: Top-down sketch of what to measure: camera at the bottom, its field of view
#: opening towards the walking line, a tape mark where each edge of the view
#: meets that line, and the distance between them.
DIAGRAM = """
<svg viewBox="0 0 520 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:560px;display:block;margin:auto">
<defs><marker id="ah" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#0B6E63"/></marker></defs>
<rect x="0" y="0" width="520" height="250" rx="16" fill="#F5F7F7"/>
<polygon points="260,214 70,72 450,72" fill="#0B6E63" fill-opacity="0.08"/>
<line x1="260" y1="214" x2="70" y2="72" stroke="#0B6E63" stroke-width="1.5" stroke-dasharray="5 4"/>
<line x1="260" y1="214" x2="450" y2="72" stroke="#0B6E63" stroke-width="1.5" stroke-dasharray="5 4"/>
<line x1="30" y1="72" x2="490" y2="72" stroke="#5B6B73" stroke-width="2"/>
<text x="490" y="62" text-anchor="end" font-family="Inter,sans-serif" font-size="12" fill="#5B6B73">walking line</text>
<rect x="62" y="64" width="16" height="16" rx="3" fill="#E8A33D"/>
<rect x="442" y="64" width="16" height="16" rx="3" fill="#E8A33D"/>
<line x1="72" y1="34" x2="448" y2="34" stroke="#0B6E63" stroke-width="2" marker-start="url(#ah)" marker-end="url(#ah)"/>
<rect x="196" y="20" width="128" height="26" rx="13" fill="#0B6E63"/>
<text x="260" y="38" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="700" fill="#fff">coverage length</text>
<text x="70" y="104" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#5B6B73">tape at left edge</text>
<text x="450" y="104" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#5B6B73">tape at right edge</text>
<rect x="244" y="206" width="32" height="22" rx="5" fill="#13232B"/>
<circle cx="260" cy="217" r="6" fill="#fff"/>
<text x="260" y="244" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" fill="#13232B">camera, side-on to the path</text>
</svg>
"""


def render(cfg: Config) -> None:
    ui.hero(
        "Camera setup",
        "Tell the tool how much floor the camera sees, so it can turn movement "
        "in the picture into walking speed in metres per second. Do this once "
        "for each place you set the camera up.",
    )

    setup = current_setup(cfg)
    _status(setup)
    _flash()

    ui.section("How to measure it", "Takes about two minutes with a tape measure.")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        _diagram()
    with right:
        ui.step(1, "Fix the camera", state="todo",
                hint="On a tripod or something solid, side-on to the walking path, "
                     "at the height and angle you will record from.")
        ui.step(2, "Mark the edges of the view", state="todo",
                hint="Look at the camera screen. Put a piece of tape on the floor, on "
                     "the line the person will walk along, exactly where the left "
                     "edge of the picture meets it. Do the same at the right edge.")
        ui.step(3, "Measure between the marks", state="todo",
                hint="Measure the straight-line distance between the two pieces of "
                     "tape, in metres, and enter it below.")

    _entry_form(cfg, setup)
    _frame_check(cfg, setup)
    _accuracy_notes()
    # The reference-video methods reach the same result the long way round;
    # useful when testing a setup, noise on a first look at the page.
    if st.session_state.get("show_technical", False):
        _other_methods(cfg)


# --------------------------------------------------------------------------
def _diagram() -> None:
    """Drawn as an image: inline SVG is stripped by Streamlit's HTML sanitiser."""
    import base64

    data = base64.b64encode(" ".join(DIAGRAM.split()).encode("utf-8")).decode("ascii")
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{data}" alt="Top-down sketch: camera '
        'side-on to the walking line, a tape mark where each edge of the view meets '
        'the line, and the coverage length measured between them" '
        'style="width:100%;max-width:560px;display:block;margin:auto">',
        unsafe_allow_html=True,
    )


def _status(setup) -> None:
    if setup is None:
        ui.banner(
            "No camera setup saved yet",
            "Walks cannot be analysed until the camera setup is entered below. "
            "Without it the tool cannot work out walking speed.",
            tone="watch",
        )
        return
    saved = _pretty_date(setup.created_at)
    ui.tiles([
        ui.tile("Camera view covers", f"{setup.coverage_m:.2f}", unit="m",
                sub="of floor, along the walking line", tone="accent"),
        ui.tile("Detail at 1080p", f"{setup.metres_per_pixel(1920) * 1000:.1f}",
                unit="mm / pixel", sub="smaller is finer"),
        ui.tile("Saved", saved or "—", sub="applies to every walk analysed"),
    ])


def _entry_form(cfg: Config, setup) -> None:
    ui.section("Enter the measurement")
    with st.form("coverage_form", border=True):
        columns = st.columns([1, 2])
        with columns[0]:
            coverage = st.number_input(
                "Floor length the camera sees (metres)",
                min_value=MIN_COVERAGE_M, max_value=MAX_COVERAGE_M,
                value=float(setup.coverage_m) if setup else 4.0,
                step=0.05, format="%.2f",
                help="The distance between the two tape marks at the left and right "
                     "edges of the picture, measured along the walking line.",
            )
        with columns[1]:
            note = st.text_input(
                "Where is this setup? (optional)",
                placeholder="e.g. Garage, tripod by the door, 1.2 m high",
            )
        submitted = st.form_submit_button("Save camera setup", type="primary",
                                          icon=":material/save:")
    if submitted:
        _save(cfg, coverage, note or None)


def _save(cfg: Config, coverage: float, note) -> None:
    try:
        setup = save_setup(open_repository(cfg), coverage, notes=note)
    except CalibrationError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not save the camera setup: {exc}")
        return
    # Redraw from the top so the status tiles show the setup just saved; the
    # confirmation is carried across the rerun and shown once.
    st.session_state["setup_flash"] = setup.coverage_m
    st.rerun()


def _flash() -> None:
    coverage = st.session_state.pop("setup_flash", None)
    if coverage is None:
        return
    st.toast(f"Camera setup saved: {coverage:.2f} m", icon="✅")
    with st.container(border=True):
        st.success(
            f"Saved. The camera view covers **{coverage:.2f} m** of floor. Every "
            "walk analysed from now on will include walking speed."
        )
        ui.nav_button("Continue to Analyse a walk", "Analyse a walk",
                      key="setup_to_analyse", primary=True,
                      icon=":material/arrow_forward:")


def _frame_check(cfg: Config, setup) -> None:
    """Optional: see the measured span drawn over a frame from the camera."""
    with st.expander("Check it against a picture from the camera (optional)",
                     icon=":material/photo_camera:"):
        st.caption(
            "Upload a photo or a short clip taken from the camera position. The "
            "edges of the picture are where your tape marks should be. Move the "
            "line to where the person walks to see the span you measured."
        )
        uploaded = st.file_uploader(
            "Photo or video from the camera", key="setup_frame",
            type=["jpg", "jpeg", "png", *VIDEO_TYPES],
        )
        if uploaded is None:
            return
        frame = _load_frame(uploaded)
        if frame is None:
            st.warning("Could not read a picture from that file.")
            return
        height = frame.shape[0]
        line = st.slider("Walking line position (from the top of the picture)",
                         0, height - 1, int(height * 0.85))
        coverage = setup.coverage_m if setup else None
        st.image(_draw_span(frame, line, coverage), width="stretch")


def _load_frame(uploaded):
    import cv2
    import numpy as np

    data = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith((".jpg", ".jpeg", ".png")):
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    path = save_upload(uploaded, "_setup")
    try:
        return grab_frame(path, 0)
    except Exception:  # noqa: BLE001
        return None


def _draw_span(frame, line_y: int, coverage):
    import cv2
    import numpy as np

    image = np.array(frame).copy()
    height, width = image.shape[:2]
    teal, amber = (11, 110, 99), (232, 163, 61)
    thick = max(2, width // 400)
    cv2.line(image, (0, line_y), (width - 1, line_y), teal, thick)
    for x in (thick, width - 1 - thick):
        cv2.line(image, (x, max(0, line_y - height // 12)),
                 (x, min(height - 1, line_y + height // 12)), amber, thick * 2)
    label = f"{coverage:.2f} m across" if coverage else "coverage not saved yet"
    scale = max(0.6, width / 1400)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x0, y0 = (width - tw) // 2, max(th + 12, line_y - 14)
    cv2.rectangle(image, (x0 - 10, y0 - th - 8), (x0 + tw + 10, y0 + 8), teal, -1)
    cv2.putText(image, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def _accuracy_notes() -> None:
    with st.expander("Getting an accurate speed", icon=":material/tips_and_updates:"):
        st.markdown(
            "- **Measure on the walking line.** The view is wider further from the "
            "camera, so a span measured along the back wall would make every walk "
            "look slower than it is.\n"
            "- **Keep the walker on that line.** Someone walking nearer the camera "
            "than the tape crosses more of the picture per step and reads faster; "
            "further away, slower. A strip of tape along the path helps.\n"
            "- **Re-measure if the camera moves.** A new height, angle or zoom "
            "changes the span. Save a new setup whenever the camera is set up "
            "again.\n"
            "- **The person must cross the frame.** Speed is measured from how far "
            "they travel across the picture, so it is not available when they walk "
            "towards the camera, on the spot, or while the camera follows them.\n"
            "- **It is an approximation.** A single span assumes everyone walks at "
            "the same distance from the camera, so each speed carries that caveat. "
            "Changes from one walk to the next are more reliable than the absolute "
            "number."
        )


# --------------------------------------------------------------------------
# alternatives that need a reference video
# --------------------------------------------------------------------------
def _other_methods(cfg: Config) -> None:
    with st.expander("Other ways to set it: from a reference video",
                     icon=":material/videocam:"):
        method = st.radio(
            "Method",
            ["Known walk distance", "Two marked points"],
            horizontal=True,
            help="Both work out the same coverage length from a video instead of a "
                 "tape measurement across the view.",
        )
        uploaded = st.file_uploader(
            "A video recorded from this exact camera position",
            type=VIDEO_TYPES, key="calibration_upload",
        )
        if uploaded is None:
            st.caption("Upload a reference video to use one of these methods.")
            return
        path = save_upload(uploaded, "_setup")
        if method == "Known walk distance":
            _from_walk(cfg, path)
        else:
            _from_points(cfg, path)


def _from_walk(cfg: Config, path: Path) -> None:
    st.markdown(
        "Measure the real distance the person walks in this video (tape measure "
        "along the floor, start to finish) and enter it below. The coverage is "
        "worked out from how far they travel across the picture."
    )
    distance_m = st.number_input(
        "Distance actually walked (metres)",
        min_value=0.5, max_value=50.0, value=4.0, step=0.5,
    )
    if not st.button("Work out the setup from this walk", type="primary"):
        return

    with st.spinner("Measuring travel across the frame…"):
        extraction = extract_session(
            path, cfg, project_root=PROJECT_ROOT, check_camera_motion=True
        )
    if not extraction.speed.feasible:
        st.error("This video cannot be used: " + (extraction.speed.reason or ""))
        return

    travel_px = extraction.speed.subject_translation_px
    coverage = distance_m * extraction.info.width / travel_px
    _save(cfg, coverage, f"from a {distance_m:g} m walk over {travel_px:.0f} px")


def _from_points(cfg: Config, path: Path) -> None:
    st.markdown(
        "Mark two points a known distance apart. Place them **along the walking "
        "path**, not across it, and as far apart as possible. A short reference "
        "turns a small marking error into a large scale error."
    )
    frame = grab_frame(path, 0)
    height, width = frame.shape[:2]

    left, right = st.columns(2)
    with left:
        x0 = st.slider("Point 1: x", 0, width - 1, int(width * 0.15))
        y0 = st.slider("Point 1: y", 0, height - 1, int(height * 0.85))
    with right:
        x1 = st.slider("Point 2: x", 0, width - 1, int(width * 0.85))
        y1 = st.slider("Point 2: y", 0, height - 1, int(height * 0.85))

    st.image(_annotate(frame, (x0, y0), (x1, y1)),
             caption="Reference points", width="stretch")
    distance_m = st.number_input(
        "Real distance between the two points (metres)",
        min_value=0.1, max_value=50.0, value=2.0, step=0.1,
    )
    if not st.button("Work out the setup from these points", type="primary"):
        return

    pixels = float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    if pixels < 1:
        st.error("The two points are on top of each other; move them apart.")
        return
    if pixels < 0.3 * width:
        st.warning(
            f"The points span only {pixels / width:.0%} of the picture, so a small "
            "marking error becomes a large speed error. Points further apart are "
            "better."
        )
    _save(cfg, distance_m * width / pixels,
          f"from two points {distance_m:g} m apart, {pixels:.0f} px")


def _annotate(frame, p0, p1):
    import cv2
    import numpy as np

    image = np.array(frame).copy()
    cv2.line(image, tuple(map(int, p0)), tuple(map(int, p1)), (11, 110, 99), 2)
    for point, tag in ((p0, "1"), (p1, "2")):
        centre = tuple(map(int, point))
        cv2.circle(image, centre, 9, (11, 110, 99), 2)
        cv2.putText(image, tag, (centre[0] + 12, centre[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (11, 110, 99), 2)
    return image


def _pretty_date(value) -> str:
    import pandas as pd

    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "—"
    return f"{stamp.day} {stamp.strftime('%b %Y')}"
