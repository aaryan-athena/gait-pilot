"""Page: one-time pixel-to-metre calibration per person and camera setup."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.shared import PROJECT_ROOT, VIDEO_TYPES, open_repository, save_upload
from gaitscreen.calibration.interactive import grab_frame
from gaitscreen.calibration.model import Calibration
from gaitscreen.config import Config
from gaitscreen.pipeline import extract_session
from gaitscreen.pose.to_pixels import subject_pixel_height


def render(cfg: Config) -> None:
    st.header("Calibration")
    st.caption(
        "Only needed for **gait speed in metres per second**. Every other metric "
        "— variability, cadence, asymmetry, double support, trunk lean — is "
        "scale-free and works without it."
    )

    user_id = st.text_input("Person ID", value="pilot01")
    method = st.radio(
        "Method",
        ["Known walk distance (easiest)", "Two marked points (more precise)"],
        help=(
            "Both give a single pixels-to-metres scale. Neither corrects "
            "perspective across the depth of the scene; a 4-point floor "
            "homography does, and is available from the command line."
        ),
    )

    uploaded = st.file_uploader(
        "A video recorded from this exact camera position",
        type=VIDEO_TYPES, key="calibration_upload",
    )
    if uploaded is None:
        st.info("Upload a reference video to calibrate.")
        _show_existing(cfg, user_id)
        return

    path = save_upload(uploaded, user_id)
    if method.startswith("Known walk"):
        _from_walk(cfg, user_id, path)
    else:
        _from_points(cfg, user_id, path)

    _show_existing(cfg, user_id)


def _from_walk(cfg: Config, user_id: str, path: Path) -> None:
    st.markdown(
        "Measure the real distance the person walks in this video (tape measure "
        "along the floor, start to finish) and enter it below. The scale is "
        "derived from how far they travel across the frame."
    )
    distance_m = st.number_input(
        "Distance actually walked (metres)",
        min_value=0.5, max_value=50.0, value=4.0, step=0.5,
    )
    if not st.button("Calibrate from this walk", type="primary"):
        return

    with st.spinner("Measuring travel across the frame…"):
        extraction = extract_session(
            path, cfg, project_root=PROJECT_ROOT, check_camera_motion=True
        )

    if not extraction.speed.feasible:
        st.error(
            "This video cannot be used for distance calibration: "
            + (extraction.speed.reason or "")
        )
        return

    travel_px = extraction.speed.subject_translation_px
    calibration = Calibration.from_two_points(
        user_id, (0.0, 0.0), (travel_px, 0.0), distance_m,
        extraction.info.width, extraction.info.height,
        notes="derived from a known walk distance rather than a marked reference",
    )
    calibration.expected_subject_px_height = subject_pixel_height(extraction.series)
    _save(cfg, calibration, travel_px=travel_px)


def _from_points(cfg: Config, user_id: str, path: Path) -> None:
    st.markdown(
        "Mark two points a known distance apart. Place them **along the walking "
        "path**, not across it, and as far apart as possible — a short reference "
        "turns a small marking error into a large scale error."
    )

    frame = grab_frame(path, 0)
    height, width = frame.shape[:2]

    left, right = st.columns(2)
    with left:
        x0 = st.slider("Point 1 — x", 0, width - 1, int(width * 0.15))
        y0 = st.slider("Point 1 — y", 0, height - 1, int(height * 0.85))
    with right:
        x1 = st.slider("Point 2 — x", 0, width - 1, int(width * 0.85))
        y1 = st.slider("Point 2 — y", 0, height - 1, int(height * 0.85))

    st.image(_annotate(frame, (x0, y0), (x1, y1)),
             caption="Reference points", use_container_width=True)

    distance_m = st.number_input(
        "Real distance between the two points (metres)",
        min_value=0.1, max_value=50.0, value=2.0, step=0.1,
    )
    if not st.button("Save calibration", type="primary"):
        return

    try:
        calibration = Calibration.from_two_points(
            user_id, (x0, y0), (x1, y1), distance_m, width, height
        )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    with st.spinner("Recording a subject-height reference for camera-drift checks…"):
        try:
            extraction = extract_session(
                path, cfg, project_root=PROJECT_ROOT, check_camera_motion=False
            )
            calibration.expected_subject_px_height = subject_pixel_height(
                extraction.series
            )
        except Exception:  # noqa: BLE001 - the calibration is usable without it
            st.warning(
                "Could not measure the subject's height in this video, so camera "
                "movement between sessions will not be detectable."
            )

    _save(cfg, calibration)


def _annotate(frame, p0, p1):
    import cv2
    import numpy as np

    image = np.array(frame).copy()
    cv2.line(image, tuple(map(int, p0)), tuple(map(int, p1)), (255, 60, 60), 2)
    for point, tag in ((p0, "1"), (p1, "2")):
        centre = tuple(map(int, point))
        cv2.circle(image, centre, 9, (255, 60, 60), 2)
        cv2.putText(image, tag, (centre[0] + 12, centre[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 60, 60), 2)
    return image


def _save(cfg: Config, calibration: Calibration, travel_px: float | None = None) -> None:
    repository = open_repository(cfg)
    try:
        repository.save_calibration(calibration)
    finally:
        repository.close()

    st.success(f"Calibration `{calibration.calibration_id}` saved.")
    details = {
        "method": calibration.method,
        "scale": f"{calibration.scale_m_per_px * 1000:.2f} mm per pixel",
    }
    if travel_px:
        details["measured travel"] = f"{travel_px:.0f} px"
    if calibration.expected_subject_px_height:
        details["subject height reference"] = (
            f"{calibration.expected_subject_px_height:.0f} px"
        )
    st.write(details)
    if calibration.notes:
        st.warning(calibration.notes)


def _show_existing(cfg: Config, user_id: str) -> None:
    repository = open_repository(cfg)
    try:
        existing = repository.active_calibration(user_id)
    finally:
        repository.close()

    st.subheader("Current calibration")
    if existing is None:
        st.info(
            f"No calibration saved for **{user_id}**. Gait speed in m/s will be "
            "reported as not measurable; everything else is unaffected."
        )
        return
    st.write({
        "id": existing.calibration_id,
        "created": existing.created_at,
        "method": existing.method,
        "frame size": f"{existing.frame_width}x{existing.frame_height}",
        "scale": (
            f"{existing.scale_m_per_px * 1000:.2f} mm/px"
            if existing.scale_m_per_px else "homography"
        ),
    })
