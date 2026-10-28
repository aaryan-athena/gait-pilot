"""Can this device do the analysis itself, or should it send walks to a server?

Answered by measurement on the device, because nothing else answers it
reliably: pose-estimation speed depends on the CPU, the model size, the
inference library's build for that CPU, and how hot the board runs.

Three things are measured:

1. **Recording.** The Pi 5 has no hardware H.264 encoder, so video is
   compressed on the CPU, live, while the camera runs. If the CPU cannot keep
   up at the camera's frame rate, frames are lost before analysis even starts.
2. **Pose estimation**, per model size. This is nearly all of the analysis
   cost. A walk does not need to be analysed in real time -- it is recorded
   first and analysed afterwards -- but a device that takes an hour per walk
   cannot keep up with a care home's day.
3. **Agreement** (optional, ``--accuracy``). A smaller model is only an option
   if it measures the same walk the same way. Each variant runs the full
   pipeline on the same clip and is compared with the heavy model.

The result is written to ``bench.json`` in the storage root, and
``processing.mode: auto`` reads it to decide where walks are analysed.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .settings import MODEL_VARIANTS, Settings, model_overrides

#: A typical recorded walk: two or three passes across the frame.
TYPICAL_WALK_S = 40.0

#: Metrics compared between model variants, with the difference that counts
#: as "the same measurement" -- the tool's own minimum detectable change where
#: one exists, so a variant passes only if its error is below what the trend
#: checks could notice anyway.
AGREEMENT = {
    "cadence_spm": 4.0,
    "stride_time_cv_pct": 1.0,
    "double_support_pct": 2.0,
    "step_length_asymmetry_pct": 3.0,
    "gait_speed_mps": 0.05,
    "trunk_ap_sway_norm": 0.01,
}


@dataclass
class VariantResult:
    model: str
    pose_fps: float
    ms_per_frame: float
    walk_minutes: float           # to analyse a TYPICAL_WALK_S walk at camera fps
    realtime_factor: float        # analysis time / walk length
    metrics: dict = field(default_factory=dict)
    disagreements: dict = field(default_factory=dict)
    agrees_with_heavy: Optional[bool] = None


@dataclass
class BenchReport:
    device: dict
    clip: str
    camera_fps: float
    record_width: int
    record_height: int
    decode_fps: float
    encode_fps: dict
    variants: list[VariantResult]
    recommendation: dict
    measured_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --------------------------------------------------------------------------
def run(settings: Settings, clip: Path, *, frames: int = 300,
        models: tuple[str, ...] = MODEL_VARIANTS, accuracy: bool = False,
        log=print) -> BenchReport:
    import cv2

    camera_fps = float(settings["camera.fps"])
    width, height = int(settings["camera.width"]), int(settings["camera.height"])

    log(f"device: {_device_summary()['summary']}")
    log(f"clip: {clip}")
    images, decode_fps = _decode(clip, frames)
    log(f"decode: {decode_fps:.0f} frames/s ({len(images)} frames)")

    resized = [cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
               for im in images[: min(len(images), 240)]]
    encode_fps = {}
    for codec in ("h264", "mjpeg"):
        encode_fps[codec] = _encode_fps(resized, codec, settings)
        verdict = "keeps up" if encode_fps[codec] >= 1.15 * camera_fps else "TOO SLOW"
        log(f"record {codec} {width}x{height}: {encode_fps[codec]:.0f} frames/s "
            f"({verdict} at {camera_fps:g} fps)")

    variants = []
    for variant in models:
        fps = _pose_fps(settings, variant, images)
        walk_frames = TYPICAL_WALK_S * camera_fps
        minutes = walk_frames / fps / 60.0
        factor = (walk_frames / fps) / TYPICAL_WALK_S
        variants.append(VariantResult(
            model=variant, pose_fps=fps, ms_per_frame=1000.0 / fps,
            walk_minutes=minutes, realtime_factor=factor,
        ))
        log(f"pose {variant:5s}: {fps:5.1f} frames/s ({1000 / fps:5.0f} ms/frame) -> "
            f"a {TYPICAL_WALK_S:.0f} s walk takes {minutes:4.1f} min ({factor:.1f}x)")

    if accuracy:
        _compare_variants(settings, clip, variants, log)

    report = BenchReport(
        device=_device_summary(), clip=str(clip), camera_fps=camera_fps,
        record_width=width, record_height=height, decode_fps=decode_fps,
        encode_fps=encode_fps, variants=variants,
        recommendation=_recommend(settings, variants, encode_fps, camera_fps),
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    rec = report.recommendation
    log(f"\nrecommendation: process {rec['mode'].upper()}"
        + (f" with the {rec['model']} model" if rec.get("model") else "")
        + f"; record as {rec['record_codec']}")
    for reason in rec["reasons"]:
        log(f"  - {reason}")
    return report


def save(report: BenchReport, settings: Settings) -> Path:
    path = settings.root / "bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    return path


def load_latest(settings: Settings) -> Optional[dict]:
    path = settings.root / "bench.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


# --------------------------------------------------------------------------
def _decode(clip: Path, frames: int):
    import cv2

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise ValueError(f"could not open {clip}")
    images = []
    t0 = time.perf_counter()
    while len(images) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        images.append(frame)
    elapsed = time.perf_counter() - t0
    cap.release()
    if not images:
        raise ValueError(f"no frames could be read from {clip}")
    return images, len(images) / max(elapsed, 1e-9)


def _encode_fps(images, codec: str, settings: Settings) -> float:
    """Frames per second this CPU can compress, as the recorder would."""
    import tempfile

    from .recorder import ClipWriter

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / ("bench.mkv")
        writer = ClipWriter(out, images[0].shape[1], images[0].shape[0],
                            fps=float(settings["camera.fps"]), codec=codec,
                            crf=int(settings["record.crf"]),
                            preset=str(settings["record.preset"]))
        t0 = time.perf_counter()
        dt = 1.0 / float(settings["camera.fps"])
        for i, image in enumerate(images):
            writer.write(image, i * dt)
        writer.close()
        return len(images) / max(time.perf_counter() - t0, 1e-9)


def _pose_fps(settings: Settings, variant: str, images) -> float:
    """Pose-landmarker throughput on real frames, as the pipeline runs it."""
    import cv2
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )

    from gaitscreen.pose.landmarker import _resolve_model

    cfg = settings.analysis_config().with_overrides(model_overrides(variant, settings.root))
    model_path = _resolve_model(cfg, settings.root)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO, num_poses=1,
    )
    rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images]
    with PoseLandmarker.create_from_options(options) as landmarker:
        # The first frames include graph start-up and the person detector
        # running before tracking takes over; they are not steady state.
        warm = min(15, len(rgb) // 4)
        for i in range(warm):
            landmarker.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb[i]),
                                        int(i * 1000 / 60))
        t0 = time.perf_counter()
        for i in range(warm, len(rgb)):
            landmarker.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb[i]),
                                        int(i * 1000 / 60))
        elapsed = time.perf_counter() - t0
    return (len(rgb) - warm) / max(elapsed, 1e-9)


def _compare_variants(settings: Settings, clip: Path, variants: list[VariantResult],
                      log) -> None:
    """Run the whole pipeline once per variant and compare with heavy."""
    from gaitscreen.pipeline import analyse_video

    for v in variants:
        cfg = settings.analysis_config().with_overrides(
            model_overrides(v.model, settings.root))
        log(f"accuracy: analysing the clip with the {v.model} model…")
        result = analyse_video(clip, cfg, user_id="bench",
                               project_root=settings.root, check_camera_motion=False)
        v.metrics = {k: result.metrics.value(k) for k in AGREEMENT}

    reference = next((v for v in variants if v.model == "heavy"), None)
    if reference is None:
        return
    for v in variants:
        diffs, ok = {}, True
        for key, tolerance in AGREEMENT.items():
            a, b = v.metrics.get(key), reference.metrics.get(key)
            if a is None or b is None:
                continue
            diffs[key] = round(abs(a - b), 4)
            ok &= abs(a - b) <= tolerance
        v.disagreements = diffs
        v.agrees_with_heavy = ok
        worst = ", ".join(f"{k} {d:g}" for k, d in diffs.items()
                          if d > AGREEMENT[k]) or "all within tolerance"
        log(f"  {v.model:5s} vs heavy: {worst}")


def _recommend(settings: Settings, variants: list[VariantResult],
               encode_fps: dict, camera_fps: float) -> dict:
    reasons = []
    record_codec = "h264" if encode_fps.get("h264", 0) >= 1.15 * camera_fps else "mjpeg"
    if record_codec == "mjpeg":
        reasons.append(
            f"H.264 compresses at only {encode_fps.get('h264', 0):.0f} frames/s here, "
            f"below the camera's {camera_fps:g}; MJPEG costs far less CPU at the price "
            "of roughly 5-10x larger files"
        )
    if encode_fps.get(record_codec, 0) < 1.15 * camera_fps:
        reasons.append("even MJPEG cannot keep up: lower camera.fps or the resolution")

    limit = float(settings["processing.max_realtime_factor"])
    # Only variants that measure like heavy are eligible, when that is known.
    eligible = [v for v in variants if v.agrees_with_heavy is not False]
    preferred = [v for v in eligible if v.realtime_factor <= limit]
    if preferred:
        best = min(preferred, key=lambda v: MODEL_VARIANTS.index(v.model))
        reasons.append(
            f"the {best.model} model analyses a {TYPICAL_WALK_S:.0f} s walk in "
            f"{best.walk_minutes:.1f} min ({best.realtime_factor:.1f}x its length), "
            f"within the {limit:g}x limit"
        )
        return {"mode": "local", "model": best.model, "record_codec": record_codec,
                "reasons": reasons}
    fastest = min(variants, key=lambda v: v.realtime_factor)
    reasons.append(
        f"even the fastest eligible model ({fastest.model}) takes "
        f"{fastest.realtime_factor:.1f}x the walk's length, over the {limit:g}x limit; "
        "send recordings to a server"
    )
    return {"mode": "remote", "model": None, "record_codec": record_codec,
            "reasons": reasons}


def _device_summary() -> dict:
    model = ""
    try:
        model = Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00\n ")
    except OSError:
        pass
    cpu = model or platform.processor() or platform.machine()
    temp = None
    try:
        temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        pass
    try:
        import mediapipe
        mp_version = mediapipe.__version__
    except Exception:  # noqa: BLE001
        mp_version = None
    return {
        "summary": f"{cpu}, {os.cpu_count()} cores, Python {platform.python_version()}",
        "model": model or None,
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "mediapipe": mp_version,
        "cpu_temp_c": temp,
    }
