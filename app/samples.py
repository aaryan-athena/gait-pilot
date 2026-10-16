"""Built-in sample walks, so the tool can be tried without recording first.

Each sample was filmed in one fixed garage setup, paced to a metronome, so it
carries that setup's coverage length rather than borrowing whatever camera
setup the viewer has saved: measuring a garage video with someone else's
living-room coverage would give a confidently wrong walking speed.

Pose tracking is nearly all of an analysis's cost and depends only on the
video, so it is cached per sample for the life of the server. The first person
to pick a sample waits for it; everyone after that gets it at once. The flags
are still worked out fresh each time, against whichever person the walk is
being looked at for.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.shared import PROJECT_ROOT, open_repository
from gaitscreen.calibration.setup import CameraSetup, ensure_fixed_setup
from gaitscreen.config import Config

SAMPLES_DIR = PROJECT_ROOT / "app" / "samples"

#: The garage the samples were filmed in. The taped walking lane is 13.5 ft
#: (4.11 m) long and its near edge runs almost exactly across the picture; the
#: speeds in the pilot results sheet imply 4.09 m. 4.10 m is used.
GARAGE_COVERAGE_M = 4.10
GARAGE_SETUP_ID = "sample-garage-4p10"


@dataclass(frozen=True)
class Sample:
    key: str
    file: str
    title: str
    description: str

    @property
    def path(self) -> Path:
        return SAMPLES_DIR / self.file

    @property
    def poster(self) -> Path:
        return self.path.with_suffix(".jpg")

    @property
    def available(self) -> bool:
        return self.path.exists()


SAMPLES: tuple[Sample, ...] = (
    Sample(
        key="vishal_normal",
        file="vishal_normal_95spm.mp4",
        title="Comfortable pace",
        description="A steady walk at about 95 steps a minute, back and forth across "
                    "the frame. Even, regular steps at an unhurried speed.",
    ),
    Sample(
        key="vishal_slow",
        file="vishal_slow_75spm.mp4",
        title="Slow pace",
        description="The same walker at about 75 steps a minute. Slower speed and "
                    "longer time on both feet, with nothing else out of place.",
    ),
    Sample(
        key="vishal_limp",
        file="vishal_limping.mp4",
        title="Simulated limp",
        description="Favouring one leg. Shows how uneven steps and irregular timing "
                    "are picked up and flagged.",
    ),
)


def available_samples() -> list[Sample]:
    return [s for s in SAMPLES if s.available]


def get_sample(key: Optional[str]) -> Optional[Sample]:
    return next((s for s in SAMPLES if s.key == key), None)


def sample_setup(cfg: Config) -> CameraSetup:
    """The garage setup, registered in the database so saved samples can link
    to it. Falls back to an unstored setup if the database is unreachable: the
    walk can still be analysed, it just cannot be saved."""
    try:
        return ensure_fixed_setup(
            open_repository(cfg), GARAGE_SETUP_ID, GARAGE_COVERAGE_M,
            notes="Garage used for the built-in sample videos",
        )
    except Exception:  # noqa: BLE001
        return CameraSetup(GARAGE_COVERAGE_M, GARAGE_SETUP_ID, "")


#: Process-wide store of finished work, keyed by (sample, camera check,
#: algorithm version). Deliberately not ``st.cache_resource``: that records any
#: Streamlit element touched while the function runs and replays it on a cache
#: hit, and the progress bar updated during tracking belongs to the run that
#: created it -- so every second analysis of a sample failed on replay.
_STORE: dict[tuple, object] = {}
_LOCK = threading.Lock()


def _remembered(kind: str, key: str, camera_check: bool, algo_version: str, make):
    slot = (kind, key, camera_check, algo_version)
    with _LOCK:
        if slot in _STORE:
            return _STORE[slot]
    value = make()
    with _LOCK:
        _STORE[slot] = value
    return value


def cached_extraction(key: str, camera_check: bool, algo_version: str,
                      cfg: Config, calibration, progress=None):
    """Pose tracking for one sample, kept for the life of the server.

    ``algo_version`` is part of the key so that deploying a new algorithm can
    never serve tracking from the old one.
    """
    from gaitscreen.pipeline import extract_session

    sample = get_sample(key)
    return _remembered("extraction", key, camera_check, algo_version, lambda: (
        extract_session(
            sample.path, cfg, calibration=calibration, project_root=PROJECT_ROOT,
            progress=progress, check_camera_motion=camera_check,
        )
    ))


def cached_overlay(key: str, camera_check: bool, algo_version: str,
                   cfg: Config, result, progress=None) -> Optional[dict]:
    """The annotated video for one sample, drawn once and reused."""
    from gaitscreen.reporting.overlay import render_overlay_video

    def draw() -> dict:
        out = (cfg.resolve_path("storage.artifacts_root", PROJECT_ROOT)
               / "_samples" / f"{key}-{algo_version}-{int(camera_check)}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        overlay = render_overlay_video(
            result.extraction, result.analysis, out, cfg,
            max_width=int(cfg["reporting.overlay.max_width"]), progress=progress,
        )
        return {"path": str(overlay.path), "playable": overlay.browser_playable,
                "note": overlay.note}

    return _remembered("overlay", key, camera_check, algo_version, draw)
