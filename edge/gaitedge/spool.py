"""A job queue on disk that survives power cuts and network outages.

Each recorded walk is a *job*: a folder holding the recording, a ``job.json``
describing it, and later the results. The folder moves between state
directories as the job progresses::

    recording/ -> pending/ -> processing/ -> upload/ -> done/
                                  \\-> failed/       \\-> failed/

Moving a folder within one filesystem is atomic, so a power cut leaves every
job wholly in one state and never half in two. On start-up anything caught in
``processing`` goes back to ``pending``, and anything caught in ``recording``
is kept if the part written before the cut is long enough to be a walk -- the
recorder writes Matroska precisely so that part is readable.

Nothing un-uploaded is ever deleted to free space. When the disk fills with
work that has not reached the server yet, the device stops recording rather
than discard walks it has not delivered.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

STATES = ("recording", "pending", "processing", "upload", "done", "failed")
#: States holding work that has not reached the server.
UNDELIVERED = ("recording", "pending", "processing", "upload")


@dataclass
class Job:
    dir: Path
    data: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def state(self) -> str:
        return self.dir.parent.name

    @property
    def clip(self) -> Path:
        return self.dir / self.data.get("clip", "clip.mkv")

    def save(self) -> None:
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.dir / "job.json")

    def note(self, event: str, **detail) -> None:
        self.data.setdefault("history", []).append(
            {"at": _now(), "event": event, **detail})


class Spool:
    def __init__(self, root: Path):
        self.root = Path(root)
        for state in STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)

    # -- creating and moving ------------------------------------------------
    def new_job(self, device_id: str, **data) -> Job:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        job_id = f"{stamp}-{device_id}-{uuid.uuid4().hex[:6]}"
        job = Job(self.root / "recording" / job_id,
                  {"id": job_id, "device_id": device_id, "created_at": _now(),
                   "clip": "clip.mkv", "attempts": 0, **data})
        job.dir.mkdir(parents=True)
        job.note("recording started")
        job.save()
        return job

    def move(self, job: Job, state: str, event: Optional[str] = None, **detail) -> Job:
        if state not in STATES:
            raise ValueError(state)
        target = self.root / state / job.dir.name
        if event:
            job.note(event, **detail)
            job.save()
        os.replace(job.dir, target)
        job.dir = target
        return job

    def jobs(self, state: str) -> list[Job]:
        out = []
        for path in sorted((self.root / state).iterdir()):
            if not path.is_dir():
                continue
            try:
                data = json.loads((path / "job.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {"id": path.name, "clip": "clip.mkv", "unreadable_manifest": True}
            out.append(Job(path, data))
        return out

    def next(self, state: str, *, ready_at: Optional[float] = None) -> Optional[Job]:
        """The oldest job in ``state`` whose retry time (if any) has come."""
        now = time.time() if ready_at is None else ready_at
        for job in self.jobs(state):
            if float(job.data.get("next_attempt_at", 0)) <= now:
                return job
        return None

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for p in (self.root / s).iterdir() if p.is_dir()) for s in STATES}

    # -- recovery ------------------------------------------------------------
    def recover(self, *, min_clip_s: float) -> list[str]:
        """Put the queue back in order after an unclean shutdown."""
        actions = []
        for job in self.jobs("processing"):
            self.move(job, "pending", "requeued after restart")
            actions.append(f"{job.id}: processing interrupted, requeued")
        for job in self.jobs("recording"):
            duration = _readable_duration(job.clip)
            if duration is not None and duration >= min_clip_s:
                job.data["duration_s"] = duration
                job.data["recovered"] = True
                self.move(job, "pending", "recording cut short, kept", seconds=duration)
                actions.append(f"{job.id}: interrupted recording kept ({duration:.1f} s)")
            else:
                self.move(job, "failed", "recording cut short, unusable")
                actions.append(f"{job.id}: interrupted recording too short, discarded")
        return actions

    # -- disk ----------------------------------------------------------------
    def disk_used_gb(self, states: Iterable[str] = STATES) -> float:
        total = 0
        for state in states:
            for path in (self.root / state).rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        return total / 1e9

    def prune(self, *, max_gb: float, keep_days: float) -> list[str]:
        """Delete finished jobs: those past ``keep_days``, then oldest first
        until under ``max_gb``. Never touches undelivered work."""
        removed = []
        cutoff = time.time() - keep_days * 86400
        finished = sorted(self.jobs("done") + self.jobs("failed"),
                          key=lambda j: j.dir.stat().st_mtime)
        for job in list(finished):
            if job.dir.stat().st_mtime < cutoff:
                shutil.rmtree(job.dir, ignore_errors=True)
                removed.append(job.id)
                finished.remove(job)
        while finished and self.disk_used_gb() > max_gb:
            job = finished.pop(0)
            shutil.rmtree(job.dir, ignore_errors=True)
            removed.append(job.id)
        return removed

    def full(self, max_gb: float) -> bool:
        """True when undelivered work alone fills the allowance."""
        return self.disk_used_gb(UNDELIVERED) >= max_gb


def _readable_duration(clip: Path) -> Optional[float]:
    """How much of a possibly truncated recording can actually be read."""
    if not clip.exists() or clip.stat().st_size == 0:
        return None
    try:
        import cv2

        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            return None
        first = last = None
        while cap.grab():
            stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            first = stamp if first is None else first
            last = stamp
        cap.release()
        if first is None:
            return None
        return (last - first) / 1000.0
    except Exception:  # noqa: BLE001
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
