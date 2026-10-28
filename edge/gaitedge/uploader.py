"""Sending walks to the server, and the contract the server must implement.

API (version 1)
---------------
Every request carries ``Authorization: Bearer <device token>`` and
``X-Device-Id: <device id>``.

``POST /api/v1/walks`` -- one walk, as ``multipart/form-data``:

* ``manifest`` (application/json, always): the job record -- id, device id,
  creation time, duration, and ``processed_on``: ``"device"`` or ``"server"``.
* ``result`` (application/json) and ``overlay`` (video/mp4): when the device
  analysed the walk itself. ``overlay`` is the annotated replay, which is also
  the footage the web platform shows.
* ``clip`` (video/x-matroska): when the server is to analyse it. The file keeps
  every frame's capture timestamp, which the analysis depends on, so the server
  must store it as received and never re-encode it before analysis.

Also sent as ``Idempotency-Key: <job id>``. A device that loses the reply
cannot tell whether the upload landed, so it sends again; the server must
answer a repeat of a key it already holds with 200 or 409, not a second walk.

Responses: 200/201 accepted; 409 already held (treated as accepted); 408, 429
and 5xx retried; any other 4xx is a permanent rejection (bad token, malformed
request) and is not retried, so a misconfigured device does not hammer the
server forever.

``POST /api/v1/devices/<device id>/heartbeat`` -- JSON status every few
minutes, so the platform can show which devices are alive and how far behind
their queues are.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .settings import Settings
from .spool import Job

#: HTTP statuses worth trying again.
RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


class UploadError(RuntimeError):
    def __init__(self, message: str, *, permanent: bool):
        super().__init__(message)
        self.permanent = permanent


class Uploader:
    def __init__(self, settings: Settings, session=None):
        import requests

        self.settings = settings
        self.base = (settings["upload.server_url"] or "").rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {settings['upload.token'] or ''}",
            "X-Device-Id": settings.device_id,
            "User-Agent": "gaitedge/0.1",
        })

    @property
    def configured(self) -> bool:
        return bool(self.base)

    def send_walk(self, job: Job) -> dict:
        """Upload one job; returns the server's JSON reply."""
        import requests

        manifest = {
            "job_id": job.id,
            "device": self.settings.describe(),
            "created_at": job.data.get("created_at"),
            "duration_s": job.data.get("duration_s"),
            "processed_on": job.data.get("processed_on"),
            "trigger": job.data.get("trigger"),
            "dropped_frames": job.data.get("dropped_frames"),
        }
        files = {"manifest": ("manifest.json", json.dumps(manifest), "application/json")}
        handles = []
        try:
            for field_name, name, mime in _attachments(job):
                handle = open(job.dir / name, "rb")
                handles.append(handle)
                files[field_name] = (name, handle, mime)
            try:
                response = self.session.post(
                    f"{self.base}/api/v1/walks", files=files,
                    headers={"Idempotency-Key": job.id},
                    timeout=float(self.settings["upload.timeout_s"]),
                    verify=bool(self.settings["upload.verify_tls"]),
                )
            except requests.RequestException as exc:
                raise UploadError(f"network: {exc}", permanent=False) from exc
        finally:
            for handle in handles:
                handle.close()

        if response.status_code in (200, 201, 409):
            try:
                return response.json()
            except ValueError:
                return {}
        raise UploadError(
            f"server answered {response.status_code}: {response.text[:200]}",
            permanent=response.status_code not in RETRYABLE,
        )

    def heartbeat(self, status: dict) -> bool:
        if not self.configured:
            return False
        try:
            response = self.session.post(
                f"{self.base}/api/v1/devices/{self.settings.device_id}/heartbeat",
                json=status, timeout=15, verify=bool(self.settings["upload.verify_tls"]),
            )
            return response.ok
        except Exception:  # noqa: BLE001 - a missed heartbeat is not an error
            return False


def _attachments(job: Job) -> list[tuple[str, str, str]]:
    """Which files go with this job, by how it was processed."""
    out = []
    if job.data.get("processed_on") == "device":
        if (job.dir / "result.json").exists():
            out.append(("result", "result.json", "application/json"))
        if (job.dir / "overlay.mp4").exists():
            out.append(("overlay", "overlay.mp4", "video/mp4"))
    else:
        out.append(("clip", job.data.get("clip", "clip.mkv"), "video/x-matroska"))
    return out


def next_retry_delay(attempts: int, settings: Settings) -> float:
    base = float(settings["upload.retry_base_s"])
    return min(float(settings["upload.retry_max_s"]), base * (2 ** max(0, attempts - 1)))
