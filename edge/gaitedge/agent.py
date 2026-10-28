"""The device service: watch, record, analyse or forward, upload.

Three threads with a strict priority between them:

1. **Camera** (the main thread). Frames go to the trigger and, during a walk,
   to the recorder. It never waits on the others: recording is the one thing
   that cannot be redone later.
2. **Worker**. Takes recorded walks and either analyses them here or marks them
   for the server, per ``processing.mode``. Analysis runs in a separate,
   lower-priority process, for two reasons. The camera thread must win the CPU
   while a walk is being recorded, or frames are lost; and pose estimation is
   a large native library, so a crash or leak in it should cost one job, not
   the service. The worker also holds off while a walk is being recorded.
3. **Uploader**. Sends finished walks, retries with backoff, prunes old files,
   and posts a heartbeat.

A device with no server configured still works: walks are analysed and kept
on the device, ready to be collected or uploaded once a server exists.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import bench
from .camera import open_source
from .recorder import BackgroundWriter, ClipWriter
from .settings import Settings
from .spool import Job, Spool
from .trigger import make_trigger
from .uploader import UploadError, Uploader, next_retry_delay

log = logging.getLogger("gaitedge")

HEARTBEAT_EVERY_S = 300
PRUNE_EVERY_S = 600


class Agent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.spool = Spool(settings.root / "spool")
        self.uploader = Uploader(settings)
        self.stop = threading.Event()
        self.recording = threading.Event()
        self.clips = 0
        self._status: dict = {"state": "starting"}
        self._child: Optional[subprocess.Popen] = None
        self._preview = None
        self._clip_started: Optional[float] = None
        self._motion: Optional[float] = None
        self._last_result: Optional[str] = None

    # -- service -------------------------------------------------------------
    def run(self, max_clips: Optional[int] = None) -> int:
        _configure_logging()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: self.stop.set())
            except ValueError:
                pass  # not the main thread (tests)

        log.info("gaitedge starting as %s, storage %s", self.settings.device_id,
                 self.settings.root)
        for action in self.spool.recover(min_clip_s=float(self.settings["trigger.min_clip_s"])):
            log.info("recovered: %s", action)
        log.info("processing mode: %s", self.processing_mode())
        self._start_preview()

        workers = [threading.Thread(target=self._worker_loop, name="worker", daemon=True),
                   threading.Thread(target=self._upload_loop, name="uploader", daemon=True)]
        for t in workers:
            t.start()
        try:
            self._camera_loop(max_clips)
            if not self.stop.is_set():
                # The camera ended by itself (a replayed file ran out, or the
                # clip limit was reached) rather than by a shutdown signal:
                # finish the queued work instead of abandoning it mid-way.
                self._drain()
        finally:
            self.stop.set()
            if self._child and self._child.poll() is None:
                self._child.terminate()
            for t in workers:
                t.join(timeout=10)
            if self._preview is not None:
                self._preview.close()
            self._write_status("stopped")
        return 0

    def _start_preview(self) -> None:
        s = self.settings
        if not s["preview.enabled"]:
            return
        from .preview import Preview

        try:
            self._preview = Preview(host=str(s["preview.host"]), port=int(s["preview.port"]),
                                    fps=float(s["preview.fps"]), width=int(s["preview.width"]),
                                    status=self._live_status)
        except OSError as exc:
            # The view is a convenience; a taken port must not stop recording.
            log.warning("live view not started on port %s: %s", s["preview.port"], exc)
            return
        log.info("live view at http://%s:%d/", s["preview.host"], self._preview.port)

    def _live_status(self) -> dict:
        recording_s = (time.monotonic() - self._clip_started
                       if self._clip_started is not None else None)
        return {"device": self.settings.device_id, "state": self._status.get("state"),
                "recording": recording_s is not None,
                "recording_s": recording_s or 0.0, "motion": self._motion,
                "queue": self.spool.counts(), "last_result": self._last_result}

    def _drain(self, timeout_s: float = 3600.0) -> None:
        """Wait until nothing is waiting to be processed or sent."""
        log.info("camera finished; completing queued work")
        deadline = time.time() + timeout_s
        waiting = ("pending", "processing") + (("upload",) if self.uploader.configured else ())
        while not self.stop.is_set() and time.time() < deadline:
            counts = self.spool.counts()
            if not any(counts[s] for s in waiting) and not self.recording.is_set():
                return
            # Uploads waiting out a long retry backoff would hold this open;
            # they stay queued for the next start instead.
            retrying = [j for j in self.spool.jobs("upload")
                        if float(j.data.get("next_attempt_at", 0)) > time.time() + 30]
            if retrying and not any(counts[s] for s in ("pending", "processing")) \
                    and len(retrying) == counts["upload"]:
                log.info("%d uploads waiting to retry; left queued", len(retrying))
                return
            self.stop.wait(1.0)

    def processing_mode(self) -> str:
        """local or remote, resolving ``auto`` from the last benchmark."""
        mode = self.settings["processing.mode"]
        if mode != "auto":
            return mode
        latest = bench.load_latest(self.settings)
        if latest and latest.get("recommendation", {}).get("mode") in ("local", "remote"):
            return latest["recommendation"]["mode"]
        # No benchmark yet. Without a server there is nowhere else to send
        # walks, so analyse here; with one, send them until a benchmark
        # shows this device can keep up.
        log.warning("no benchmark found; run `gaitedge bench` once on this device")
        return "remote" if self.uploader.configured else "local"

    # -- camera --------------------------------------------------------------
    def _camera_loop(self, max_clips: Optional[int]) -> None:
        s = self.settings
        source = open_source(s)
        log.info("camera %s: %dx%d @ %g fps", s["camera.backend"], source.width,
                 source.height, source.fps)
        trigger = make_trigger(s, float(s["camera.fps"]))
        writer: Optional[BackgroundWriter] = None
        job: Optional[Job] = None
        paused_for_disk = False
        try:
            for image, t in source.frames():
                if self.stop.is_set():
                    break
                event = trigger.feed(image, t)
                self._motion = getattr(trigger, "last_moving_frac", None)
                if self._preview is not None:
                    self._preview.offer(image, motion=self._motion, recording_s=(
                        time.monotonic() - self._clip_started
                        if self._clip_started is not None else None))
                if event.kind == "start":
                    if self.spool.full(float(s["storage.max_disk_gb"])):
                        if not paused_for_disk:
                            log.error("disk allowance full of undelivered walks; "
                                      "not recording until some are uploaded")
                            paused_for_disk = True
                        continue
                    paused_for_disk = False
                    job, writer = self._open_clip(source, event.reason)
                if writer is not None:
                    for frame, ft in event.frames:
                        writer.write(frame, ft)
                if event.kind == "stop" and writer is not None:
                    self._close_clip(job, writer, event.reason)
                    writer, job = None, None
                    self.clips += 1
                    if max_clips and self.clips >= max_clips:
                        break
            final = trigger.flush()
            if final is not None and writer is not None:
                self._close_clip(job, writer, final.reason)
                writer, job = None, None
                self.clips += 1
        finally:
            if writer is not None:
                self._close_clip(job, writer, "shutdown")
            source.close()

    def _open_clip(self, source, reason: str):
        s = self.settings
        job = self.spool.new_job(s.device_id, trigger=reason,
                                 camera={"width": source.width, "height": source.height,
                                         "fps": float(s["camera.fps"])})
        writer = BackgroundWriter(ClipWriter(
            job.clip, source.width, source.height, fps=float(s["camera.fps"]),
            codec=s["record.codec"], crf=int(s["record.crf"]),
            preset=str(s["record.preset"]),
        ))
        self.recording.set()
        self._clip_started = time.monotonic()
        self._write_status("recording", job=job.id)
        log.info("recording %s (%s)", job.id, reason)
        return job, writer

    def _close_clip(self, job: Job, writer: BackgroundWriter, reason: str) -> None:
        try:
            writer.close()
            duration = writer.writer.duration_s
            job.data.update(duration_s=round(duration, 2), frames=writer.writer.frames,
                            dropped_frames=writer.dropped)
            if duration < float(self.settings["trigger.min_clip_s"]):
                log.info("discarding %s: %.1f s is too short to be a walk", job.id, duration)
                shutil.rmtree(job.dir, ignore_errors=True)
                return
            if writer.dropped:
                log.warning("%s: %d frames dropped while recording (CPU too slow?)",
                            job.id, writer.dropped)
            self.spool.move(job, "pending", "recorded", reason=reason, seconds=duration)
            log.info("recorded %s: %.1f s, %d frames", job.id, duration, writer.writer.frames)
        except Exception as exc:  # noqa: BLE001
            log.exception("recording %s failed", job.id)
            job.data["error"] = str(exc)
            self.spool.move(job, "failed", "recording failed", error=str(exc))
        finally:
            self.recording.clear()
            self._clip_started = None
            self._write_status("watching")

    # -- worker --------------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self.stop.is_set():
            # Analysis waits while a walk is being recorded: the recorder needs
            # every core it can get to keep up with the camera.
            if self.recording.is_set():
                self.stop.wait(1.0)
                continue
            job = self.spool.next("pending")
            if job is None:
                self.stop.wait(2.0)
                continue
            try:
                self._handle(job)
            except Exception as exc:  # noqa: BLE001
                log.exception("job %s failed", job.id)
                job.data["error"] = str(exc)
                self.spool.move(job, "failed", "processing failed", error=str(exc))

    def _handle(self, job: Job) -> None:
        mode = self.processing_mode()
        if mode == "remote":
            job.data["processed_on"] = "server"
            self.spool.move(job, "upload", "queued for server analysis")
            return

        self.spool.move(job, "processing", "analysis started")
        started = time.time()
        ok = self._analyse_in_child(job)
        if self.stop.is_set():
            return  # left in processing; recovered on the next start
        if not ok:
            raise RuntimeError("analysis process failed; see the service log")

        result = json.loads((job.dir / "result.json").read_text(encoding="utf-8"))
        job.data["processed_on"] = "device"
        job.data["analysis_s"] = round(time.time() - started, 1)
        job.data["summary"] = result.get("summary")
        if not result["summary"]["is_walk"]:
            log.info("%s: no measurable walk (%s); discarded", job.id,
                     result["summary"]["headline"])
            shutil.rmtree(job.dir, ignore_errors=True)
            return
        log.info("%s analysed in %.0f s: %s", job.id, job.data["analysis_s"],
                 result["summary"]["headline"])
        self._last_result = f"{time.strftime('%H:%M')} {result['summary']['headline']}"
        target = "upload" if self.uploader.configured else "done"
        self.spool.move(job, target, "analysed on device")

    def _analyse_in_child(self, job: Job) -> bool:
        cmd = [sys.executable, "-m", "gaitedge"]
        if self.settings.source:
            cmd += ["--config", str(self.settings.source)]
        cmd += ["process", str(job.clip), "--out", str(job.dir)]
        if not self.settings["processing.make_overlay"]:
            cmd.append("--no-overlay")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [
            str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]))
        kwargs = {}
        if hasattr(os, "nice"):
            kwargs["preexec_fn"] = lambda: os.nice(10)
        with open(job.dir / "analysis.log", "w", encoding="utf-8") as logfile:
            self._child = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                                           env=env, **kwargs)
            while self._child.poll() is None:
                if self.stop.wait(1.0):
                    self._child.terminate()
                    return False
        return self._child.returncode == 0 and (job.dir / "result.json").exists()

    # -- uploader --------------------------------------------------------------
    def _upload_loop(self) -> None:
        last_heartbeat = last_prune = 0.0
        while not self.stop.is_set():
            now = time.time()
            if now - last_prune > PRUNE_EVERY_S:
                last_prune = now
                removed = self.spool.prune(max_gb=float(self.settings["storage.max_disk_gb"]),
                                           keep_days=float(self.settings["storage.keep_days"]))
                if removed:
                    log.info("pruned %d finished jobs", len(removed))
            if now - last_heartbeat > HEARTBEAT_EVERY_S:
                last_heartbeat = now
                self.uploader.heartbeat(self._heartbeat())
            job = self.spool.next("upload") if self.uploader.configured else None
            if job is None:
                self.stop.wait(3.0)
                continue
            self._upload(job)

    def _upload(self, job: Job) -> None:
        try:
            reply = self.uploader.send_walk(job)
        except UploadError as exc:
            job.data["attempts"] = int(job.data.get("attempts", 0)) + 1
            if exc.permanent:
                log.error("%s rejected by server: %s", job.id, exc)
                self.spool.move(job, "failed", "rejected by server", error=str(exc))
                return
            delay = next_retry_delay(job.data["attempts"], self.settings)
            job.data["next_attempt_at"] = time.time() + delay
            job.note("upload failed, will retry", error=str(exc), retry_in_s=delay)
            job.save()
            log.warning("%s upload failed (%s); retrying in %.0f s", job.id, exc, delay)
            return
        job.data["server"] = reply
        if not self.settings["storage.keep_raw_after_upload"]:
            # The server has it; the raw footage of a person does not need to
            # stay on a device on a corridor wall.
            job.clip.unlink(missing_ok=True)
        self.spool.move(job, "done", "uploaded", walk_id=reply.get("walk_id"))
        log.info("%s uploaded", job.id)

    # -- status ----------------------------------------------------------------
    def _heartbeat(self) -> dict:
        from . import __version__

        temp = None
        try:
            temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
        except (OSError, ValueError):
            pass
        return {
            "device": self.settings.describe(),
            "version": __version__,
            "state": self._status.get("state"),
            "processing_mode": self.processing_mode(),
            "queue": self.spool.counts(),
            "disk_used_gb": round(self.spool.disk_used_gb(), 2),
            "disk_free_gb": round(shutil.disk_usage(self.settings.root).free / 1e9, 1),
            "cpu_temp_c": temp,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def _write_status(self, state: str, **extra) -> None:
        self._status = {"state": state, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
        try:
            path = self.settings.root / "status.json"
            path.write_text(json.dumps(self._status), encoding="utf-8")
        except OSError:
            pass


def record_once(settings: Settings, *, seconds: float) -> Optional[Job]:
    """Record one clip of ``seconds`` now, ignoring the motion trigger."""
    _configure_logging()
    spool = Spool(settings.root / "spool")
    source = open_source(settings)
    job = spool.new_job(settings.device_id, trigger="manual")
    writer = ClipWriter(job.clip, source.width, source.height,
                        fps=float(settings["camera.fps"]), codec=settings["record.codec"],
                        crf=int(settings["record.crf"]), preset=str(settings["record.preset"]))
    start = None
    try:
        for image, t in source.frames():
            start = t if start is None else start
            writer.write(image, t)
            if t - start >= seconds:
                break
    finally:
        writer.close()
        source.close()
    if writer.frames == 0:
        shutil.rmtree(job.dir, ignore_errors=True)
        return None
    job.data.update(duration_s=round(writer.duration_s, 2), frames=writer.frames)
    return spool.move(job, "pending", "recorded", reason="manual")


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s")
