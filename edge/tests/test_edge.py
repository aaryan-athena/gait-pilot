"""The device side: recording, queue, trigger, uploads, and the service loop."""
import json
import threading
import time

import numpy as np
import pytest

from conftest import SAMPLE


# --------------------------------------------------------------------------
# recording keeps the real clock
# --------------------------------------------------------------------------
def _frames(n, w=320, h=180):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    for i in range(n):
        frame = base.copy()
        frame[:, (i * 4) % w:(i * 4) % w + 20] = 255  # something moving
        yield frame


@pytest.mark.parametrize("codec", ["h264", "mjpeg"])
def test_a_dropped_frame_survives_as_a_gap(tmp_path, codec):
    """The point of stamping frames: a lost frame must read back as a gap.

    The gait pipeline rebuilds its clock from these stamps. If the file
    pretended frames were evenly spaced, every stride around a dropped frame
    would be measured short.
    """
    from gaitedge.recorder import ClipWriter
    from gaitscreen.io.video import frame_grid, timed_frames

    path = tmp_path / "clip.mkv"
    writer = ClipWriter(path, 320, 180, fps=60, codec=codec)
    stamps = [i / 60 for i in range(120) if not 50 <= i < 55]   # 5 frames lost
    for frame, t in zip(_frames(len(stamps)), stamps):
        writer.write(frame, t)
    writer.close()

    read = np.array([ms for _, _, ms in timed_frames(path)])
    assert read.size == len(stamps)
    grid = frame_grid(read, container_fps=float(len(read)) / 2)
    assert grid.fps == pytest.approx(60, rel=0.01)
    assert grid.dropped == 5


def test_background_writer_never_blocks_the_camera(tmp_path):
    from gaitedge.recorder import BackgroundWriter, ClipWriter

    writer = BackgroundWriter(ClipWriter(tmp_path / "c.mkv", 320, 180, fps=60), max_queue=4)
    start = time.perf_counter()
    for i, frame in enumerate(_frames(200)):
        writer.write(frame, i / 60)
    assert time.perf_counter() - start < 1.0   # put_nowait, never waits
    writer.close()
    assert writer.writer.frames + writer.dropped == 200


# --------------------------------------------------------------------------
# queue
# --------------------------------------------------------------------------
def test_jobs_move_through_states(tmp_path):
    from gaitedge.spool import Spool

    spool = Spool(tmp_path)
    job = spool.new_job("dev")
    assert job.state == "recording"
    spool.move(job, "pending", "recorded")
    assert spool.counts()["pending"] == 1
    assert spool.next("pending").id == job.id
    data = json.loads((job.dir / "job.json").read_text())
    assert [h["event"] for h in data["history"]] == ["recording started", "recorded"]


def test_recovery_after_a_power_cut(tmp_path):
    from gaitedge.recorder import ClipWriter
    from gaitedge.spool import Spool

    spool = Spool(tmp_path)
    interrupted = spool.new_job("dev")
    writer = ClipWriter(interrupted.clip, 320, 180, fps=60)
    for i, frame in enumerate(_frames(360)):
        writer.write(frame, i / 60)
    writer.close()
    stub = spool.new_job("dev")              # cut off before a usable clip
    processing = spool.new_job("dev")
    spool.move(processing, "processing")

    spool.recover(min_clip_s=4.0)

    assert {j.id for j in spool.jobs("pending")} == {interrupted.id, processing.id}
    assert [j.id for j in spool.jobs("failed")] == [stub.id]


def test_undelivered_work_is_never_pruned(tmp_path):
    from gaitedge.spool import Spool

    spool = Spool(tmp_path)
    waiting = spool.new_job("dev")
    (waiting.dir / "clip.mkv").write_bytes(b"x" * 2_000_000)
    spool.move(waiting, "upload")
    finished = spool.new_job("dev")
    (finished.dir / "clip.mkv").write_bytes(b"x" * 2_000_000)
    spool.move(finished, "done")

    spool.prune(max_gb=0.000001, keep_days=30)

    assert spool.counts()["upload"] == 1
    assert spool.counts()["done"] == 0
    assert spool.full(max_gb=0.000001)


# --------------------------------------------------------------------------
# trigger
# --------------------------------------------------------------------------
def test_motion_starts_and_stops_a_clip(settings_factory):
    from gaitedge.trigger import MotionTrigger

    settings = settings_factory(trigger__warmup_s=0.5, trigger__pre_roll_s=0.5,
                                trigger__stop_after_s=1.0)
    trigger = MotionTrigger(settings, fps=30)
    background = np.full((180, 320, 3), 90, dtype=np.uint8)
    events = []
    for i in range(30 * 8):
        t = i / 30
        frame = background.copy()
        if 2.0 <= t < 5.0:                     # a "person" crosses the frame
            x = int((t - 2.0) / 3.0 * 280)
            frame[40:170, x:x + 40] = 230
        events.append(trigger.feed(frame, t).kind)

    starts = [i for i, k in enumerate(events) if k == "start"]
    stops = [i for i, k in enumerate(events) if k == "stop"]
    assert len(starts) == 1 and len(stops) == 1
    assert 2.0 <= starts[0] / 30 < 2.6          # within a few frames of entering
    assert 5.5 <= stops[0] / 30 < 6.6           # after stop_after_s of stillness


def test_pre_roll_keeps_the_first_steps(settings_factory):
    from gaitedge.trigger import MotionTrigger

    settings = settings_factory(trigger__warmup_s=0.0, trigger__pre_roll_s=1.0)
    trigger = MotionTrigger(settings, fps=30)
    background = np.full((180, 320, 3), 90, dtype=np.uint8)
    for i in range(60):
        event = trigger.feed(background, i / 30)
    frame = background.copy()
    for i in range(60, 70):
        frame = background.copy()
        frame[40:170, (i - 60) * 10:(i - 60) * 10 + 40] = 230
        event = trigger.feed(frame, i / 30)
        if event.kind == "start":
            break
    assert event.kind == "start"
    assert len(event.frames) == 30               # a full second of pre-roll


# --------------------------------------------------------------------------
# uploads
# --------------------------------------------------------------------------
@pytest.fixture
def receiver(tmp_path):
    from dev_receiver import serve

    def start(fail_first=0, token="devtoken"):
        server = serve(0, tmp_path / "received", token, fail_first)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    servers = []

    def factory(**kw):
        server, url = start(**kw)
        servers.append(server)
        return url

    yield factory
    for s in servers:
        s.shutdown()


def _ready_job(settings, processed_on="device"):
    from gaitedge.spool import Spool

    spool = Spool(settings.root / "spool")
    job = spool.new_job(settings.device_id, duration_s=12.0, processed_on=processed_on)
    (job.dir / "clip.mkv").write_bytes(b"clip")
    (job.dir / "result.json").write_text(json.dumps({"summary": {"is_walk": True}}))
    (job.dir / "overlay.mp4").write_bytes(b"mp4")
    job.save()
    return spool, spool.move(job, "upload")


def test_upload_delivers_a_walk(settings_factory, receiver, tmp_path):
    from gaitedge.uploader import Uploader

    url = receiver()
    settings = settings_factory(upload__server_url=url, upload__token="devtoken")
    _, job = _ready_job(settings)

    reply = Uploader(settings).send_walk(job)

    stored = tmp_path / "received" / "walks" / reply["walk_id"]
    assert sorted(p.name for p in stored.iterdir()) == ["manifest.json", "overlay.mp4",
                                                       "result.json"]
    manifest = json.loads((stored / "manifest.json").read_text())
    assert manifest["processed_on"] == "device"
    assert manifest["device"]["device_id"] == "test-device"


def test_server_analysis_sends_the_raw_clip(settings_factory, receiver, tmp_path):
    from gaitedge.uploader import Uploader

    settings = settings_factory(upload__server_url=receiver(), upload__token="devtoken")
    _, job = _ready_job(settings, processed_on="server")
    reply = Uploader(settings).send_walk(job)
    stored = tmp_path / "received" / "walks" / reply["walk_id"]
    assert sorted(p.name for p in stored.iterdir()) == ["clip.mkv", "manifest.json"]


def test_a_repeated_upload_is_not_stored_twice(settings_factory, receiver, tmp_path):
    """A device that lost the reply sends again; that must not be a second walk."""
    from gaitedge.uploader import Uploader

    settings = settings_factory(upload__server_url=receiver(), upload__token="devtoken")
    _, job = _ready_job(settings)
    uploader = Uploader(settings)
    first = uploader.send_walk(job)
    second = uploader.send_walk(job)
    assert second.get("duplicate") is True
    assert second["walk_id"] == first["walk_id"]
    assert len(list((tmp_path / "received" / "walks").iterdir())) == 1


def test_an_outage_is_retried_and_a_bad_token_is_not(settings_factory, receiver):
    from gaitedge.uploader import UploadError, Uploader

    settings = settings_factory(upload__server_url=receiver(fail_first=1),
                                upload__token="devtoken")
    _, job = _ready_job(settings)
    with pytest.raises(UploadError) as outage:
        Uploader(settings).send_walk(job)
    assert not outage.value.permanent

    bad = settings_factory(upload__server_url=receiver(), upload__token="wrong")
    _, job = _ready_job(bad)
    with pytest.raises(UploadError) as rejected:
        Uploader(bad).send_walk(job)
    assert rejected.value.permanent


# --------------------------------------------------------------------------
# the service, end to end, on a replayed walk
# --------------------------------------------------------------------------
def test_service_records_and_forwards_a_walk(settings_factory, receiver, tmp_path):
    """Replay a real walk through the service in server-analysis mode.

    Exercises camera -> trigger -> recorder -> queue -> uploader without pose
    estimation, so it runs in seconds.
    """
    from gaitedge.agent import Agent

    url = receiver()
    settings = settings_factory(upload__server_url=url, upload__token="devtoken",
                                processing__mode="remote", trigger__max_clip_s=12.0)
    agent = Agent(settings)
    agent.run(max_clips=1)          # returns once the walk is delivered

    walks = list((tmp_path / "received" / "walks").iterdir())
    assert len(walks) == 1
    clip = walks[0] / "clip.mkv"
    assert clip.stat().st_size > 10_000
    from gaitscreen.io.video import timed_frames
    stamps = [ms for _, _, ms in timed_frames(clip)]
    assert len(stamps) > 60 * 4
    # raw footage leaves the device once delivered
    done_job = agent.spool.jobs("done")[0]
    assert not done_job.clip.exists()


def test_live_view_serves_page_status_and_frames():
    import json
    import urllib.request

    import cv2

    from gaitedge.preview import Preview

    preview = Preview(port=0, fps=100, status=lambda: {"device": "d1", "recording": True})
    base = f"http://127.0.0.1:{preview.port}"
    try:
        assert b"gaitedge" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/status").read())["device"] == "d1"
        frame = np.full((720, 1280, 3), 90, np.uint8)
        preview.offer(frame, recording_s=3.0, motion=0.05)
        snap = urllib.request.urlopen(base + "/snapshot.jpg").read()
        image = cv2.imdecode(np.frombuffer(snap, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[1] == 960                      # downscaled
        assert image[2, 480, 2] > image[2, 480, 0] + 50   # red recording border

        stream = urllib.request.urlopen(base + "/stream", timeout=5)
        time.sleep(0.05)
        preview.offer(frame)                              # a viewer is now watching
        assert stream.readline().startswith(b"--frame")
        stream.close()
    finally:
        preview.close()
