# gaitedge — the headless device

Runs on a Raspberry Pi 5 (8 GB) with a camera attached, with no screen. It
watches the camera, records each walk that passes, analyses it on the device
or sends it to a server for analysis, and uploads the result.

The gait analysis is **not** reimplemented here: it is the `gaitscreen`
package from the repository root, installed as a dependency. The device, the
server and the Streamlit app therefore all measure a walk identically.

```
 camera ──► trigger ──► recorder ──► queue (on disk) ──► worker ──► uploader ──► server
            motion       MKV with       survives          analyse here    retries,
            + pre-roll   real frame     power cuts        (local) or      idempotent
                         timestamps                       forward (remote)
```

## Can the Pi do the processing?

**Yes, with the `full` pose model** — measured on a Pi 5 below. The code
measures this on each device rather than assuming it. Run `gaitedge bench` once on each unit;
`processing.mode: auto` then uses its answer.

### What was measured (development laptop, Intel i5-1135G7, 4 cores / 8 threads)

| | heavy | **full** | lite |
|---|---|---|---|
| pose estimation | 23.8 frames/s | **56.7 frames/s** | 64.8 frames/s |
| a 40 s walk at 60 fps | 1.7 min | **0.7 min** | 0.6 min |
| measures like heavy? (one clip) | reference | **yes** — cadence ±0.5, timing variability ±0.24, time on both feet ±1.3 | borderline — variability +0.93 (limit 1.0), both-feet −1.9 (limit 2.0) |

* `full` measures nearly identically to `heavy` at 2.4x the speed. `lite`
  adds only 14% more speed and sits at the edge of the tolerances, so it is
  not worth it. **`full` is the default.**
* Whole-walk analysis (pose + camera-movement check + analysis) took 84 s
  for a 52 s, four-pass walk with `full` — about 1.5x the pose time alone.
* Video compression at 1280×720: H.264 195 frames/s, MJPEG 215 frames/s.

### Measured on a Raspberry Pi 5 (8 GB, Python 3.13)

`gaitedge bench` on the pilot device, with the comfortable-pace sample clip:

| | measured |
|---|---|
| H.264 recording, 1280×720 | **135 frames/s** — keeps up at 60 fps with room to spare |
| MJPEG recording, 1280×720 | 88 frames/s (slower than H.264 on the Pi; keep `h264`) |
| `heavy` pose estimation | 7.4 frames/s |
| **`full` pose estimation** | **23.9 frames/s** |
| `lite` pose estimation | 30.7 frames/s |

What that means in practice:

* **The Pi can do the processing.** With `full`, pose estimation for a 40 s
  walk takes about 50 s at 30 fps or 100 s at 60 fps; with the rest of the
  analysis, roughly 1.5–3 minutes a walk. `heavy` would take 4–8 minutes.
* **Recording keeps up**, even though the Pi 5 has no hardware H.264 encoder
  and compresses on the CPU. Record at 720p.
* **Analysis does not need to be real time.** Walks are recorded first and
  analysed afterwards, in a lower-priority process that pauses while a new
  walk is being recorded. Bursts queue on disk and drain afterwards. Watch
  `queue.pending` in the heartbeat.
* **Use the official active cooler.** Sustained inference throttles a bare
  Pi 5. The benchmark records the CPU temperature.
* **Send to a server instead (`processing.mode: remote`)** when one Pi serves
  a very busy corridor, when you want the `heavy` model, or when many cameras
  feed one site. The device code is identical either way; only the mode changes.

## Install on a Pi

Raspberry Pi OS **64-bit** (Bookworm). MediaPipe has no 32-bit ARM build.

```bash
git clone <this repository> ~/gait
sudo bash ~/gait/edge/deploy/install_pi.sh  # `bash` needed if the file is not executable
sudo nano /etc/gaitedge/edge.yaml           # device id, coverage_m, server_url, token
sudo -u gaitedge /opt/gaitedge/venv/bin/python -m gaitedge --config /etc/gaitedge/edge.yaml doctor
sudo -u gaitedge /opt/gaitedge/venv/bin/python -m gaitedge --config /etc/gaitedge/edge.yaml record --seconds 40
sudo -u gaitedge /opt/gaitedge/venv/bin/python -m gaitedge --config /etc/gaitedge/edge.yaml bench --clip <that walk> --accuracy
sudo systemctl start gaitedge && journalctl -u gaitedge -f
```

`doctor` checks, in pass/fail lines: libraries, the camera library, ffmpeg,
storage and free space, the pose model (downloaded on first use), that the
camera really delivers the frame rate set, the server, and the benchmark.

## Commands

| command | does |
|---|---|
| `run` | the service: watch, record, analyse or forward, upload |
| `bench --clip F [--accuracy]` | measure recording and analysis speed per model, compare their results, recommend local or remote |
| `record --seconds N` | record one walk now, ignoring the trigger |
| `process FILE [--out DIR]` | analyse any video now; writes `result.json` and `overlay.mp4` |
| `status` | queue, disk use, last benchmark |
| `doctor` | check the install end to end |

## How it behaves

* **When it records.** Background subtraction on a ~160-pixel thumbnail, with
  1.5 s of pre-roll so the first steps are kept. The clip stays open through
  8 s of stillness, because a walker turning at the end of the walkway is out
  of frame for several seconds: closing sooner split a four-pass pilot walk
  into four 2–4-stride clips, too short to measure step-to-step timing. The
  cost is that two people passing within 8 s share a clip.
* **Timing.** Every frame is written with its real capture time (the camera
  sensor's own timestamp on a Pi camera). A frame lost to a busy CPU becomes a
  visible gap, which the analysis already handles, rather than silently
  shortening every stride around it.
* **Power cuts.** Recordings are MKV, which stays readable up to the last
  frame written. The queue is folders moved atomically between states; on
  restart, interrupted analysis is redone and interrupted recordings are kept
  if long enough.
* **Disk.** Finished jobs are pruned past `keep_days` or `max_disk_gb`.
  Nothing undelivered is ever deleted: when undelivered walks fill the
  allowance, recording pauses instead.
* **Privacy.** A clip in which no gait could be measured (a door, someone
  pausing) is deleted on the device, not sent. Raw footage is deleted once the
  server has it (`keep_raw_after_upload: false`). What leaves the device by
  default is the result and the annotated replay.
* **Who walked.** The device cannot know. Results are uploaded as
  `unassigned`, with only the absolute checks applied (e.g. speed under the
  illustrative threshold). Personal-trend checks need the walk assigned to a
  person, which belongs on the server/platform.
* **No server configured.** Walks are analysed and kept on the device.

## Live view

The service publishes what the camera sees at `http://127.0.0.1:8080/`:
the image, a red border and `● REC` timer while a walk is being recorded,
the motion level the trigger is reacting to, the queue, and the last result.
It encodes a few frames a second, and only while someone is watching.

* **On an attached monitor**, `install_pi.sh` adds
  `~/.config/autostart/gaitedge-preview.desktop` for the user who ran it. At
  desktop login it opens the view full screen if a monitor is connected, and
  does nothing otherwise. Alt+F4 closes it; `/opt/gaitedge/edge/deploy/preview_kiosk.sh`
  reopens it. (The service cannot open the window itself: it runs as a
  system account with no access to the desktop.)
* **From another computer**, set `preview.host: 0.0.0.0` and open
  `http://<pi>:8080`. Useful while aiming the camera; do not leave it open
  where people are filmed.
* `preview.enabled: false` turns it off.

## Upload API (what the server must implement)

Full contract in `gaitedge/uploader.py`; an executable reference in
`tools/dev_receiver.py`.

* `POST /api/v1/walks`, multipart: `manifest` (JSON) always; `result` +
  `overlay` when analysed on the device; `clip` (MKV, store unmodified — the
  timestamps inside are part of the measurement) when the server is to
  analyse it. Headers: `Authorization: Bearer <token>`, `X-Device-Id`,
  `Idempotency-Key: <job id>`.
* A repeated `Idempotency-Key` must be answered 200/409, never stored twice:
  a device that lost the reply sends again.
* 408, 429 and 5xx are retried with backoff (15 s doubling to 15 min); other
  4xx are permanent and not retried.
* `POST /api/v1/devices/<id>/heartbeat`: JSON status every 5 minutes —
  queue depths, disk, CPU temperature, processing mode.

## Development without a Pi

```bash
cd edge
python tools/dev_receiver.py --port 8700 --store received --token devtoken   # one terminal
# edge.yaml with camera.backend: file and camera.source: ../app/samples/vishal_limping.mp4
PYTHONPATH="../src:." python -m gaitedge --config edge.yaml run
PYTHONPATH="../src:." python -m pytest tests
```

The `file` backend replays a recording with its real timestamps, so the whole
loop — trigger, recorder, queue, analysis, upload — runs on any computer.

## Not verified yet

* **Live camera recording on the Pi.** `install_pi.sh` and `bench` have run
  on a Pi 5; the `picamera2` capture path and the systemd service have not
  yet been confirmed recording real walks. `doctor` is the first thing to run
  on a new unit.
* **picamera2 alongside pip packages.** picamera2 comes from apt and the
  venv uses `--system-site-packages`; if pip's newer numpy clashes with an
  apt-built module, `doctor` will show it on the picamera2 line.
* **Model agreement** was checked on one pilot clip. Run
  `bench --accuracy` on a few walks from the real site before relying on
  `full`.
