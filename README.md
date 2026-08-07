# gaitscreen

Video-based gait screening and trend monitoring for elderly users.

**This is a screening and trend-monitoring tool, not a diagnostic instrument.** It
exists to flag concerning *changes* in an individual's walking over time and to
give caregivers a simple trend view. It prioritises consistency and repeatability
over absolute accuracy. Every clinical threshold in `config/default.yaml` is
illustrative and must be reviewed against current geriatric literature before real
use.

Read [`docs/limitations.md`](docs/limitations.md) before trusting any number this
tool produces. It is not boilerplate — several limitations constrain which metrics
are meaningful for a given recording.

## Pilot platform

```bash
streamlit run app/main.py
```

A four-page web UI for pilot testing: upload a video, get metrics, save it to a
person's history, and view trends. It is built to **show its working**, because
the failure mode that matters in a pilot is a plausible-looking wrong number:

- every unmeasurable metric states *why* it could not be measured;
- every unreliable one carries its caveat next to the value, not in a footnote;
- detected heel strikes and toe-offs are plotted over the raw signal, so a
  tester can see at a glance whether segmentation actually locked on;
- an independent cadence check, sharing no code with the main detector, is shown
  alongside the pipeline's own figure.

Pages: **Analyse a walk**, **Trends**, **Calibration** (only needed for gait
speed in m/s — everything else is scale-free), and **Limitations**.

### Deploying to Streamlit Community Cloud

Point it at this repository with `app/main.py` as the entry point. Three files
in the repo exist for that deployment and are needed there:

| file | why |
| --- | --- |
| `requirements.txt` | Python dependencies |
| `packages.txt` | `libgl1`, `libglib2.0-0t64` — **required**, see below |
| `.streamlit/config.toml` | 400 MB upload limit for video files |

`packages.txt` is not optional. `mediapipe` depends on `opencv-contrib-python`,
which links against `libGL.so.1` and `libgthread-2.0.so.0`; neither is in the
deployment image, and without them the app dies at import. Switching to headless
OpenCV in `requirements.txt` does not help, because pip installs the full build
anyway to satisfy mediapipe.

Three traps in that one small file, each of which produces a failed build rather
than a useful message:

- **Bare package names only — no comments, no padding.** Every line goes
  straight to `apt-get install`, so a `#` explanation becomes a list of
  nonexistent packages (`E: Unable to locate package Required,` …), and an
  apostrophe anywhere in it breaks `xargs` before apt even runs.
- **One bad name takes down the whole step.** apt installs nothing if any entry
  is unresolvable, so a wrong guess also loses the entries that were correct.
  Verify names against `https://packages.debian.org/<suite>/<package>` before
  adding them — a real package has a "Details of package" page, a virtual one
  does not.
- **Names are specific to the base image**, currently Debian trixie. Both
  libraries were renamed at some point: `libgl1-mesa-glx` became `libgl1` after
  Debian 11, and `libglib2.0-0` became `libglib2.0-0t64` in the 64-bit `time_t`
  transition. The old names will not resolve on trixie.

Two things to know about the hosted environment:

- **The filesystem is ephemeral.** The session database, uploaded videos and
  per-session artifacts all live under `data/`, which is wiped on restart. That
  is fine for trying the tool out, but a pilot that needs to accumulate a trend
  across weeks needs external storage — the whole point of the tool is the
  trend, and a baseline needs at least five sessions.
- **Memory is roughly 1 GB** on the free tier, against a 30 MB heavy model plus
  the TFLite runtime. If the app is being killed, switch `pose.model_path` and
  `pose.model_url` to the `full` or `lite` variant. Do that only if you have to:
  lighter models are less precise about foot landmarks, which is exactly what
  gait event detection depends on.

## Setup

```bash
python -m pip install -r requirements.txt   # or requirements-dev.txt to run tests
python -m pip install -e .
```

Python 3.11. Current MediaPipe releases have removed the legacy
`mediapipe.solutions` API, so the Tasks `PoseLandmarker` and its `.task` model
bundle are required. The bundle is ~30 MB, so it is **not** in the repository —
it downloads on first use and caches in `models/`. Set `pose.auto_download:
false` in the config for air-gapped installs and place the file there yourself.

Then check everything works:

```bash
pytest                                   # 137 tests, no video needed
gaitscreen probe sample_video/*.mp4      # what the sample clips will support
```

## Recording protocol

The measurement is only as good as the recording. In order of impact:

1. **60 fps.** At 25–30 fps the frame interval is as large as the stride-time
   standard deviation being measured, so stride-time variability — the most
   fall-risk-predictive metric — is degraded to indicative only.
2. **Fixed camera on a tripod, side-on.** Do not pan or follow the subject. A
   camera that tracks the walker cancels the displacement that gait speed is
   computed from, and the tool will refuse to report speed.
3. **Frame the whole walk.** The subject must cross a good fraction of the frame.
4. **Multiple passes per session**, one in each direction. Two reasons: enough
   strides for a stable variability estimate, and each limb gets to be the
   near (unoccluded) limb once, which is what makes left/right asymmetry
   trustworthy from a single camera.
5. **Don't move the camera between sessions.** If you must, recalibrate. The tool
   detects probable camera movement and suppresses distance metrics rather than
   reporting rescaled ones.
6. **Record assistive-device use in session metadata.** Pose estimation cannot see
   a cane.

## Usage

```bash
# What is this video, and what will it support?
gaitscreen probe walk.mp4

# One-time calibration per user/camera setup. Mark the reference ALONG the
# walking path. --method homography (4 floor points) is more accurate.
gaitscreen calibrate --user alice --video walk.mp4 --distance-m 2.0

# Extract landmarks, smooth, and report what is measurable.
gaitscreen extract walk.mp4 --user alice

# Full pipeline: metrics, quality score and flags.
gaitscreen analyze walk.mp4 --user alice --save

# History and trends.
gaitscreen sessions --user alice
```

## Design notes

Decisions where the implementation departs from the obvious approach, and why:

- **Zero-phase Butterworth, not One-Euro.** One-Euro is built for real-time input:
  it trades accuracy for latency, and adapts its cutoff to signal *speed*. That
  makes its lag vary between fast and slow strides, which biases stride-time
  variability undetectably. Since processing is offline, `filtfilt` gives
  exactly zero phase distortion instead. One-Euro and Kalman are implemented
  behind the same interface for a future live-camera path.
  See `tests/test_filters.py`, which holds this claim to account.
- **Zeni coordinate-based event detection, not ankle vertical minima.** Ankle
  vertical minimum corresponds to foot-flat, not heel contact, and the foot
  landmarks are the jitteriest MediaPipe produces. The maximum anterior heel
  position relative to the pelvis is a sharper and better-established marker,
  and it yields toe-off (from minimum anterior toe position) at the same time —
  which double-support time requires and heel strikes alone cannot give.
- **Sub-frame event timing everywhere.** See limitation 1.
- **Speed is refused, not estimated,** when the recording cannot support it. A
  fabricated low speed reads as the tool's high-risk signal and would poison the
  personal baseline permanently.
- **Raw landmarks are retained and `ALGO_VERSION` is stamped per session.**
  Changing the algorithm changes the measurement; mixing versions in one person's
  history creates a step change that mimics real decline. History must be
  reprocessable under a single version.
- **Metrics are nullable with a stated reason.** A metric that could not be
  measured is reported as missing, never defaulted — in a screening tool a
  plausible default reads as a normal result.
- **Frames are never dropped.** Low-visibility samples become NaN on a uniform
  time base; short gaps are interpolated, long ones split the recording into
  separate analysis segments. Dropping frames would compress the time axis and
  shorten every stride time it touched.
- **Stride period is estimated before events are detected.** A fixed minimum
  peak separation cannot serve both a brisk walker and a slow one: set it low
  and one heel strike is counted twice, set it high and every second strike is
  merged into a double-length stride. Both failures were observed on the sample
  clips before this was added. The pelvis-relative signal is also detrended
  first, because a panning camera adds drift whose autocorrelation swamps the
  gait rhythm entirely.
- **Three independent self-consistency checks**, because each individual metric
  can look perfectly plausible while the segmentation underneath is wrong:
  cadence must equal `120 / stride time` (one stride is two steps); the opposite
  foot must strike near mid-stride (otherwise the limbs are not being told
  apart); and an ankle-separation step count, which shares no code path with the
  main detector, must agree with the reported cadence. On the sample clips these
  fire on exactly the one recording where segmentation genuinely fails.

## Layout

Three top-level pieces: **`app/`** is the pilot UI, **`src/gaitscreen/`** is the
analysis library, and everything else is configuration, docs or data.

```
app/                       the Streamlit pilot platform, self-contained
  main.py                    entry point — streamlit run app/main.py
  shared.py                  config, database access, widgets used by several views
  views/                     one module per page: analyse, trends, calibration,
                             limitations

src/gaitscreen/            the analysis library — no UI code
  config.py  types.py        config tree; dataclasses + coordinate conventions
  version.py                 ALGO_VERSION, stamped on every session
  io/                        video probing, frame iteration, camera-motion estimation
  pose/                      landmark schema, Tasks-API extraction, pixel conversion
  signal/                    gap handling, filters, sub-frame event timing
  segmentation/              direction and passes, Zeni events, gait cycles
  features/                  the six metrics, joint angles, session orchestration
  flagging/                  absolute thresholds, personal baselines, trend rules
  quality/                   quality score, speed feasibility, assistive-device proxy
  calibration/               scale/homography model, point picking, drift check
  storage/                   SQLite schema, repository, per-session artifacts
  reporting/charts.py        figures, shared by the app and the CLI
  pipeline.py  cli.py        stage composition and the command line

config/default.yaml        every threshold; nothing clinical is hardcoded
docs/limitations.md        what to distrust and why — the app renders this directly
docs/original-brief.md     the specification this was built from
sample_video/              four 25 fps test clips (the .mp4s are gitignored, so
                           they are local-only and not part of a deployment)
tests/                     137 tests; fixtures/synthetic.py generates known gait

requirements.txt           runtime dependencies (requirements-dev.txt adds pytest)
packages.txt               system libraries for Streamlit Cloud — required
.streamlit/config.toml     upload limit and theme, committed for the deployment

models/                    holds the pose model, downloaded on first use (ignored)
data/                      database, uploads, artifacts, created at runtime (ignored)
```

Two things worth knowing about this layout:

`app/` contains no analysis. It calls the same pipeline as the CLI, so a result
seen in the app and one from `gaitscreen analyze` are the same numbers. The page
modules live in `views/` rather than `pages/` because Streamlit treats a `pages/`
folder beside the entry script as an automatic multi-page app and builds its own
navigation alongside the one here.

`tests/fixtures/synthetic.py` is load-bearing rather than incidental. Without a
motion-capture reference, generating landmark trajectories from *known* stride
times is the only way to show the pipeline measures variability rather than
manufacturing it from sampling noise.

## Sample videos

The four clips in `sample_video/` are 768×432 @ 25 fps. MediaPipe tracks all of
them at 100% detection with 0.85–0.98 ankle visibility, including the pure-black
silhouette. But the subject translates only 3–8% of frame width in each (a real
overground pass is 60–90%) — they are treadmill/in-place or camera-tracked, and
one is a visible tracking shot with 382 px of background pan.

So they serve as a **negative fixture for gait speed** (all four must be refused)
and a **positive fixture for timing metrics and angle curves**. Because looped
stock footage has near-identical strides, the variability measured from them is
close to a pure measurement-noise floor — a useful regression bound.
