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

A four-page web UI for pilot testing: upload a video, get results, save the
session to a person's history, and view trends.

It is built to **show its working**, because the failure mode that matters in a
pilot is a plausible-looking wrong number. The results are written for two
readers, in the order each needs: a caregiver gets a plain verdict, six
plainly-named measures and the annotated video, while a clinician or tester gets
the same session's exact metric names, event plots and diagnostic numbers one
expander down.

- **Plain language throughout.** "Step-to-step consistency", not "stride-time
  coefficient of variation". Each measure says what it is for and which
  direction is good, since a reader cannot infer that from a number. No measure
  is ever described as abnormal — a screening tool cannot diagnose, so its
  vocabulary does not imply that it has.
- **The annotated video** replays the clip with the tracked skeleton and the
  detected heel strikes and toe-offs drawn on. Everything the tool reports rests
  on the skeleton following the right joints and the events landing at the right
  moments, and this is the only way a non-specialist can check either. Joints
  whose position was *inferred* across a dropout are drawn hollow rather than
  solid, so an estimate never looks like an observation.
- **Recording feedback** names what was wrong with the *video* and what to
  change — subject too small, camera not side-on, loose clothing hiding the
  legs, walk too short — ranked so the one change that helps most comes first.
- **Every unmeasurable metric states why**, in plain terms, and is grouped
  separately so a reader is not stepping over blanks to find the results.
- **Known measurement bias is disclosed where the number is shown.** Double
  support reads 8–10 points high from 2D video, so it carries that caveat on its
  card and is not allowed to headline the plain summary on the strength of an
  absolute threshold — only on a change across sessions, where a consistent bias
  cancels.

Pages: **Analyse a walk**, **Trends**, **Calibration** (only needed for gait
speed in m/s — everything else is scale-free), and **Limitations**.

### Deploying to Streamlit Community Cloud

Point it at this repository with `app/main.py` as the entry point. Three files
in the repo exist for that deployment and are needed there:

| file | why |
| --- | --- |
| `requirements.txt` | Python dependencies |
| `packages.txt` | system libraries plus `ffmpeg` — **required**, see below |
| `.streamlit/config.toml` | 400 MB upload limit for video files |

`packages.txt` is not optional, and the entries are not guesswork — they are
what the wheels' own ELF headers declare. Reading `DT_NEEDED` from the Linux
wheels gives the exact list:

- **mediapipe** needs `libEGL.so.1` and `libGLESv2.so.2`. Versions up to 0.10.18
  vendored their own copies via auditwheel; from 0.10.20 on they do not, so the
  OS must supply them.
- **opencv-contrib-python** (pulled in by mediapipe, which is why headless
  OpenCV in `requirements.txt` does not avoid any of this) needs `libGL.so.1`,
  `libglib-2.0.so.0`, `libgthread-2.0.so.0`, `libz.so.1`, and — through its Qt
  XCB platform plugin — `libX11`, `libXext`, `libxcb`, `libICE` and `libSM`.

`ffmpeg` is there for a different reason: the annotated playback video needs
H.264, and OpenCV's own H.264 writer cannot be relied on. At some frame sizes it
reports success, fails to load its encoder, and writes a kilobyte of nothing —
so frames are piped to ffmpeg instead. Without ffmpeg the app still works and
still renders the video, but as MPEG-4 Part 2, which most browsers will not play;
it then says so and offers a download rather than showing a dead player.

To re-derive the list after a dependency bump rather than discovering it one
failed deploy at a time:

```bash
pip download mediapipe --platform manylinux_2_28_x86_64 \
  --python-version 3.11 --only-binary :all: --no-deps -d /tmp/whl
# then read DT_NEEDED from each .so in the wheel and drop the libc/libstdc++
# entries and anything the wheel bundles itself
```

Three traps in that one small file, each of which produces a failed build rather
than a useful message:

- **Bare package names only — no comments, no padding.** Every line goes
  straight to `apt-get install`, so a `#` explanation becomes a list of
  nonexistent packages (`E: Unable to locate package Required,` …), and an
  apostrophe anywhere in it breaks `xargs` before apt even runs.
- **One bad name takes down the whole step.** apt installs nothing if any entry
  is unresolvable, so a wrong guess also loses the entries that were correct.
  Check `https://packages.debian.org/<suite>/<package>` first: a real package has
  a "Details of package" page, a virtual one does not.
- **Names are specific to the base image**, currently Debian trixie. Both GL and
  glib were renamed at some point: `libgl1-mesa-glx` became `libgl1` after Debian
  11, and `libglib2.0-0` became `libglib2.0-0t64` in the 64-bit `time_t`
  transition. The old names will not resolve on trixie.

**Reboot after a source-only push.** Streamlit Cloud reloads the entry script
when source changes but does not re-import modules already in `sys.modules`; it
restarts the process only when *dependencies* change. Because the pipeline lives
in an imported package, a pushed fix can sit on disk unloaded while the old code
keeps running. The tell in the deploy log is a `Pulling code changes` /
`Updated app!` pair with **no `Stopping...` and no `Uvicorn server started`**
between them — a real restart shows both. Use Manage app → Reboot.

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
pytest                                   # 181 tests, no video needed
gaitscreen probe sample_video/*.mp4      # what the sample clips will support
```

## Recording protocol

Ordered by what actually went wrong on real pilot recordings, not by theory. The
app reports which of these applied to each clip, so this list is the reference
rather than something to memorise.

1. **Two to three passes back and forth, in one recording.** The most common
   reason a session cannot be measured. There is a hard geometric trade-off:
   framed well, a person covers most of the frame in two or three strides, so a
   single walk-past yields 2–5 strides where 10 are needed for a variability
   figure. Zooming out to fit more strides makes the subject too small instead.
   Passes also let each leg be the near (unoccluded) one, which is what makes
   left/right asymmetry trustworthy from a single camera.
2. **Fill at least half the frame height with the person.** Landmark error is a
   roughly fixed number of pixels, so subject size sets the precision of
   everything downstream.
3. **Legs and ankles visible.** Loose or flowing trousers hide the knee and
   ankle, and the tracker then infers their position rather than seeing it —
   producing confident, wrong numbers rather than missing ones.
4. **Fixed camera, square to the walking path.** Tripod or propped, never
   handheld, never following the walker — a camera that tracks the subject
   cancels the displacement gait speed is computed from.
5. **Whole walk inside the frame**, feet above the bottom edge throughout.
6. **60 fps.** At 25–30 fps the frame interval is as large as the stride-time
   standard deviation being measured, degrading variability to indicative only.
7. **Only the walker in shot.** Bystanders can make the tracker switch person
   mid-recording.
8. **Don't move the camera between sessions.** If you must, recalibrate — the
   tool detects probable movement and suppresses distance metrics rather than
   reporting rescaled ones.
9. **Record assistive-device use in session metadata.** Pose estimation cannot
   see a cane.

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
- **Segmentation gates on the feet and pelvis, not the knees.** Requiring every
  gait joint continuously made the far knee — the most occluded landmark in a
  sagittal view, since it passes behind the near leg each stride — a single point
  of failure for the whole session. On pilot footage it was present for 60% of
  frames with a longest clean run of 1.4s, while every foot landmark had runs
  over six seconds. The knee is needed for angle curves, not for finding heel
  strike, so its absence now degrades the curves instead of discarding the
  recording. Combined with basing the minimum segment length on the *minimum*
  stride time rather than the maximum, this turned the pilot set from mostly
  unusable into mostly measurable: **31 of 47 real recordings (66%) produced no
  metrics at all before the change; 43 of 47 (91%) now yield at least three**,
  with cadence 74-129 steps/min and stride times 0.93-1.70s across the set.
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
