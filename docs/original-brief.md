# Project prompt for Claude Code — Elderly Gait Screening & Monitoring Tool

Copy everything below into Claude Code as your initial project prompt.

---

## Project goal

Build a video-based gait screening and monitoring tool for elderly users. This is a **screening/trend-monitoring tool, not a diagnostic instrument**. The goal is to flag concerning changes over time and give caregivers/clinicians a simple trend view — not to produce clinical-grade absolute measurements. Prioritize consistency and repeatability of measurement over precision.

## Tech stack

- Python 3.11
- `mediapipe` for pose landmark extraction (BlazePose, full-body model)
- `opencv-python` for video I/O
- `numpy` / `scipy` for signal processing (filtering, peak detection)
- `pandas` for session data and trend storage
- SQLite (or plain CSV/JSON files, whichever is simpler to start) for persistent per-user session history
- `matplotlib` or `plotly` for trend chart output
- Keep it a modular Python package, not a notebook — I want reusable modules I can later wrap in a CLI or simple web UI

## Pipeline to implement (in this order)

1. **Video ingestion & one-time calibration**
   - Accept a video file path (sagittal/side-on view as primary input; optional second frontal-view file)
   - One-time calibration step: user marks two points a known real-world distance apart (e.g. a floor line) at onboarding; store this scale factor per user/camera-setup, reused across sessions — don't recalibrate every session
   - Validate fps and resolution; warn if fps < 30

2. **Landmark extraction**
   - Run MediaPipe Pose on each frame
   - Store per-landmark (x, y, z, visibility) per frame
   - Drop or interpolate frames/landmarks below a visibility threshold (make this threshold configurable, default ~0.5)

3. **Temporal smoothing**
   - Apply a One-Euro filter (or Kalman filter — your choice, explain tradeoff) to landmark trajectories before any downstream computation
   - Make filter parameters configurable

4. **Gait cycle segmentation**
   - Detect heel-strike events heuristically from ankle vertical position/velocity local minima (per leg)
   - Segment the walk into individual gait cycles (heel-strike to heel-strike, same leg)
   - Normalize each cycle to 0–100% for cross-cycle comparison
   - Flag and exclude cycles with insufficient landmark confidence

5. **Feature extraction per session**
   Compute these five core metrics per session (this is the priority list — implement in this order):
   - **Gait speed** (m/s, using the calibration scale factor)
   - **Stride time variability** (coefficient of variation of stride time across all cycles in the session — this is the single most fall-risk-predictive metric per the geriatric literature, prioritize getting this right)
   - **Step length / step time asymmetry** (left vs right ratio — should be robust even without calibration)
   - **Cadence** and **double-support time** (% of gait cycle)
   - **Trunk sway amplitude** (lateral deviation of hip/shoulder midpoint, needs frontal view if available; skip gracefully if not)
   - Also compute and store hip/knee/ankle flexion-extension angle curves (0–100% of cycle) even though these aren't part of the core flagging logic yet — store them for future use and for generating a richer session report

6. **Session quality flag**
   - Store an overall data-quality score per session based on landmark visibility, number of valid cycles detected, and whether the person appears to be using a cane/walker or wearing loose/occluding clothing (a simple heuristic is fine here — e.g. persistent very low visibility on ankle/knee landmarks, or hand landmark near a vertical line to the ground, can be a proxy) — sessions below a quality threshold should be flagged as low-confidence, not silently included in trend calculations

7. **Flagging / alert logic** — implement both, either can trigger a flag:
   - **Absolute threshold check**: e.g. gait speed < 0.6 m/s = high risk, 0.6–1.0 m/s = moderate (make thresholds configurable constants, cite that these are illustrative and should be reviewed against current geriatric clinical literature before real deployment)
   - **Personal trend check**: maintain a rolling baseline (mean + SD) per metric per user, built from their first 3–5 sessions, updated with a rolling window (e.g. trailing 90 days); flag when a new session deviates beyond ~1.5–2 SD from that personal baseline
   - Bias toward sensitivity over specificity — a missed decline is worse than a false alarm
   - Log which trigger(s) fired and why, in plain language, alongside the raw numbers

8. **Data storage schema**
   One row/record per session: `user_id`, `date`, `video_source`, the 5 core metrics, the joint angle curve arrays, `quality_score`, `flags_fired` (list), `notes`. Design this schema first as its own module since everything else writes to it.

9. **Trend reporting**
   - Generate a simple per-user report: a trend line chart per core metric over time, with flagged sessions visually marked
   - Output as a static image or simple HTML — doesn't need to be fancy for v1

## Non-goals for this version

- No cane/walker-specific gait modeling — just flag those sessions as low-confidence for now
- No dual-task (cognitive load) gait testing
- No real-time/live camera processing — batch processing of recorded video files is fine
- No mobile app — a Python CLI or minimal local web UI is enough

## What I want from you first

Before writing code: propose the module/file structure for this package, and confirm the segmentation and flagging approach above make sense given the MediaPipe landmark model's actual output format — flag anything in this brief that seems technically off before you start implementing. Then build incrementally: calibration + extraction first, get that working on a sample video, before moving to segmentation and feature extraction.
