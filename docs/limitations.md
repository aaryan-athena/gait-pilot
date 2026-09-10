# Known limitations

This is a **screening and trend-monitoring tool, not a diagnostic instrument.**
It is designed to flag concerning *changes* in one person's walking over time.
Every limitation below is a reason to distrust an absolute value, not a reason to
distrust a trend — but several of them constrain the trend too, and those are
marked.

## 1. Frame rate limits stride-time variability

Stride-time variability (the coefficient of variation of stride time) is the most
fall-risk-predictive of the metrics, and the most fragile.

Elderly stride time is roughly 1.1 s with a CV of 2–3%, so the standard deviation
being measured is about **25–35 ms**. One frame is 40 ms at 25 fps and 33 ms at
30 fps. Taking the integer frame index of each gait event would therefore
quantise away the entire signal, and the reported "variability" would be a
property of the sampling grid.

Two things address this:

- **Sub-frame event refinement** (parabolic interpolation through each extremum
  and its neighbours) is always applied. Tests in `tests/test_events.py` hold the
  refined timing error below 8 ms at 25 fps, against >15 ms for integer indexing.
- **The metric is marked low-confidence below `video.fps_variability_min`**
  (default 50 fps).

**Record at 60 fps if you care about this metric.** At 25–30 fps, treat variability
as indicative and rely on the other metrics.

## 2. Framing and stride count trade off against each other

This is the single most common reason a pilot recording cannot be measured, and
it is a geometric constraint rather than a software one.

Precision depends on how many pixels tall the subject is: landmark error is
roughly a fixed number of pixels, so a small subject means large relative error.
But a well-framed subject — filling half the frame — has a stride length of
roughly 1.4 leg lengths, which covers a third of the frame width. Two or three
strides and they have crossed it.

So a single walk-past at good framing yields **two to five strides**, where ten
are needed before stride-time variability means anything. Zooming out to fit more
strides makes the subject too small and degrades every measurement instead.

The only resolution is **two to three passes back and forth within one
recording**. This is not optional advice; a single short pass cannot produce a
variability figure at any framing. It also happens to fix the asymmetry bias in
limitation 7, since each leg gets to be the near leg.

Configured minimum is `features.min_strides_for_cv` (default 10); below that the
CV is reported as null rather than as a misleadingly precise number, and the
recording diagnostics say how much more walking is needed.

How binding this is, measured on the pilot set of 47 phone recordings: the median
clip yielded **4 usable strides** and exactly **one clip of 47 reached ten**. So
under a single-pass protocol, stride-time variability — the metric the geriatric
literature rates most highly for fall risk — is effectively never available. This
is the one limitation that a change in recording practice, rather than a change in
software, actually removes.

## 3. Gait speed needs a static camera and real forward travel

Speed is derived from image-space displacement, so it is meaningless when the
displacement does not correspond to forward travel:

- treadmill or in-place walking — the subject never translates;
- a camera that pans to follow the subject — the translation is cancelled;
- a handheld or zooming camera — the scale changes mid-measurement.

A naive pipeline still emits a number in all three cases, and a **small number is
exactly the tool's high-risk signal**, which would then enter the personal
baseline permanently. So speed is refused with a stated reason instead. The
checks are subject translation across the frame (`speed.min_subject_translation_frac`)
and background optical flow (`speed.max_camera_motion_frac`).

All four sample clips in `sample_video/` fail this check by design — they are the
negative test fixture.

## 4. Two-point calibration is approximate

A single metres-per-pixel scalar is only strictly valid in a plane parallel to the
sensor at the depth of the marked points. Consequences:

- Mark the reference **along the walking path**, not across it.
- The scale is derived on the floor plane, but speed is measured from hip motion
  about a metre above it, so there is a systematic error that grows as the camera
  gets closer to the subject.
- Use the **4-point floor homography** (`--method homography`) where the operator
  can manage it; it removes the perspective error along the walk.

Because this is a trend tool, a consistent bias matters far less than a varying
one — which is why the next item exists.

## 5. A moved camera mimics a real decline — *affects trends*

Reusing one calibration across sessions assumes the camera does not move. Over 90
days in a home, tripods get nudged, raised, and relocated. A moved or zoomed
camera rescales every distance-based metric while leaving timing untouched, which
is the exact signature of genuine gait slowing.

Each session therefore re-checks the subject's pixel height against the value
stored at calibration. A deviation over 10% suppresses metric distances and asks
for recalibration.

## 6. Single sagittal view biases left/right asymmetry

Asymmetry is a ratio, so it needs no calibration — but it is not occlusion-free.
The **far limb is hidden behind the near limb** through much of stance, so it is
systematically noisier. In the sample clips the far arm was untracked in 240 of
250 frames.

Protocol fix, not an algorithm fix: record **one pass in each direction**, so each
limb is the near limb once, and compare near-limb measurements. Sessions store
`camera_side` per pass to support this.

## 7. No frontal view means no lateral trunk sway

Lateral (side-to-side) trunk sway needs a frontal camera. This deployment has none,
so the metric is **anterior-posterior trunk lean from the sagittal view**
(`features.trunk_sway_axis: sagittal_ap`), stored as `trunk_ap_sway_norm`. It is a
substitute, not the same measurement, and should not be compared against
published lateral-sway norms.

Two unsynchronised video files also cannot be fused per gait cycle — there is no
common clock. If a frontal camera is added later, it must be treated as an
independent pass reporting session-level amplitude only.

## 8. Assistive devices cannot be detected from pose alone

MediaPipe Pose is a body-keypoint model. **A cane or walker is not in its output**,
and a wrist held low is indistinguishable from an arm at rest. What is observable
is suppressed and asymmetric arm swing, which is a weak proxy — and in a sagittal
view the far arm is usually occluded, so the asymmetry half of it is often
unavailable.

Therefore: `assistive_device` in session metadata, entered by whoever recorded the
walk, is **authoritative**. The arm-swing heuristic may only lower confidence. It
never asserts a device and never reports "no device detected".

## 9. Personal baselines from few sessions are unstable — *affects trends*

An SD estimated from 3–5 sessions is very noisy, so a "1.5–2 SD" threshold does
not mean what it appears to. Mitigations:

- minimum 5 sessions (`flagging.baseline.min_sessions`);
- robust centre and spread (median and 1.4826 × MAD) so one bad session cannot
  redefine normal;
- a **minimum-detectable-change floor** per metric, so a deviation must exceed the
  tool's own test-retest error as well as the SD multiple. The shipped MDC values
  are placeholders — measure them on your own capture setup;
- direction-aware flagging: only deterioration flags, not improvement;
- low-confidence sessions are excluded from baselines (but still stored and shown).

Note that with six metrics each tested at ~1.5 SD, the per-session probability of
at least one flag is substantial. That is a deliberate choice — the brief asks for
sensitivity over specificity, since a missed decline is worse than a false alarm —
but it means the flag list should be read as "look at this", not "something is
wrong". The `confirm_window` setting escalates a deviation seen in 2 of the last 3
sessions to `confirmed`, to help triage without losing sensitivity.

## 10. A 90-day window is too short at monthly cadence — *affects trends*

At monthly screening, 90 days is three sessions. The baseline window is therefore
`max(window_days, window_min_sessions)`, not days alone.

## 11. Changing the algorithm creates a fake trend break — *affects trends*

Any change to filtering or segmentation changes the *measurement*. Mixing
algorithm versions within one person's history produces a step change that is
indistinguishable from clinical decline.

Guards: `ALGO_VERSION` is stamped on every session, raw landmarks are retained so
history can be re-derived without re-running pose estimation, and
`gaitscreen sessions` warns when a user's history spans more than one version.
**Reprocess a user's full history after any algorithm change.**

## 12. Double-support time reads high — *needs local calibration*

Measured double support on the sample clips sits around 31% of the gait cycle
for healthy adults walking normally, where the literature expects roughly 20–25%.
The synthetic fixture, where the true value is known to be 20%, is recovered
correctly, so the arithmetic is right — the bias is in event timing from 2D
markerless landmarks, most likely the heel reaching its anterior maximum slightly
before actual contact.

Consequence: the illustrative absolute threshold (>30% = high concern) currently
fires on healthy adults. That threshold has deliberately **not** been widened to
make the sample data pass, because tuning a clinical constant to fit four stock
videos would be worse than leaving it visibly wrong. Measure the offset on your
own setup and set the threshold from that. The *trend* in double support is
unaffected, since a consistent bias cancels when comparing a person to
themselves — which is the tool's actual purpose.

## 13. Assistive-device asymmetry cannot be judged from one side

The arm-swing asymmetry heuristic is now only applied when both wrists are
confidently tracked (mean visibility ≥ 0.8). On ordinary sagittal footage the far
arm is foreshortened and intermittently hidden behind the torso, which shrinks
its measured swing for reasons unrelated to a walking aid; before this gate, the
heuristic produced ratios above 2.5 and a suspicion note on *every* unaided
recording. The bilateral-suppression test (a walker holds both arms still) is
view-robust and still runs.

## 14. Thresholds are illustrative

Every clinical constant in `config/default.yaml` — the 0.6 m/s high-risk speed, the
CV cutoffs, the MDC values — is a placeholder drawn from general reading. **Review
all of them against current geriatric literature, and validate against your own
population, before any real-world use.**

## 15. What the tool tells you about a bad recording

Earlier versions reported symptoms — "only 3 valid strides were recovered" —
which left whoever recorded the video guessing at the cause, so the next
recording failed the same way. Every session now also carries **recording
diagnostics**: measured properties of the video paired with the specific change
that fixes each one. They appear above the metrics in the app, and in
`gaitscreen analyze` output.

What is checked, and what each is measured from:

| Diagnostic | Measured from |
| --- | --- |
| Subject too small | subject pixel height / frame height, and leg length in px |
| Subject not in frame | fraction of frames with every gait joint inside the frame |
| Feet clipped | foot landmarks at or past the bottom edge |
| Oblique camera angle | horizontal shoulder separation vs the width implied by trunk height |
| Legs poorly visible | lower-body landmark visibility relative to torso |
| Unsteady foot tracking | frame-to-frame heel jitter, as a fraction of leg length |
| Tracking switched person | single-frame body displacement beyond what walking allows |
| Camera shake | background path length beyond any steady pan |
| Walk too short | usable strides against the number needed |
| Frame rate too low | reported fps |

Two limits worth knowing:

**The camera-angle estimate has a floor.** Pose estimation infers the position of
the hidden far shoulder, so a genuinely side-on recording reads 9–23° rather than
0° on pilot footage. The threshold (30°) is set above that floor, which means the
check reliably catches badly oblique views but will not flag a mildly angled one.
Treat a reading under 30° as "not obviously wrong" rather than as confirmation.

**What fired, and how often.** On the 47-clip pilot set: walk too short 98%,
subject not fully in frame 66%, subject too small 60%, legs poorly visible 57%,
camera shake 26%, tracking switched person 15%, oblique angle 2%, low frame rate
2%. The near-universal one is not a miscalibrated check — the median clip really
did yield 4 strides against 10 needed — but it does mean the *ranking* is what
makes the feedback usable: as the top-priority fix, "walk too short" led on only
10 of 47 clips, behind subject size (17) and framing (16).

The clothing check is the threshold most likely to need adjusting for a different
population, since it depends on local dress. At 0.85 it flagged 57% of this set,
which reflects a group where several people wore loose kurtas and flowing
trousers rather than a fault in the check.

**Thresholds are calibrated on one pilot set** — 47 phone recordings, 832×464 to
1920×1080, mostly 60 fps. They are engineering limits rather than clinical ones,
but they are still specific to that camera and setting, and should be re-checked
against yours. Where a threshold could not be set from observed data, the config
says so inline.

## 16. The annotated video shows the analysis, not the raw tracking

The skeleton drawn on the playback is the **filtered, gap-filled** trajectory --
the same one the measurements were taken from, which is what makes it a fair
check on those measurements. It is not a raw dump of what pose estimation
returned frame by frame.

Two consequences to keep in mind when using it to judge a recording:

- Joints whose position was *interpolated* across a short dropout are drawn
  hollow rather than solid, so an inferred position never looks like an observed
  one. Long dropouts are not interpolated at all and simply leave the limb
  undrawn -- a leg segment vanishing for a while is the far knee being occluded,
  which is normal in a side-on view.
- Smoothing means the drawn skeleton is slightly steadier than the raw
  detections were. If tracking looks good in the video but the foot-jitter
  diagnostic still fires, trust the diagnostic: it is measured before smoothing.

## 17. A deployed fix is not necessarily a running fix

Streamlit Community Cloud reloads the entry script when source changes but does
**not** re-import modules already in `sys.modules`. It restarts the process only
when *dependencies* change. Because the whole pipeline lives in an imported
package, a source-only fix can sit on disk, unloaded, while the old code keeps
running and the old error keeps appearing.

The tell in the deploy log is a `Pulling code changes` / `Updated app!` pair with
**no `Stopping...` and no `Uvicorn server started`** between them. Compare a real
restart, which shows both.

Two things follow:

- After pushing a source-only change, **reboot the app** (Manage app → Reboot).
  Pushing alone is not enough.
- The sidebar shows the git revision, algorithm version and encoder revision of
  the code actually loaded, and video failures quote the same string. If it is
  older than what you deployed, the process needs rebooting rather than the code
  needing another change.

## 18. `z` coordinates are not used

MediaPipe's `z` is a depth estimate relative to the hip midpoint in units that are
neither metric nor reliable. It is archived for completeness and never computed
from.
