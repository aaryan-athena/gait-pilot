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

## 2. A single short walk does not contain enough strides

A CV computed from 4–8 strides is unstable; the literature generally wants 12–30.
Configured minimum is `features.min_strides_for_cv` (default 10); below that the
CV is reported as null and low-confidence rather than as a misleadingly precise
number.

To get enough strides in a domestic hallway, record **multiple passes** in one
session. The first and last stride of each pass are excluded automatically
(acceleration and deceleration inflate variability), as are turns.

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

## 15. `z` coordinates are not used

MediaPipe's `z` is a depth estimate relative to the hip midpoint in units that are
neither metric nor reliable. It is archived for completeness and never computed
from.
