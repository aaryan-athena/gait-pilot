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

## 7. Lateral trunk sway needs a towards-camera recording

Lateral (side-to-side) trunk sway cannot be seen from the side: it happens
along the axis a sagittal camera projects away. The default deployment is
sagittal, so the sway metric on a normal session is **anterior-posterior trunk
lean** (`features.trunk_sway_axis: sagittal_ap`), stored as
`trunk_ap_sway_norm`. It is a substitute, not the same measurement, and should
not be compared against published lateral-sway norms.

**Partly addressed in ALGO_VERSION 0.2.0.** A clip filmed towards the camera is
now recognised and measured on its own terms, and real lateral sway
(`trunk_lateral_sway_norm`) is one of the two things it yields — the other
being step width, which a side view also cannot see. See item 18 for what such
a recording gives up in exchange, which is most of the rest of the measurement.

Two unsynchronised video files still cannot be fused per gait cycle — there is
no common clock. A towards-camera recording is therefore treated as a separate
session measuring different quantities, never as a second channel merged into
a sagittal one.

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

## 12. Double-support time is the least trustworthy metric — *still needs local calibration*

Double support is the share of the cycle with both feet down. It is derived
from toe-off, and toe-off is the hardest of the four events to see in 2D:
a heel strike is a clear arrival, whereas a toe leaving the ground is a
gradual loss of contact with no sharp landmark signature.

**Fixed in ALGO_VERSION 0.2.0.** Toe-off was previously taken from the toe's
anterior minimum (the Zeni rule). That rule is badly conditioned for toe-off,
because the pelvis travels forward over a planted foot for the whole of
stance, so the pelvis-relative toe position slides downward continuously and
has no minimum anywhere near the moment the foot actually lifts. It landed
late and grew later the slower the walk, which is why the effect was worst on
exactly the frail, slow gait the tool exists to watch. Toe-off is now measured
as the end of ground contact, from foot speed normalised to leg lengths per
stride; a planted foot measures near zero and a swinging foot around three,
so the separation is wide and pace-independent.

Effect on the pilot clips:

| clip | stride | stance | double support (0.1.x → 0.2.0) |
| --- | --- | --- | --- |
| normal pace, left | 1.16 s | 58.1% | 32.2% → **16.4%** |
| normal pace, right | 1.11 s | 58.9% | — → **18.5%** |
| limping, left | 1.56 s | 65.2% | — → **30.2%** |
| slow, left | 1.96 s | 65.8% | 44.0% → **31.3%** |
| slow, 2 rounds | 2.28 s | 68.9% | 47.4% → **37.6%** |

Stance now measures 58–59% of the cycle at a normal pace against a textbook
60%, and correctly measures more when the walk is slow or limping.

**Residual limitation.** At a normal pace the values now sit slightly *below*
the 20–25% the literature reports, rather than far above it. The remaining
error is in the same place it always was — event timing from markerless 2D
landmarks — and it has not been tuned away, because fitting a clinical
constant to a handful of pilot videos would be worse than leaving a known
offset visible. Measure the offset on your own setup before relying on the
absolute threshold. The *trend* remains the usable signal, since a consistent
bias cancels when comparing a person to themselves.

**Stored sessions from 0.1.x are not comparable with these values** and must
be reprocessed from retained raw landmarks (see item 11).

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

## 17. `z` coordinates are not used

MediaPipe's `z` is a depth estimate relative to the hip midpoint in units that are
neither metric nor reliable. It is archived for completeness and never computed
from.

## 18. A towards-camera recording measures something else entirely

Everything in the sagittal pipeline assumes the camera stands side-on to the
walk, so the direction of travel lies in the image plane. A clip filmed towards
the camera — the person walking at the lens and away again — breaks that
assumption completely, and the failure is quiet rather than loud.

**The danger is not that it crashes.** It does not. The old code ran happily on
the pilot coronal clip and reported a step-length asymmetry of 7.8% and a
cadence of 44 steps/min, at a quality score of 0.93 with no low-confidence
mark. The true cadence is about 80. Both numbers came from peaks in a signal
that was mostly projection artefact. In a screening tool, a confident wrong
number is worse than a missing one, so the camera angle is now identified
before any sagittal metric is computed.

**What is refused.** On a coronal clip, gait speed, step length, step-length
asymmetry, step-time asymmetry, double support and stride-time variability are
all reported as unavailable with a reason. Heel strike and toe-off are not
detected at all: the Zeni rule reads anterior position, which is precisely the
axis that has been projected away.

**What is measured instead.** Step width (`step_width_norm`) and lateral trunk
sway (`trunk_lateral_sway_norm`), neither of which a side view can produce —
one leg hides the other, and side-to-side motion is projected away. Cadence and
mean stride time are also recovered, from the vertical bob of each ankle
relative to the pelvis, and accepted only when the two legs independently agree
on the period. On the pilot clip that gave 80 steps/min across four passes with
the legs agreeing to within 0.02 s.

**Stride-time variability is deliberately withheld**, not merely absent. The
period is stable enough to average over a pass, but individual heel strikes
cannot be located precisely enough in this view to time one stride against the
next, and a CV built from imprecise event times measures the detector rather
than the person — while being read as the fall-risk signal that stride-time
variability is.

**How the view is detected.** By comparing how far the subject travels *across*
the image against how much their apparent *size* changes. A person crossing the
frame stays at a near-constant distance; a person walking at the camera does
the opposite. On the pilot clips the ratio separates the two cases by roughly a
factor of forty (coronal 3.4, sagittal 0.016–0.085), so both thresholds sit far
from anything observed.

The earlier shoulder-separation check is kept, but is not the gate. It compares
observed shoulder width against the width implied by trunk height, which needs
an assumed ratio between the two — and on the pilot footage that assumption
cost it most of its range: a subject walking straight at the camera measured
51° off side-on where the truth is nearer 90°, simply because their build did
not match the constant. Under-reading in the one case that matters makes it
unfit as a gate, though it remains useful for reporting foreshortening on
genuinely oblique clips.

**Not validated: diagonal walks.** The pilot set contains side-on clips and one
towards-camera clip, and nothing in between. Clips between the two thresholds
are classified `oblique` and sent down the sagittal path with the existing
foreshortening warning, which is the conservative choice — but where exactly a
diagonal walk stops being measurable has not been established against real
footage.

**Sessions filmed at different angles are never mixed.** They measure different
things by different means, so pooling them into one baseline or one trend line
would produce a step change that is the camera moving rather than the person —
the same failure mode as mixing algorithm versions (item 11), and just as
indistinguishable from real decline after the fact. `view_kind` is stored on
every session, baselines are restricted to matching sessions, and the trends
page shows one angle at a time. Sessions stored before `view_kind` existed are
treated as sagittal, which is what they were.

**The right fix is still to move the camera.** A coronal recording is reported
as a blocker-level recording problem, because the two measures it adds do not
come close to replacing the six it costs.


## 19. Phones skip frames, and the frame count cannot see it — *fixed in ALGO_VERSION 0.4.0*

A phone does not deliver a frame every 1/60 s. When the scene is dim it
cannot expose each frame in time, so it skips some and writes nothing where
they would have been. The container then reports an *average* frame rate. On
the garage pilot clips, nominally 60 fps, that average ranged from 43 to 57
fps: 2.7–28% of frames were never recorded, in bursts of up to 7.

Until 0.4.0 the pipeline counted decoded frames at that average rate. Over a
whole clip that is correct; inside it, it is wrong everywhere the skips cluster.
Measured against the camera's own timestamps it misplaced events by up to
1.8 s and mis-timed individual strides by 5–20% — several times the
stride-time variability being measured. On a metronome-paced walk, where true
variability is near zero, it reported a stride-time CV of **11.8%**, above the
5% line this tool treats as high fall risk.

**The fix.** Every frame carries a timestamp from the camera. The nominal frame
interval is measured from those timestamps (not assumed: several pilot phones
shoot at 59.94, and treating that as 60 misplaces a frame every 16 seconds),
each decoded frame is placed on its true row, and each skipped frame becomes a
NaN row — the representation already used for a briefly hidden foot. Short
gaps are filled; gaps longer than `landmarks.max_interpolation_gap_frames`
split the clip, exactly as a long occlusion does.

Metronome clips, same landmarks, old clock vs rebuilt clock:

| clip | frames skipped | stride CV before → after | cadence (truth) |
| --- | --- | --- | --- |
| 100 steps/min | 28.0% | 11.8% → **1.9%** | 99.8 (100) |
| 80 steps/min | 19.8% | 6.2% → **3.9%** | 81.5 (80) |
| 60 steps/min | 5.2% | 7.3% → **4.1%** | 57.8 (60) |

The free walks matter more, because they are what the tool will actually see.
At skip rates of only 2.6–8.9%, the old clock put all four healthy walks above
the 3% "moderate risk" line for stride-time CV, and one above 5%:

| clip | frames skipped | stride CV before → after |
| --- | --- | --- |
| walk 1 | 6.2% | 4.5% → **1.7%** |
| walk 2 | 2.6% | 4.6% → **1.5%** |
| walk 3 | 5.3% | 3.8% → **1.4%** |
| walk 4 | 8.9% | 6.3% → **2.7%** |

Cadence and double support barely move, being averages. Clips that skipped
nothing are unchanged: all eight earlier pilot clips reproduce their previous
results to within 0.1.

**Skipped frames are not treated as tracking failures.** The per-event
confidence gate rejects a stride when the foot was not observed around its
heel strike, because a hidden foot is still reported — guessed — and the guess
is confidently wrong. A skipped frame is different: the samples either side are
genuine, and a few missing frames of a signal with nothing above 6 Hz fill
almost exactly. On the metronome clips the strides the gate would have rejected
for skipped frames alone matched the metronome as well as the rest, and
rejecting them cost up to a quarter of the usable strides. They are still drawn
hollow in the annotated video, because nothing was measured there.

**Residual error.** The camera's own capture times jitter by a few
milliseconds, so no uniform grid fits them exactly; each frame is placed within
half a frame interval (8 ms at 60 fps) of when it was captured. That is small
next to a 25–35 ms stride-time standard deviation.

**Thresholds.** Skipped frames are mentioned on the report above 2% and mark
stride-time variability low-confidence above 30%, just above the heaviest clip
validated (28%). Beyond that nothing has been tested.

**Not reproduced: heel-strike detection firing late.** The review that found
this bug also reported, from 24 hand-labelled heel strikes on the 60 steps/min
clip, that about 30% fired 8 or more frames late. Two independent checks do not
reproduce it. Against the moment the heel landmark itself stops moving, Zeni
heel strikes land within about a frame on all three metronome clips, with 0–5%
late by 8+ frames. And MediaPipe's own temporal smoothing, which a person
watching the video would see but a landmark-based check would not, delays the
heel landmark by about one frame against per-frame detection — a near-constant
bias, not a miss. Neither check can rule out the tracker placing the heel
wrongly in the first place, which only labels made by eye can test. The labels
behind the original finding should be compared against the current detector
directly before this is either fixed or dismissed.

**Stored sessions from before 0.4.0** on clips that skipped frames carry the
distorted timing and must be reprocessed. Their raw landmarks were stored one
row per decoded frame with no timestamps, so this needs the original video.
