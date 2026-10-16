# Known limitations

This is a **screening and trend-monitoring tool, not a diagnostic instrument.**
It flags *changes* in one person's walking over time. Most limits below are
reasons to distrust an absolute value rather than a trend; those that also
affect trends are marked.

## 1. Frame rate limits stride-time variability

Elderly stride-time SD is about 25–35 ms — roughly one frame at 25–30 fps.
Gait events are refined to sub-frame precision (under 8 ms error at 25 fps in
`tests/test_events.py`), but the metric is still marked low-confidence below
`video.fps_variability_min` (default 50 fps). **Record at 60 fps.**

## 2. One walk-past is too short — record two or three passes

A well-framed subject crosses the frame in two or three strides, but
variability needs ten (`features.min_strides_for_cv`); below that it is
reported as null. Zooming out to fit more strides makes the subject too small
and degrades every other metric. On the 47-clip pilot set the median clip gave
4 strides and only one reached ten. **Walk back and forth two or three times in
one recording** — this also lets each leg be the near leg (item 5).

## 3. Gait speed needs a static camera and real forward travel

Treadmill or in-place walking, a camera panning to follow, or a handheld or
zooming camera would all produce a falsely low speed — which is the tool's
high-risk signal. Speed is refused with a reason in these cases
(`speed.min_subject_translation_frac`, `speed.max_camera_motion_frac`). The
clips in `sample_video/` fail this check by design.

## 4. Calibration is approximate, and a moved camera mimics decline — *affects trends*

Two-point calibration is only exact at one depth: mark the reference along the
walking path, and prefer the 4-point floor homography (`--method homography`)
where possible. A consistent bias cancels out in a trend; a changing one does
not. A nudged or zoomed camera rescales distances but not timing — the same
signature as genuine slowing — so each session re-checks the subject's pixel
height against calibration and suppresses distances if it differs by over 10%.

## 5. A side view biases left/right asymmetry

The far limb is hidden behind the near one for much of stance (the far arm was
untracked in 240 of 250 frames on the sample clips), so it is noisier. Record
one pass in each direction; `camera_side` is stored per pass to support this.

Measured on the 10 metronome clips in `data/CACDemo_Final`: healthy walks read
3–12% step-length asymmetry. Whichever foot is nearer the camera reads a step
several percent longer, flipping sign with walking direction, and each leg
contributes only about eight steps, so the same walk re-encoded read 2.7% and
then 6.5%. Neither effect can be removed with one camera. Since 0.6.0 the value
is still shown, but a flag needs the difference to clear the threshold at the
lower 95% bound given the step-to-step spread: the two healthy clips that had
flagged at 11–12% no longer do, and the simulated limp (33%) still flags high.

## 6. Walking aids cannot be detected

MediaPipe Pose does not see a cane or walker. The `assistive_device` field
entered by whoever recorded the walk is **authoritative**; the arm-swing
heuristic may only lower confidence, and never asserts or rules out a device.
Its asymmetry test runs only when both wrists are confidently tracked
(visibility ≥ 0.8); the bilateral-suppression test always runs.

## 7. Personal baselines are noisy — *affects trends*

An SD from a handful of sessions is unreliable, so baselines use: at least 5
sessions over `max(window_days, window_min_sessions)`; median and MAD rather
than mean and SD; a minimum-detectable-change floor per metric (shipped values
are placeholders — measure your own); flags on deterioration only; and no
low-confidence sessions. With six metrics each tested at ~1.5 SD, some flag is
common by design — read the list as "look at this", not "something is wrong".
A deviation in 2 of the last 3 sessions is escalated to `confirmed`.

## 8. Algorithm changes create fake trend breaks — *affects trends*

A change in processing changes the measurement, and looks like decline in a
person's history. `ALGO_VERSION` is stamped on every session, raw landmarks are
retained, and `gaitscreen sessions` warns on mixed versions. **Reprocess a
user's full history after any algorithm change.** Sessions from before 0.2.0
(double support) and before 0.4.0 (clips with skipped frames — needs the
original video) are not comparable with current values.

## 9. Double support is the least trustworthy metric

It depends on toe-off, the hardest gait event to see in 2D. Since 0.2.0
toe-off is taken from foot speed, which brought stance to 58–59% of the cycle
at normal pace (textbook: 60%). Double support now reads slightly *below* the
20–25% literature range; this offset has deliberately not been tuned away.
Measure it on your own setup before relying on the absolute threshold — the
trend remains usable.

## 10. Phones skip frames — *fixed in 0.4.0*

In dim light phones drop frames (2.7–28% on the 60 fps pilot clips, in bursts
of up to 7). Counting frames at the average rate mis-timed strides by 5–20% and
inflated stride-time CV — 11.8% on a metronome walk whose true value was ~2%.
Frames are now placed by the camera's timestamps, with skipped frames treated
as short gaps (long gaps split the clip). Residual error is under half a frame
(8 ms at 60 fps). Skips are reported above 2% and make variability
low-confidence above 30%; nothing beyond 28% has been tested.

**Open question:** a review reported ~30% of hand-labelled heel strikes firing
8+ frames late. Landmark-based checks do not reproduce this; it needs comparing
against those original labels.

## 11. A towards-camera recording measures something else

Side-view metrics are meaningless when the person walks at the lens — earlier
versions reported confident wrong values (cadence 44 against a true ~80). The
view is now detected first, from sideways travel versus change in apparent
size. A towards-camera clip yields step width, lateral trunk sway, cadence and
mean stride time; speed, step length, asymmetries, double support and
stride-time variability are withheld. Diagonal walks are unvalidated: they are
classed `oblique` and treated as side views with a foreshortening warning.
Sessions from different angles are never mixed in baselines or trends. **The
better fix is to move the camera.**

## 12. Trunk sway from the side is front-to-back

Lateral sway is invisible from the side, so normal sessions report
front-to-back trunk lean (`trunk_ap_sway_norm`). Do not compare it with
published lateral-sway norms. Real lateral sway needs a towards-camera
recording (item 11); two separate videos cannot be fused.

## 13. Recording diagnostics have calibrated limits

Each session lists measured problems with the video and the fix for each:
subject too small or out of frame, feet clipped, oblique angle, legs poorly
visible, unsteady foot tracking, tracking switched person, camera shake, walk
too short, low frame rate. Thresholds come from one pilot set of 47 phone
recordings — re-check them on yours. The camera-angle estimate reads 9–23° even
when truly side-on, so the 30° threshold only catches clearly oblique views.
The leg-visibility check depends on clothing (it flagged 57% of a group
wearing loose kurtas).

## 14. The annotated video shows the filtered skeleton

Interpolated joints are drawn hollow; long dropouts are left undrawn (normal
for the far knee). The drawing is smoothed, so if tracking looks fine but the
foot-jitter diagnostic fires, trust the diagnostic.

## 15. Thresholds are illustrative

Every clinical constant in `config/default.yaml` — the 0.6 m/s high-risk speed,
the CV cutoffs, the MDC values — is a placeholder. **Review them against
current geriatric literature and validate on your own population before any
real-world use.** MediaPipe's `z` depth estimate is unreliable and is never
used.
