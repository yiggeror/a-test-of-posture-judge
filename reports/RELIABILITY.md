# Reliability

How far each reading moves when the posture does not.

Before this, the project had no number for this at all. The handoff described
the tool as *"a thermometer whose shell is finished — it gives readings, the
digits are precise, but it has never been compared against a reference, it
drifts by ±4 degrees, and it will happily give you a reading off a potato."*
This document measures the drift and the potato. It does not fix the missing
reference — see [What this cannot tell you](#what-this-cannot-tell-you).

Regenerate everything here with:

```bash
./.venv/bin/python scripts/repeatability.py --dir <image dirs> --write-noise
```

---

## How you measure precision with no labelled data

Nobody has told us the true forward-head angle of anyone in any photograph, so
accuracy cannot be measured. Precision can, because some image transformations
have a mathematically known effect on the true answer:

| transform | true effect on the reading | what a deviation means |
|---|---|---|
| **rotate by θ** | every angle-from-vertical and angle-from-horizontal changes by **exactly θ** | measurement error, with the known part removed |
| **rescale / JPEG / brightness / contrast** | **no change at all** | pure measurement error |
| **mirror** | frontal readings negate exactly, sagittal readings are preserved | measurement bias |

The rotation test is the strong one: it gives a *graded* ground truth over a
range of known values on any photograph whatsoever, so it detects a scale
error (slope ≠ 1) that an invariance test alone would miss.

**Measurement basis.** 130 images from the mined COCO reference set, 2470
transform comparisons. Frontal metrics n=742 rotation samples, sagittal n=260.

---

## 1. Subject re-acquisition

Before asking how far a reading moved, ask whether the detector was still
looking at the same person.

| outcome | count | share |
|---|---|---|
| same subject re-acquired | 2387 | 96.7% |
| **different subject locked on** | 19 | 0.8% |
| no detection at all | 64 | 2.6% |
| images skipped, subject < 250px | 15 | — |

This is reported separately on purpose. An earlier version of the harness
pooled these with the precision statistics and produced a rotation slope of
−0.25 where −1.00 was expected, which reads as near-total failure. It was not:
on one 193px-tall subject the post-transform "subject" was a bystander, with
landmarks 1.98 body-lengths away, and those comparisons dominated the fit.
Two different problems were being averaged into one meaningless number.

---

## 2. Invariance — transforms that must change nothing

RMS movement in degrees. Ideal is 0.

| metric | brightness | contrast | JPEG | rescale | pooled | p95 |
|---|---|---|---|---|---|---|
| `shoulder_tilt` | 1.47 | 1.61 | 1.07 | 0.99 | **1.24** | 2.30 |
| `lateral_head_shift` | 1.33 | 1.23 | 0.98 | 0.97 | **1.10** | 2.17 |
| `pelvis_tilt` | 3.40 | 1.57 | 1.49 | 1.47 | **1.98** | 3.22 |
| `head_over_hip` | 1.08 | 10.44 | 2.40 | 1.52 | **4.68** | 4.04 |
| `shoulder_protraction` | 2.39 | 9.89 | 2.77 | 3.13 | **4.90** | 3.75 |
| `head_tilt` | 4.47 | 7.38 | 4.12 | 5.18 | **5.29** | 5.96 |
| `head_vs_shoulder_tilt` | 4.59 | 7.57 | 4.29 | 5.29 | **5.43** | 6.61 |
| `trunk_sway` | 4.94 | 6.91 | 7.13 | 5.25 | **6.08** | 14.75 |
| `knee_deviation` | 8.91 | 8.15 | 10.13 | 9.01 | **9.18** | 14.20 |
| `forward_head` | 4.24 | 15.06 | 11.80 | 9.48 | **10.72** | 12.94 |

Re-saving a photograph at JPEG quality 70 moves `forward_head` by 11.8° on
average. Nothing about the person changed.

---

## 3. Rotation — a known, graded ground truth

Sagittal metrics (measured from vertical) must track **−1.000**; frontal
metrics (measured from horizontal) **+1.000**; metrics that are
rotation-invariant by construction **0.000**.

"LS" is the least-squares slope, "robust" the median of per-sample ratios.
"gross" is how often the residual exceeds 5°, which is the width of a whole
verdict band.

| metric | n | expect | LS | robust | median resid | p90 | **gross** |
|---|---|---|---|---|---|---|---|
| `shoulder_tilt` | 742 | +1.0 | 0.99 | **0.98** | 0.89° | 2.37° | **1%** |
| `lateral_head_shift` | 742 | n/a | −0.34 | −0.35 | 0.83° | 2.41° | **2%** |
| `pelvis_tilt` | 742 | +1.0 | 1.00 | **1.00** | 1.13° | 3.12° | **3%** |
| `head_tilt` | 742 | +1.0 | 0.99 | **1.01** | 1.99° | 5.69° | **12%** |
| `head_vs_shoulder_tilt` | 742 | 0.0 | 0.00 | **0.00** | 2.25° | 5.52° | **14%** |
| `knee_deviation` | 260 | 0.0 | 0.02 | **0.01** | 2.85° | 10.86° | **28%** |
| `shoulder_protraction` | 260 | −1.0 | −0.02 | −0.71 | 2.65° | 22.00° | **40%** |
| `head_over_hip` | 260 | −1.0 | −0.03 | −0.80 | 2.08° | 22.31° | **43%** |
| `forward_head` | 260 | −1.0 | −0.05 | −0.61 | 4.30° | 23.63° | **48%** |
| `trunk_sway` | 260 | −1.0 | −0.09 | −0.70 | 3.32° | 25.31° | **48%** |

Three things to read out of this table.

**The frontal metrics are genuinely good, and they validate the harness.**
Three independent metrics recover a known rotation to within 2%, and the two
rotation-invariant metrics correctly read 0.00 and 0.01. A broken measurement
chain does not produce that by accident.

**The sagittal metrics fail on the tail, not on the scale.** Their median
residual is small — `head_over_hip` is 2.08°, better than `head_tilt` — but
their p90 is above 22° and roughly 40–48% of readings are wrong by more than a
whole verdict band. The distribution is bimodal: mostly fine, catastrophically
wrong a large minority of the time. That is worse than being uniformly
imprecise, because nothing in the output distinguishes the two cases.

**The least-squares/robust gap is the finding, not an artefact.** Where they
disagree (every sagittal metric), the typical reading tracks a known rotation
well and a heavy tail does not. Reporting either number alone would describe a
tool that does not exist.

---

## 4. Mirror

Frontal readings must negate exactly; sagittal readings must be preserved.

| metric | n | RMS | max |
|---|---|---|---|
| `lateral_head_shift` | 95 | 1.79° | 8.33° |
| `shoulder_tilt` | 95 | 2.31° | 8.97° |
| `pelvis_tilt` | 95 | 2.85° | 8.64° |
| `shoulder_protraction` | 34 | 4.30° | 16.68° |
| `head_over_hip` | 34 | 5.34° | 22.87° |
| `head_tilt` | 95 | 5.43° | 20.15° |
| `head_vs_shoulder_tilt` | 95 | 5.83° | 18.98° |
| `forward_head` | 34 | 10.53° | 35.50° |
| `knee_deviation` | 34 | 15.86° | 60.53° |
| `trunk_sway` | 34 | 23.88° | **130.47°** |

A 130° mirror error on `trunk_sway` is a sign flip: the facing direction was
resolved the other way, which inverts the sagittal sign convention. The
`facing_ambiguous` guard warns about this but does not block it.

---

## 5. Landmark noise, and where the imprecision actually comes from

RMS displacement as a fraction of shoulder-to-ankle height, from the
invariance transforms. Overall across the critical landmarks: **0.0257**.

| landmark | noise |
|---|---|
| ear | ~0.0068 |
| nose / eye | ~0.0071 |
| hip | ~0.0140 |
| **shoulder** | **~0.0207** |
| ankle | ~0.0273 |
| wrist / hand | 0.050–0.067 |

**The ear is one of the most stable landmarks; the shoulder is three times
worse.** This overturns the natural reading of the earlier hair-over-ear
failure — the ear is not the weak point. `forward_head` is imprecise because
it measures across the *shortest span in the metric set* (ear→shoulder, ~9 cm)
using the *worst landmark in it*.

Uncertainty propagates as `hypot(σ_a, σ_b) / span`, so span dominates:

| metric | span | ± (1σ) |
|---|---|---|
| `head_over_hip` | 390px | **±2.1°** |
| `shoulder_protraction` | 300px | ±4.0° |
| `trunk_sway` | 430px | ±4.7° |
| `shoulder_tilt` | 220px | ±6.2° |
| `knee_deviation` | 215px | ±8.0° |
| `head_tilt` | 64px | ±8.2° |
| `forward_head` | 90px | **±11.6°** |

`forward_head` carries ±11.6° against thresholds of 10°/18°. It cannot resolve
its own bands. `head_over_hip` measures the same anatomy — where the head sits
relative to the body — without the shoulder and over four times the span, at
±2.1°.

**This is the most actionable result in this document.** If a forward-head
reading is wanted, `head_over_hip` is the one to use. `forward_head` as
defined is below the noise floor and the app now refuses to give it a verdict.

---

## 6. How much of this is the images rather than the tool

The obvious objection to section 3 is that the measurement set is hard:
mined COCO photography, 250–600px subjects, candid, often partly occluded.
Re-running the whole harness restricted to subjects at least 430px tall
separates that out.

| metric | gross @≥250px | gross @≥430px | p90 @≥250px | p90 @≥430px | robust @≥250px | robust @≥430px |
|---|---|---|---|---|---|---|
| `forward_head` | 48% | **36%** | 23.6° | 19.6° | −0.61 | **−0.87** |
| `head_over_hip` | 43% | **25%** | 22.3° | 18.1° | −0.80 | **−0.93** |
| `shoulder_protraction` | 40% | **27%** | 22.0° | 18.7° | −0.71 | **−0.88** |
| `trunk_sway` | 48% | **32%** | 25.3° | 24.4° | −0.70 | **−0.86** |
| `knee_deviation` | 28% | 29% | 10.9° | 10.1° | 0.01 | 0.06 |
| `head_tilt` | 12% | 11% | 5.7° | 5.1° | 1.01 | 1.04 |
| `pelvis_tilt` | 3% | 2% | 3.1° | 3.0° | 1.00 | 1.00 |
| `shoulder_tilt` | 1% | 1% | 2.4° | 2.4° | 0.98 | 0.99 |

(130 images / 2387 comparisons versus 56 images / 1039 comparisons.)

**Sagittal reliability depends strongly on subject size; frontal reliability
barely does.** Raising the size floor cuts sagittal gross errors by a quarter
to a half in relative terms and moves the robust slopes from about −0.7 to
about −0.9, while the frontal metrics move by a percentage point or less —
they were already near their ceiling.

Two conclusions, and the second matters as much as the first:

1. **A large part of the sagittal unreliability is input quality, and it is
   actionable.** The app now warns when the subject is under 430px
   shoulder-to-ankle (`subject_too_small`), with that threshold's provenance
   tagged `measured` and pointing here. A deliberately shot photograph will
   usually be far above it.

2. **It does not explain all of it.** Even at ≥430px, sagittal gross-error
   rates are 25–36% against 1–3% for the best frontal metrics, and the robust
   slopes still sit short of −1.0. Something intrinsic remains — plausibly the
   short measurement spans (section 5) and the fact that in a lateral view the
   far-side joints are occluded and their positions inferred. Raising image
   quality alone will not make the sagittal readings as trustworthy as the
   frontal ones.

This is a stratification of the existing set, not an experiment on good
photographs. It shows a direction convincingly; it does not predict where the
rate lands for a deliberately shot photo, because no such photographs exist in
this project.

---

## 7. What the app does about all this

- Every reading shows `value ± uncertainty`, propagated per landmark.
- When the error bar crosses a threshold, the band is reported as
  undetermined rather than rounded to a side.
- When the uncertainty is at least as wide as a whole band, the app reports
  **测量精度不足以判定** ("precision insufficient to judge") and offers no
  advice. At current precision this fires for `forward_head`, `trunk_sway`
  and `knee_deviation`.
- Frontal thresholds come from a measured distribution
  (`population-percentile`). Sagittal thresholds are still tagged `guess`,
  and `trunk_sway` is explicitly refused a measured threshold because its own
  error exceeds the proposed cut point.

Readings with real signal still resolve: `shoulder_protraction +37.5 ±3.5`
reports 明显倾向 with no hedging.

---

## What this cannot tell you

**These are lower bounds.** Everything above perturbs *the same photograph*.
Real test-retest variability — the subject re-standing, the photographer
re-framing, clothing shifting, a different time of day — is strictly larger
and is still unmeasured. It needs someone to photograph the same person
several times. No number for it exists.

**No accuracy, for any metric.** Precision is not accuracy. A reading can be
perfectly repeatable and consistently wrong, and nothing here would detect it.
Detecting it needs a reference measurement on a real body, which this project
has never had.

**No clinical meaning.** No image in this project carries a clinical label.
Thresholds derived from the reference distribution say where a reading sits
among other photographs. The reference population is unscreened general
photography. Unusual is not unhealthy.

**The test images are hard, and harder than the real use case.** The
measurement set is mined COCO photography: 250–600px subjects, candid, often
partly occluded, frequently mid-stride. Section 6 shows subject size accounts
for a substantial part of the sagittal error rate but not all of it. What
section 6 does *not* do is measure the tool on deliberately shot photographs,
because this project contains none. Extrapolating from a size stratification
of candid photos to "how it behaves on a photo you took on purpose" is a
guess, and it is the cheapest high-value experiment left. See `HANDOFF.md`,
Open question 1.
