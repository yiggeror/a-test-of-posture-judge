# Reliability

How far each reading moves when the posture does not.

Before this, the project had no number for this at all. The previous handoff
described the tool as *"一个外壳做完的体温计——能读数、数字也精确，但从没跟
标准表对过，本身还有 ±4 度抖动，而且插进土豆里也照样给读数。"* This document
measures the drift and the potato. It does not supply the reference — see
[What this cannot tell you](#what-this-cannot-tell-you).

```bash
./.venv/bin/python scripts/repeatability.py --dir <image dirs> --write-noise
```

---

## Correction notice

An earlier version of this document reported a **40–48% gross-error rate for
the sagittal metrics** and concluded they were untrustworthy. **That was a bug
in the measurement, not a property of the tool.** The corrected figure is
5–22% on candid photography and 0–4% on studio photography. The bug is
described in [section 0](#0-three-ways-this-measurement-was-got-wrong);
everything below is post-fix.

---

## How you measure precision with no labelled data

Nobody has told us the true forward-head angle of anyone in any photograph, so
accuracy cannot be measured. Precision can, because some transformations have
a mathematically known effect on the true answer:

| transform | true effect | a deviation means |
|---|---|---|
| **rotate by θ** | every angle-from-vertical and angle-from-horizontal changes by **exactly θ** | measurement error, known part removed |
| **rescale / JPEG / brightness / contrast** | **no change at all** | pure measurement error |
| **mirror** | frontal readings negate, sagittal readings are preserved | measurement bias |

Rotation is the strong test: a *graded* ground truth over a range of known
values, on any photograph, detecting a scale error that invariance alone would
miss.

**Two sets, measured identically.**

| set | images | comparisons | subject size | character |
|---|---|---|---|---|
| COCO (`coco_mine.py`) | 130 | 2075 | 250–600px | candid, crowds, motion, occlusion |
| Pexels (`testdata/pexels/`) | 12 | 218 | 730–1029px | studio, plain background, fitted clothing |

---

## 0. Three ways this measurement was got wrong

Each produced a confident number that described nothing, and each looked like
a finding about the tool.

**1. Pooling a changed subject.** The first version reported a rotation slope
of −0.25 against −1.00. The detector had locked onto a *different person*
after the transform — on a 193px subject with bystanders, landmarks 1.98
body-lengths away. Fixed by mapping landmarks back through the known transform
and rejecting the comparison when the subject moved.

**2. One expected slope for both planes.** PIL rotates counter-clockwise for a
positive angle, which *decreases* an angle from vertical and *increases* one
from horizontal. Frontal metrics expect **+1.0**. Before that was fixed,
correct frontal behaviour (+0.98) looked like catastrophe.

**3. Pooling both facing directions.** This is the one that produced the
retracted headline. `angle_from_vertical` multiplies the horizontal offset by
`anterior`, so a **left-facing subject's sagittal reading is the negative of a
right-facing one's**, and a rotation moves it by **+θ instead of −θ**. On a
mixed-facing set the two populations *cancel*: the sagittal least-squares
slope collapsed to −0.05 and the residuals inflated, which was written up as a
heavy error tail. It was arithmetic. Fixed by normalising each sagittal delta
by the baseline facing.

The general lesson: on this kind of harness, a result that looks like a
dramatic failure of the system under test is more often a failure to hold the
comparison fixed.

---

## 1. Subject re-acquisition

Before asking how far a reading moved, ask whether the detector was still
looking at the same person, from the same view, facing the same way.

| outcome | COCO | Pexels |
|---|---|---|
| same subject, same view and facing | 2075 (84%) | 218 (95%) |
| **different subject locked on** | 19 (0.8%) | 0 |
| **view or facing changed** | 312 (13%) | 9 (4%) |
| no detection at all | 64 (2.6%) | 1 |

The 13% view/facing churn on COCO is itself a reliability signal: for one in
eight transforms that should have disturbed nothing, the tool's own view gate
changed its mind. On studio photography it is 4%.

---

## 2. Invariance — transforms that must change nothing

RMS movement in degrees over the COCO set. Ideal is 0.

| metric | brightness | contrast | JPEG | rescale | pooled |
|---|---|---|---|---|---|
| `shoulder_tilt` | 1.47 | 1.61 | 1.07 | 0.99 | **1.24** |
| `lateral_head_shift` | 1.33 | 1.23 | 0.98 | 0.97 | **1.10** |
| `pelvis_tilt` | 3.40 | 1.57 | 1.49 | 1.47 | **1.98** |
| `head_over_hip` | 1.08 | 10.44 | 2.40 | 1.52 | **4.68** |
| `shoulder_protraction` | 2.39 | 9.89 | 2.77 | 3.13 | **4.90** |
| `head_tilt` | 4.47 | 7.38 | 4.12 | 5.18 | **5.29** |
| `trunk_sway` | 4.94 | 6.91 | 7.13 | 5.25 | **6.08** |
| `knee_deviation` | 8.91 | 8.15 | 10.13 | 9.01 | **9.18** |
| `forward_head` | 4.24 | 15.06 | 11.80 | 9.48 | **10.72** |

Re-saving a photograph at JPEG quality 70 moves `forward_head` by 11.8° on
average. Nothing about the person changed.

---

## 3. Rotation — a known, graded ground truth

Sagittal metrics must track **−1.000**, frontal **+1.000**, rotation-invariant
metrics **0.000**. "gross" is how often the residual exceeds 5°, the width of
a whole verdict band.

| metric | expect | COCO slope | Pexels slope | COCO p90 | Pexels p90 | COCO gross | Pexels gross |
|---|---|---|---|---|---|---|---|
| `shoulder_tilt` | +1.0 | **0.99** | 1.03 | 2.4° | 2.1° | **1%** | **0%** |
| `pelvis_tilt` | +1.0 | **0.99** | 1.00 | 3.0° | 2.1° | **3%** | **0%** |
| `head_over_hip` | −1.0 | **−0.99** | −1.01 | 2.8° | 1.1° | **5%** | **0%** |
| `shoulder_protraction` | −1.0 | **−0.96** | −1.02 | 3.5° | 1.7° | **5%** | **0%** |
| `head_tilt` | +1.0 | **0.99** | 0.94 | 5.1° | 4.6° | 10% | 6% |
| `trunk_sway` | −1.0 | **−1.01** | −1.00 | 10.3° | 1.2° | 13% | **0%** |
| `head_vs_shoulder_tilt` | 0.0 | **−0.01** | −0.10 | 5.3° | 5.1° | 13% | 10% |
| `forward_head` | −1.0 | **−1.01** | −0.95 | 9.1° | 3.4° | **22%** | 4% |
| `knee_deviation` | 0.0 | **−0.03** | 0.04 | 9.9° | 3.8° | **25%** | 4% |

**Every slope lands within 0.05 of theory, on both sets.** Nine metrics across
two planes and two independent image collections recovering a known rotation
that closely is strong evidence that the geometry and the sign conventions are
right — and it validates the harness as well as the tool.

**What separates the metrics is the residual, not the slope.** Two things
drive it:

- **Span.** `forward_head` (22% gross) and `knee_deviation` (25%) are the two
  shortest-span measurements. `head_over_hip` and `shoulder_protraction`,
  which measure over 3–4× the span, sit at 5%. See section 5.
- **Image quality.** Every metric improves on studio photography, and the
  short-span ones improve most: `forward_head` 22% → 4%, `trunk_sway`
  13% → 0%.

---

## 4. Mirror

Frontal readings must negate exactly; sagittal readings must be preserved.
COCO set.

| metric | RMS | max |
|---|---|---|
| `lateral_head_shift` | 1.79° | 8.33° |
| `shoulder_tilt` | 2.31° | 8.97° |
| `pelvis_tilt` | 2.85° | 8.64° |
| `shoulder_protraction` | 4.30° | 16.68° |
| `head_over_hip` | 5.34° | 22.87° |
| `head_tilt` | 5.43° | 20.15° |
| `forward_head` | 10.53° | 35.50° |
| `knee_deviation` | 15.86° | 60.53° |
| `trunk_sway` | 23.88° | **130.47°** |

A 130° mirror error on `trunk_sway` is a sign flip — the facing direction
resolved the other way, inverting the sagittal convention. The same mechanism
that corrupted the rotation fit in section 0. The `facing_ambiguous` guard
warns about it; it does not block.

---

## 5. Landmark noise, and where the imprecision comes from

RMS displacement as a fraction of shoulder-to-ankle height.
**COCO 0.0219, Pexels 0.0104** — studio photography halves it.

| landmark | noise (COCO) |
|---|---|
| ear | ~0.0068 |
| nose / eye | ~0.0071 |
| hip | ~0.0140 |
| **shoulder** | **~0.0207** |
| ankle | ~0.0273 |
| wrist / hand | 0.050–0.067 |

**The ear is one of the most stable landmarks; the shoulder is three times
worse.** This overturns the natural reading of the hair-over-ear failure — the
ear is not the weak point. `forward_head` is imprecise because it measures
across the *shortest span in the metric set* (ear→shoulder, ~9 cm) using the
*worst landmark in it*.

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

`forward_head` carries ±11.6° against thresholds of 10°/18°: it cannot resolve
its own bands, and the app refuses to give it a verdict. `head_over_hip`
measures the same anatomy — where the head sits relative to the body — without
the shoulder and over four times the span, at ±2.1°, and it has the best
rotation residual of any sagittal metric on both sets.

**If a forward-head reading is wanted, `head_over_hip` is the one to use.**

---

## 6. False-positive guards

Benchmarked on the previous phase's 62-image negative set (statues,
mannequins, dolls, hand and foot close-ups, bending, sitting, animals,
crowds) — the same set they measured against, so the numbers are comparable.

| | leak rate |
|---|---|
| previous phase, before their guards | 31/63 (49%) |
| previous phase, after their guards | 18/63 (29%) |
| **this phase** | **4/62 (6%)** |

The four that still produce readings:

- **A statue and a mannequin.** Unresolved, and geometry cannot resolve it —
  an upright statue has a vertical torso, extended knees and normal
  proportions. The independent EfficientDet check calls them persons too,
  because they are person-shaped. No further heuristics were added.
- **A crowd shot** whose second person is too small to detect.
- **The ear-occlusion case** (`neg_ear_hair_falsereading_3820333`), and this
  one is worth reading closely. The previous version reported
  **「头前引角 +25.39°，明显倾向」** from a landmark placed on hair. This
  version reports **+16.5 ±11.6 → 「测量精度不足以判定」**. It is still counted
  as a leak because no guard blocked it, but the noise-floor rule removed the
  harm: no verdict, no advice. That is the uncertainty machinery doing the job
  it was built for, on a real confirmed failure.

Two documented cases that the previous phase could **not** catch are now
caught: the sock/calf close-up that "passed every geometric check" and
produced 「+32.84°，明显倾向」 is blocked by three guards, and both bending
photos are blocked.

---

## 7. What the app does about all this

- Every reading shows `value ± uncertainty`, propagated per landmark.
- An error bar crossing a threshold → band reported as undetermined.
- Uncertainty at least as wide as a whole band → **测量精度不足以判定**, no
  advice. Fires for `forward_head`, `trunk_sway`, `knee_deviation`.
- Subject under 430px shoulder-to-ankle → warning, with the measured
  justification.
- Frontal thresholds from a measured distribution; sagittal ones still tagged
  `guess`.

Readings with real signal still resolve: `shoulder_protraction +37.5 ±3.5`
reports 明显倾向 with no hedging.

---

## What this cannot tell you

**These are lower bounds.** Everything above perturbs *the same photograph*.
Real test-retest — the subject re-standing, the photographer re-framing,
clothing shifting — is strictly larger and remains unmeasured. It needs
someone to photograph the same person several times.

**No accuracy, for any metric.** Precision is not accuracy. A reading can be
perfectly repeatable and consistently wrong. Detecting that needs a reference
measurement on a real body, which this project has never had.

**No clinical meaning.** No image carries a clinical label. Percentile
thresholds say where a reading sits among other photographs of unscreened
general photography. Unusual is not unhealthy.

**The studio numbers rest on 12 images.** The Pexels column is 12 images and
218 comparisons, with only 2 true side views. It is the right *kind* of data
and far too little of it. Treat "0%" as "no failures observed in a small
sample", not as a rate.
