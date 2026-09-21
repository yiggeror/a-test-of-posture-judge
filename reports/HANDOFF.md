# Handoff

State of the project, what changed this phase, and what is actually open.

---

## Situation at the start of this phase

**Correction.** An earlier version of this document stated that the previous
phase's work had never been pushed and that everything here was rebuilt from
the handoff prose alone. That was wrong, and the error was mine.

The previous phase's work is on **`claude/posture-assessment-demo-q1x8nj`**
(`a26fdd0`): five commits carrying `app.py`, `posture/`, `reports/HANDOFF.md`,
`reports/PHASE0/2/3.md`, `scripts/triage_images.py`, 74 test images and
`testdata/sources_full.csv` — exactly what the handoff described.

I missed it because I ran `git branch -a`, which lists only refs this clone
had already fetched, and the clone had not fetched that branch. The
authoritative check is `git ls-remote --heads origin`, and I did not run it
until the user pushed back. **Use `git ls-remote`, not `git branch -a`, to ask
what exists on a remote.**

The rebuild still happened, because the reconstruction was largely complete
before the branch surfaced. What changed afterwards is documented below under
"What the previous phase's work then contributed" — it was not wasted; it
supplied the test set that overturned this phase's headline result.

### A branch-naming conflict worth knowing about

The previous handoff says "所有开发和推送都在 `...-demo-q1x8nj` 这个分支上，
不要推别的分支". This session was instructed by its own harness to develop on
`claude/posture-assessment-reliability-2jazt2`. I followed the harness and did
not touch the other branch. The two branches have therefore diverged from
`7130ed2` and **nobody has merged them**. That is a decision for the repo
owner, not something this session should have resolved unilaterally.

## What is different about this environment

Outbound network is open. Concretely:

- `images.cocodataset.org` → 200. **This is the important one.**
- Flickr (`*.staticflickr.com`) → 200, and the `_b` (1024px) size variant
  exists where COCO only ships 640px.
- `commons.wikimedia.org` → 200.
- `www.pexels.com` → 403 to non-browser clients.

---

## The main thing that changed: stop looking at images

The previous phase measured a ~0.1% hit rate searching stock galleries by eye
(~1400 Pexels images inspected, ~350 Wikimedia with zero hits) and correctly
concluded this was content scarcity rather than a network limitation. That
conclusion still holds — open network does not make the photographs exist.

The way out is not a better gallery. It is to search a corpus that ships
*annotations*. COCO labels every person instance with 17 keypoints carrying
per-point visibility flags, including both ears and both ankles. Every
property that previously had to be judged by eye is decidable from that
vector:

| judged by eye before | decidable from annotations |
|---|---|
| is the ear visible | `ear` visibility flag == 2 |
| are the feet in frame | both `ankle` flags != 0 |
| is this a side view | shoulder x-separation ÷ shoulder-to-ankle height |
| are the arms hanging free | wrist y below hip y |
| is it one person | second instance's area ÷ subject's area |
| is the subject standing | hip-to-ankle span ÷ head-to-ankle span |

`scripts/coco_mine.py` applies these to 268,030 annotated person instances
and only downloads the hits. Pool sizes it produces:

| view | all licenses | redistributable (ids 4,5,7,8) |
|---|---|---|
| side | 47 | 5 |
| front | 114 | 33 |

**One trap worth not repeating.** Requiring *both* ankles to be `V_VISIBLE`
structurally excludes lateral views: in a true side view the far ankle is
occluded by the near leg, so annotators mark it `V_OCCLUDED`, not visible.
The same applies to the far shoulder and far hip. With that predicate, side
hits were **0**. Relaxing it to "one ankle visible, the other at least
localised" took it to **33**. The filter was excluding exactly what it was
built to find.

This is not a solved problem — 47 side candidates is better than 2 and still
not enough (see Open questions). But the search is now a query, so widening
it is a matter of relaxing predicates rather than of looking at more pictures.

---

## What the previous phase's work then contributed

Once the missed branch surfaced, its 74-image set was merged in under
`testdata/pexels/`. It was not redundant with the COCO set — it was the
control the COCO set could not provide, and it changed three conclusions.

**1. It exposed the facing bug.** The Pexels side views are almost all
left-facing, so they produced a *consistently* positive rotation slope where
−1.0 was expected, instead of the near-zero average a mixed set gives. A
clean sign error is obvious; a cancelled average looks like noise. Without a
single-facing set the bug would have stayed in the report.

**2. It supplied the studio control.** The open question this phase left —
"is the error rate the tool or the images?" — needed deliberately shot
photographs, and I had written that none existed in the project. They did:
730–1029px subjects, plain backgrounds, camera level. Measured on them,
landmark noise halves and every sagittal metric except `forward_head` reaches
a 0% gross-error rate.

**3. Its 62-image negative set is the real false-positive benchmark.** Far
richer than the 14 non-person COCO images I had assembled — statues,
mannequins, dolls, hand and foot close-ups, bending, sitting, animals, crowds
— and it comes with the previous phase's own measured leak rate (18/63, 29%),
so the comparison is like-for-like. It also caught a guard gap: a seated
subject classified `front`, where the sagittal knee guard does not apply, was
measured without complaint. Fixed.

It also let two of the previous phase's specific documented failures be
re-tested rather than taken on trust:

| previous phase's finding | this phase |
|---|---|
| ratio-based view test called two 53°/58° yaw shots "front" and produced shoulder-tilt readings on them | both classified oblique and blocked — but the margin is thin, one sits at spread 0.217 against a 0.22 threshold |
| sock/calf close-up "passed every geometric check", output 「+32.84°，明显倾向」 | blocked by three guards |
| hair-over-ear photo output 「头前引角 +25.39°，明显倾向」 | reads +16.5 ±11.6 → 「测量精度不足以判定」; still not *blocked*, but the harm is gone |

---

## The second thing: reliability is now measured

This was the stated blocker and it turned out not to require the missing side
photographs at all.

Some image transformations have a mathematically known effect on the true
answer, so they provide ground truth with no labelled data:

- **Rotation by θ** changes every angle-from-vertical and
  angle-from-horizontal by exactly θ. The posture is untouched; only the
  camera's idea of "down" moved.
- **Scale, JPEG, brightness, contrast** do not change the true answer at all.
- **Horizontal flip** negates frontal readings exactly and preserves sagittal
  ones.

`scripts/repeatability.py` runs these. Results are in
[`RELIABILITY.md`](RELIABILITY.md). Short version:

- **All nine metrics recover a known rotation to within 0.05 of theory**, on
  two independent image sets. The geometry and the sign conventions are right.
- What separates them is the residual. **Span drives it**: the two
  shortest-span metrics (`forward_head`, `knee_deviation`) sit at 22–25%
  gross error on candid photography, the long-span ones at 1–5%.
- **Image quality drives it too**: on studio photography everything except
  `forward_head` reaches 0%, and landmark noise halves.
- Guard leak rate on the previous phase's own 62-image negative set:
  **6%, against the 29% they recorded.**

### Three measurement mistakes worth recording

The first version of that harness reported a rotation slope of **−0.25** where
−1.00 was expected, which reads as the tool almost completely failing to
track rotation. Three separate errors, each of which looked like a finding
about the tool:

1. **Pooling different failure modes.** After a transform the detector
   sometimes locked onto a *different person* in the frame — on a 193px-tall
   subject, the post-transform "subject" was a bystander, with landmarks 1.98
   body-lengths away. Those comparisons dominated the fit. The harness now
   maps landmarks back through the known transform and rejects comparisons
   where the subject changed, reporting re-acquisition rate separately from
   reading precision. They are different problems and pooling them describes
   neither.

2. **One expected slope for two planes.** PIL rotates counter-clockwise for a
   positive angle, which *decreases* an angle from vertical but *increases*
   an angle from horizontal. Frontal metrics therefore have expected slope
   **+1.0**, not −1.0. Before that fix, correct frontal behaviour
   (+0.98) looked like catastrophic failure.

3. **Pooling both facing directions.** This one produced a headline that had
   to be retracted. `angle_from_vertical` multiplies the horizontal offset by
   `anterior`, so a **left-facing subject's sagittal reading is the negative
   of a right-facing one's**, and a rotation moves it by **+θ instead of −θ**.
   On a mixed-facing set the two populations cancel: the sagittal
   least-squares slope collapsed to −0.05 and the residuals inflated. I wrote
   that up as "sagittal metrics have a 40–48% gross-error rate and are not
   trustworthy". It was arithmetic, not a property of the tool. After
   normalising by facing, the same data gives −0.96..−1.01 and 5–22%.

After all three fixes, every metric lands within 0.05 of its expected slope on
both image sets. Nine metrics, two planes, two independent collections — that
is strong evidence the geometry is right, and it validates the harness too.

**The general lesson, and the reason this is written up at length:** on a
harness like this, a result that looks like a dramatic failure of the system
under test is more often a failure to hold the comparison fixed. All three
bugs looked like findings. Two of them got written into a report before being
caught, and one of those reports was pushed.

---

## What calibration turned out to mean

There are no clinical labels anywhere in this project, so no threshold can be
validated against "has the condition". The most that is available is
norm-referencing: where a reading sits in the distribution of readings from
comparable photographs.

`scripts/reference_distribution.py` does that and applies two refusals:

- **n < 20** → keep the hand-picked threshold and its `guess` tag.
- **proposed cut < the metric's own p90 measurement error** → refuse the
  percentile threshold entirely. A cut point below the noise is a coin flip
  with a number attached.

Current outcome:

| metric | threshold source |
|---|---|
| `shoulder_tilt`, `pelvis_tilt`, `head_tilt`, `head_vs_shoulder_tilt` | `population-percentile`, n=91 |
| `forward_head`, `shoulder_protraction`, `knee_deviation`, `head_over_hip` | still `guess` — n=11 |
| `trunk_sway` | still `guess` — refused, its p90 error (10.3°) exceeds the proposed cut (7.1°) |

A concrete result: the hand-picked `shoulder_tilt` threshold was 2.0°, the
measured 80th percentile is 6.0°. The guess would have flagged most people.

---

## Open questions

### 1. Is `forward_head` worth keeping at all?

This replaces the previous open question ("is the sagittal error rate the tool
or the images?"), which the Pexels set largely answered: **both, and the
metric's span decides which dominates.**

On candid COCO photography `forward_head` is wrong by more than a verdict band
22% of the time and `knee_deviation` 25%, while the long-span sagittal metrics
sit at 5%. On studio photography everything except `forward_head` reaches 0%.
The pattern is consistent and mechanical — `forward_head` measures the
shortest span in the metric set (ear→shoulder, ~9 cm) using the noisiest
landmark in it (the shoulder, ~3× the ear's noise).

It already cannot resolve its own thresholds (±11.6° against 10°/18°), so the
app refuses it a verdict. Meanwhile `head_over_hip` measures the same anatomy
— where the head sits relative to the body — without the shoulder, over four
times the span, at ±2.1°, with the best rotation residual of any sagittal
metric on both sets.

**The open call: promote `head_over_hip` from diagnostic to the primary
forward-head reading, and either retire `forward_head` or keep it only as a
diagnostic.** I did not make that change unilaterally because "头前引角" is
the metric the product promises and renaming it is a product decision. But
the measurement says the tool currently leads with its worst reading and
hides its best one.

### 2. Side-view data is still short

47 candidates, ~35 classified as side views, 9 passing all guards. Percentile
thresholds need n≥20 and ideally n≥100. Untried avenues:

- **Relax the mining predicates.** `multiple_people` (27,934) and
  `incomplete_annotation` (24,482) are the biggest rejection buckets and both
  are conservative.
- **Other annotated corpora.** MPII Human Pose ships activity labels
  including standing categories; licensing needs checking before any image is
  committed.
- **Synthetic rendering.** Still never tried, and it is the only route to
  *true* ground truth on absolute angles rather than on deltas. Licensing of
  the body model matters: SMPL/SMPL-X are research-only.
- **Ask the user to photograph themselves.** Also the only route to real
  test-retest repeatability (below).

### 3. Real repeatability is still unmeasured

Everything in `RELIABILITY.md` is a *noise floor*: the same photograph
perturbed. Real test-retest — the subject re-standing, the photographer
re-framing, clothing shifting — is strictly larger and needs someone to
photograph the same person several times. No number for it exists.

### 4. Statues, mannequins, dolls

Unresolved and I do not think geometry can resolve it. Both the pose model and
the independent EfficientDet check call them persons because they are
human-shaped. The independent detector *did* add real value for the palm/sock
class of failure — one non-person COCO image produced a hallucinated skeleton
and all three guards caught it — but that is a different problem. A texture or
material classifier is the plausible next step; it is a real project, not a
heuristic tweak, and whether it is worth it depends on how often the case
actually occurs in use.

### 5. Metric design

`forward_head` and `shoulder_protraction` are still coupled through the shared
shoulder landmark. `head_over_hip` now exposes the coupling but does not
remove it. Whether to report the pair as one combined sagittal finding rather
than two independent ones is an open design call.

---

## Constraints observed

- Models are Apache-2.0 only (MediaPipe Pose Landmarker, EfficientDet-Lite0).
  No YOLO-pose (AGPL-3.0), no OpenPose (academic).
- No fabricated landmarks, results or provenance. Every committed image has
  its source URL, COCO id and license in a `sources.csv`. Every number in
  `reports/` is regenerable by a script in `scripts/`.
- Every threshold constant carries a provenance tag; a test asserts that the
  ones tagged `guess` say so in words.
- User-facing wording is 参考 / 倾向, never 诊断. A test enforces this across
  the template and all user-visible strings — it caught a real slip where a
  diagnostic-only metric was labelled 「仅供诊断参考」.
- Failures are reported: acceptance rate, rejection reasons, gross-error
  rates and the harness bug above are all in the reports rather than only the
  favourable numbers.

## Running it

```bash
./scripts/setup.sh                                    # libs + venv + models + tests
./.venv/bin/python -m pytest tests/ -q                # 152 passed
./.venv/bin/python app.py                             # http://127.0.0.1:5000
./.venv/bin/python scripts/validate.py --dir testdata/front --save-overlays
./.venv/bin/python scripts/repeatability.py --dir testdata/front testdata/side
```

`setup.sh` installs `libegl1 libgles2 libgl1` — the mediapipe wheel links EGL
even for CPU-only inference and import fails without them.
