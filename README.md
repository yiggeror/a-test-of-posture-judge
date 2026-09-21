# a-test-of-posture-judge

Upload a full-body photograph, get reference readings on standing posture.

**This is not a medical device and does not diagnose anything.** It reports
measured angles with their uncertainty, and describes them as tendencies
(倾向) for reference (参考). Read [Known limitations](#known-limitations)
before believing any number it produces.

---

## Honest status

| | |
|---|---|
| Code | working end to end, 144 tests |
| Frontal-plane readings (front photo) | measured precision ~1-3°, thresholds derived from a measured distribution |
| Sagittal-plane readings (side photo) | **not trustworthy** — 40-48% of readings are off by more than a whole verdict band |
| Clinical validity | **none, for any metric** — no image anywhere in this project carries a clinical label |

The single most useful thing this repo now contains is
[`reports/RELIABILITY.md`](reports/RELIABILITY.md), which says how far each
reading moves when the posture does not. Those numbers were previously
unmeasured.

---

## Quick start

```bash
./scripts/setup.sh                              # system libs + venv + models + tests
./.venv/bin/python -m pytest tests/ -q          # 144 passed
./.venv/bin/python app.py                       # http://127.0.0.1:5000
```

`setup.sh` installs `libegl1 libgles2 libgl1`. That is not optional on a
headless box: the mediapipe wheel links against EGL/GLES even for CPU-only
inference, and importing it otherwise fails with
`libEGL.so.1: cannot open shared object file`.

Useful scripts:

```bash
# run the pipeline over a folder; prints acceptance rate and rejection reasons
./.venv/bin/python scripts/validate.py --dir testdata/front --save-overlays

# measure how much readings move under transforms that don't change the posture
./.venv/bin/python scripts/repeatability.py --dir testdata/front testdata/side

# rebuild the norm-reference distribution the thresholds come from
./.venv/bin/python scripts/reference_distribution.py --dirs testdata/front testdata/side

# find more candidate photos (see "Test data" below)
./.venv/bin/python scripts/coco_mine.py --ann-dir <coco-annotations> --view side --out cand.csv
./.venv/bin/python scripts/fetch_from_manifest.py --manifest cand.csv --out-dir <dir>
```

---

## What it measures

Which metrics are available depends on the camera view, which the tool
determines itself from the projected shoulder separation.

**Front view**

| metric | meaning |
|---|---|
| `shoulder_tilt` | shoulder levelness |
| `pelvis_tilt` | lateral pelvic obliquity |
| `head_tilt` | head side-tilt |
| `head_vs_shoulder_tilt` | head tilt relative to the shoulders — **camera roll cancels in this one** |
| `lateral_head_shift` | diagnostic only, no verdict |

**Side view**

| metric | meaning |
|---|---|
| `forward_head` | ear relative to shoulder |
| `shoulder_protraction` | shoulder relative to hip ("rounded shoulders") |
| `trunk_sway` | whole-body forward/backward lean |
| `knee_deviation` | + flexed / − hyperextended |
| `head_over_hip` | diagnostic only — ear relative to hip, **does not use the shoulder landmark** |

Every reading is reported as `value ± uncertainty`. The uncertainty is not a
decoration: it is propagated from the measured landmark noise through the
specific segment being measured, and when it crosses a threshold the tool
reports the band as undetermined instead of picking a side.

### Two deliberate design changes from the original metric set

**"Pelvic tilt" was removed.** Anterior/posterior pelvic tilt is defined
clinically by the ASIS–PSIS line. BlazePose emits one approximate hip-*joint*
centre per side and no pelvic landmarks at all, so any number shipped under
that name would have been a rescaled hip position wearing a clinical label.
`trunk_sway` replaces it and measures what the landmarks can actually support.
`pelvis_tilt` in the table above is *lateral* obliquity only, and is named so.

**`head_over_hip` was added.** The original set's forward-head and
rounded-shoulder metrics are not independent — they share the shoulder
landmark, with opposite sign, so a mislocated shoulder pushes one up and the
other down. `head_over_hip` skips the shoulder entirely, so when the two
disagree it tells you which landmark is responsible.

---

## Known limitations

These are measured or confirmed, not hypothetical.

1. **Sagittal readings have a high gross-error rate.** Rotate a photo by a
   known angle and the true change in every angle is exactly that angle.
   Under that test, 40–48% of sagittal readings are wrong by more than 5°,
   which is the width of an entire verdict band. Frontal readings are 1–14%.
   See [`reports/RELIABILITY.md`](reports/RELIABILITY.md).

2. **Thresholds for every sagittal metric are still guesses.** They are
   tagged `guess` in `posture/thresholds.py` and shown as such in the UI. The
   frontal thresholds are derived from a measured distribution and tagged
   `population-percentile`.

3. **No clinical validity, at all.** There is no labelled data in this
   project. The best available calibration is norm-referencing — where your
   reading sits among other photographs — and the reference population is
   unscreened general photography, not a screened cohort. Unusual is not the
   same as unhealthy.

4. **Camera roll is indistinguishable from a real shoulder height
   difference.** A degree or two of handheld roll adds directly to
   `shoulder_tilt` and `pelvis_tilt`, and is the same size as the effect being
   measured. `head_vs_shoulder_tilt` is the one frontal reading immune to it.

5. **An oblique camera angle silently compresses sagittal readings**, and the
   reading itself gives no sign of it. The tool blocks oblique views rather
   than reporting them.

6. **Upright statues, mannequins and dolls are accepted as people.** Both the
   pose model and the independent object detector call them persons, because
   they are human-shaped. Geometry cannot separate them. No heuristic here
   attempts to.

7. **MediaPipe's `visibility` / `presence` scores are not evidence.** A
   landmark placed on hair covering an ear scored 1.000 while being wrong by
   enough to produce a spurious +25° forward-head reading. No guard in this
   codebase reads those channels, and a test enforces that.

8. **Acceptance rate on real-world photos is low.** Over the mined COCO side-view
   set, 19% of images produced any verdict; the rest were blocked, mostly for
   a flexed knee or a leaning trunk — i.e. the subject was mid-stride rather
   than standing neutrally. That is the guards working, but it means candid
   photographs mostly do not work.

---

## Test data

`testdata/` holds only images whose COCO license id permits redistribution
*and* derivative works (4 = CC BY 2.0, 5 = CC BY-SA 2.0, 7 = no known
copyright restrictions, 8 = US Government Work). Per-image provenance —
source URL, COCO image id, license — is in `sources.csv` in each folder.
Nothing in this repo is of unrecorded origin.

| folder | n | purpose |
|---|---|---|
| `testdata/front/` | 33 | frontal metrics |
| `testdata/side/` | 5 | sagittal metrics |
| `testdata/negative/` | 14 | images containing no person — regression tests for the palm/sock false-positive failures |

The larger working set used for the primary reliability measurement includes
NonCommercial-licensed COCO images, which are used for statistics only and are
**not** committed. It is reproducible from a clean checkout by re-running
`coco_mine.py --all-licenses` followed by `fetch_from_manifest.py`; the mining
is deterministic.

### How the candidates were found

The previous phase searched stock galleries by eye and measured a ~0.1% hit
rate for usable side-view standing photographs (~1400 Pexels images, plus ~350
Wikimedia images with zero hits). That is content scarcity, not a network
limitation, so better connectivity does not fix it.

COCO changes the economics because it ships *annotations*: every person
instance carries 17 keypoints with per-point visibility flags, including both
ears and both ankles. The properties that previously had to be judged by eye —
is the ear visible, are the feet in frame, is this a side view, are the arms
hanging free — are all decidable from the annotation vector. So instead of
viewing N images to find 0.001·N candidates, `coco_mine.py` filters 268,030
annotated person instances programmatically and only downloads the hits.

One trap worth recording: requiring *both* ankles to be `V_VISIBLE`
structurally excludes lateral views, because in a true side view the far ankle
is occluded by the near leg and annotators mark it `V_OCCLUDED`. Fixing that
one predicate took side-view hits from 0 to 33.

---

## Licensing

Models are Apache-2.0 only:

- MediaPipe Pose Landmarker (BlazePose) — pose estimation
- MediaPipe EfficientDet-Lite0 — independent "is there a person" check

Excluded by policy, not oversight: **YOLO-pose** (AGPL-3.0) and **OpenPose**
(academic license). COCO annotations are CC BY 4.0.

---

## Layout

```
app.py                 Flask single-page app
posture/
  geometry.py          pure angle maths, no vision deps, exactly testable
  landmarks.py         MediaPipe Tasks wrapper (the mp.solutions.* API is gone in 1.0.x)
  view.py              front/side/oblique classification, facing direction
  metrics.py           the measurements, and uncertainty propagation
  thresholds.py        every cut point, with a provenance tag and its basis
  guards.py            when NOT to answer
  assess.py            orchestration, verdict assembly, advice
  overlay.py           skeleton drawing
scripts/
  setup.sh
  coco_mine.py               find candidates by annotation, not by eye
  fetch_from_manifest.py     download with provenance recording
  validate.py                pipeline over a folder + acceptance statistics
  repeatability.py           the reliability measurement
  reference_distribution.py  norm-reference percentiles for thresholds
reports/
  RELIABILITY.md             what the numbers are worth  <- read this
  HANDOFF.md                 state and next steps
  repeatability.json         raw measurements
  reference_distribution.json
  landmark_noise.json        feeds the ± on every reading
```
