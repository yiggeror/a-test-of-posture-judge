# a-test-of-posture-judge

Upload a full-body photograph; get told, in plain words, which posture
problems it shows and what exercises help.

**This is not a medical device and does not diagnose anything.** Underneath the
plain answer is an instrument that measures angles with their uncertainty and
refuses to call a result it cannot resolve. Read
[Known limitations](#known-limitations) before believing any number it produces.

The user-facing app is in [`web/`](web/README.md): it runs the pose model in the
browser, so the photo never leaves the device.

---

## Honest status

| | |
|---|---|
| Code | working end to end, 210 tests; browser app verified identical to Python on 116 photos |
| Rotation tracking, all 9 metrics | **within 0.05 of theory** on two independent image sets |
| Response to a **real** posture change | slope **1.007** / **0.984** — does not under-report actual deviation |
| Frontal readings | reliable: 1–3% gross error, thresholds from a measured distribution |
| Sagittal readings, long-span (`head_over_hip`, `shoulder_protraction`) | reliable: 5% on candids, 0% on studio photos |
| Sagittal readings, short-span (`forward_head`, `knee_deviation`) | **weak** — 22–25% on candids; `forward_head` cannot resolve its own thresholds |
| False-positive guards | 6% leak on the 62-image negative set, against 29% previously |
| Thresholds | **5 of 6** judged metrics now use measured percentiles, including both sagittal ones |
| Clinical validity | **none, for any metric** — no image in this project carries a clinical label |

The single most useful thing this repo contains is
[`reports/RELIABILITY.md`](reports/RELIABILITY.md), which says how far each
reading moves when the posture does not. Those numbers were previously
unmeasured.

> **Correction.** An earlier version of this README reported a 40–48%
> gross-error rate for the sagittal metrics and called them untrustworthy.
> That was a bug in the measurement harness, not a property of the tool: it
> pooled left-facing and right-facing subjects, whose sagittal readings have
> opposite sign, so the two populations cancelled. Corrected figures are
> above; the bug is written up in `RELIABILITY.md` section 0.

---

## Quick start

```bash
./scripts/setup.sh                              # system libs + venv + models + tests
./.venv/bin/python -m pytest tests/ -q          # 210 passed
./.venv/bin/python scripts/build_web.py         # the user-facing app -> web/dist/ (verifies the JS port)
./.venv/bin/python api.py                       # JSON API,  http://127.0.0.1:5001
./.venv/bin/python app.py                       # engineering view with every reading, http://127.0.0.1:5000
```

Building a mini-program or mobile client? Read
[`reports/DEPLOYMENT.md`](reports/DEPLOYMENT.md) first — it has the measured
model sizes and timings, the on-device vs server trade-off, and the one
binding constraint that decides it (every threshold and uncertainty here is
tied to BlazePose's 33-point topology; swapping the pose model invalidates all
of them).

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

# measure the response to a KNOWN posture change (accuracy of a delta)
./.venv/bin/python scripts/synthetic_warp.py --dir testdata/pexels/side

# find more candidate photos (see "Test data" below)
./.venv/bin/python scripts/coco_mine.py --ann-dir <coco-annotations> --view side --out cand.csv
./.venv/bin/python scripts/fetch_from_manifest.py --manifest cand.csv --out-dir <dir>
```

---

## The app (`web/`) and what a person is told

`scripts/build_web.py` builds a single page: upload a photo, the pose model runs
in the browser, a scan animation plays, and the result is a short list of
problems — each with a one-line explanation, a likely cause and three exercises —
plus what looked fine. No angles, no error bars, no provenance tags: those stay
in the engineering view (`app.py`) and the API's `metrics` field.

The translation from instrument to plain answer is `posture/report.py`, with all
wording in `posture/consumer_zh.json` (shared by the API's `report` field and the
page). It is where the tool's caution has to survive, so the rule is explicit:
the person is told the **least severe** thing the reading's error bar cannot
rule out. A reading straddling slight/notable is reported as slight; one
straddling normal/slight as 临界 (borderline); one the instrument cannot resolve
is "not measured", never "fine"; a photo on which nothing could be judged is a
retake request, never "no problems found".

`web/posture.js` ports all of it. Parity is checked, not asserted: every build
runs Python and the port over every test photo (116 with a detection) and
compares guards, readings, bands **and the report**; any difference fails the
build. Widening that check from 6 samples to every photo caught a real
divergence — the old port skipped the knee/trunk neutrality guards when the
feet were out of frame, so half-body photos got a result in the browser that
Python rejects. Fixed to match Python. Details in [`web/README.md`](web/README.md).

## What it measures

Which metrics are available depends on the camera view, which the tool
determines itself from the projected shoulder separation.

**Which metrics carry a verdict was decided by measurement**, not by which
ones have familiar clinical names. Everything is computed and shown; only the
readings that survived the reliability work produce a finding.

**Side view** — reported

| metric | 中文 | ± | meaning |
|---|---|---|---|
| `head_over_hip` | 头部前移 | ±1.1° | ear relative to hip — **the primary forward-head reading** |
| `shoulder_protraction` | 圆肩 | ±1.4° | shoulder relative to hip |
| `trunk_sway` | 躯干前后倾 | ±2.0° | whole-body forward/backward lean |

**Front view** — reported

| metric | 中文 | ± | meaning |
|---|---|---|---|
| `lateral_head_shift` | 头部侧偏 | ±0.9° | head offset from the shoulder midline |
| `shoulder_tilt` | 高低肩 | ±1.3° | shoulder levelness |
| `pelvis_tilt` | 骨盆侧倾 | ±3.5° | lateral pelvic obliquity (often comes back unresolved) |

**Diagnostic only** — measured and shown, never a verdict

| metric | why |
|---|---|
| `forward_head` | ±2.6°, 22% gross error. Kept because reading it against `head_over_hip` isolates shoulder mislocalisation |
| `head_tilt` | 64px span, the shortest in the set |
| `head_vs_shoulder_tilt` | camera roll cancels in it, but it inherits `head_tilt`'s noise |
| `knee_deviation` | 25% gross error; used as a **neutrality guard**, not a finding |

Every reading is `value ± uncertainty`, propagated from measured per-landmark
noise through the specific segment. When the error bar crosses a threshold the
band is reported as undetermined; when it is as wide as a whole band the tool
says 测量精度不足以判定 and offers no advice.

### Three design changes, each forced by a measurement

**"头前引角" is no longer the headline reading.** Measured the obvious way
(ear vs shoulder) it is the *worst* thing this tool produces: shortest span in
the set, measured with its noisiest landmark, ±2.6° against 10°/18°
thresholds, 22% gross error. `head_over_hip` measures the same anatomy —
where the head sits relative to the body — without the shoulder and over four
times the span, at ±1.1°. It is now primary; `forward_head` is diagnostic.

**`lateral_head_shift` was promoted.** It started as a throwaway diagnostic
and turned out to measure best of anything here (±0.9°, 0% gross error).

**Anterior/posterior pelvic tilt is absent, deliberately.** It is defined
clinically by the ASIS–PSIS line and BlazePose emits neither landmark, only an
approximate hip-*joint* centre per side. A number under that name would have
been a rescaled hip position wearing a clinical label. `pelvis_tilt` above is
*lateral* obliquity only and is named so.

---

## Known limitations

These are measured or confirmed, not hypothetical.

1. **Short-span metrics have a high gross-error rate.** Rotate a photo by a
   known angle and the true change in every angle is exactly that angle. Under
   that test, on candid photography, `forward_head` is wrong by more than 5°
   (a whole verdict band) 22% of the time and `knee_deviation` 25%. The
   long-span metrics are at 1–5%. On studio photography everything drops to
   0–6%. See [`reports/RELIABILITY.md`](reports/RELIABILITY.md).

2. **One threshold is still a guess, and the measured ones rest on n=35.**
   `head_over_hip` (n=35), `shoulder_protraction` (n=35) and the three frontal
   metrics (n=95) now use measured percentiles. `trunk_sway` is still `guess`
   and is *refused* a measured threshold, because its own p90 error (10.3°)
   exceeds the proposed 80th-percentile cut (7.8°). n=35 clears the n≥20
   minimum but is not a stable percentile estimate; treat the sagittal cuts as
   provisional.

   Worth noting how wrong the hand-picked values were: `head_over_hip` was set
   by hand at 5.0° and measures 14.6° at the 80th percentile,
   `shoulder_protraction` 8.0° against 10.4°, `shoulder_tilt` 2.0° against
   6.0°. Every guess was too strict, i.e. would have flagged most people.

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

9. **`forward_head` cannot resolve its own thresholds.** At ±11.6° against
   cut points of 10°/18°, it is below the noise floor, and the app says so
   rather than offering a verdict. The cause is structural: it measures across
   the shortest span in the metric set (ear→shoulder, ~9 cm) using the
   *noisiest* landmark in it — the shoulder, at ~3× the ear's localisation
   noise. `head_over_hip` measures the same anatomy without the shoulder over
   four times the span.

10. **Image quality matters more than anything else measured here.** On studio
    photography (730–1029px subjects, plain background) every sagittal metric
    except `forward_head` reaches a 0% gross-error rate, and landmark noise
    halves (0.0219 → 0.0104 of body height). The app warns below 430px
    shoulder-to-ankle. The studio figures rest on only 12 images, so read
    "0%" as "no failures in a small sample", not as a rate.

11. **Statues and mannequins are still accepted as people.** Both the pose
    model and the independent detector call them persons because they are
    person-shaped. They are 2 of the 4 remaining guard leaks. Geometry cannot
    separate them and no further heuristics were attempted.

12. **Lateral pelvic tilt never resolves, so users are not shown it.** On all
    40 front-view test photos its uncertainty exceeded the width of its own
    slight band: the two hip-joint landmarks are too close together for their
    noise. The instrument still computes it; `report.py` leaves it out rather
    than print "could not measure" on every result.

13. **Portrait phone photos were measured sideways by the API** until EXIF
    orientation was honoured in `load_rgb`. Browsers always honoured it. No
    committed test image carries the tag, which is why nothing caught it; a
    test now builds one.

---

## Test data

`testdata/` holds only images whose COCO license id permits redistribution
*and* derivative works (4 = CC BY 2.0, 5 = CC BY-SA 2.0, 7 = no known
copyright restrictions, 8 = US Government Work). Per-image provenance —
source URL, COCO image id, license — is in `sources.csv` in each folder.
Nothing in this repo is of unrecorded origin.

Two independently collected sets, kept separate because they differ in a way
that turned out to matter more than anything else measured here.

**`testdata/pexels/`** — the previous phase's hand-curated set, carried over
from `claude/posture-assessment-demo-q1x8nj`. Pexels License, per-image
provenance in `testdata/pexels/sources_full.csv`. Deliberately shot studio
photography: plain backgrounds, single subject, fitted clothing, camera level,
subject 730–1029px shoulder-to-ankle.

| folder | n | purpose |
|---|---|---|
| `pexels/side/` | 2 | sagittal metrics |
| `pexels/front/` | 4 | frontal metrics |
| `pexels/edge/` | 6 | known-hard cases: two shots at 53°/58° torso yaw that defeated the previous view test, plus lean / head-back / weight-on-one-leg |
| `pexels/negative/` | 62 | statues, mannequins, dolls, hand and foot close-ups, bending, sitting, animals, crowds |

**`testdata/{side,front,negative}/`** — mined from COCO by `coco_mine.py`,
redistributable license ids only (4, 5, 7, 8), provenance in each folder's
`sources.csv`. Candid photography, 250–600px subjects.

The side set grew from 5 to 31 when the miner stopped requiring the feet to
be in frame — see "How the candidates were found" below.

| folder | n | purpose |
|---|---|---|
| `front/` | 33 | frontal metrics |
| `side/` | 31 | sagittal metrics |
| `negative/` | 14 | COCO images containing no person |

The 62-image negative set is the better false-positive benchmark of the two
and is what the guard leak rate below is measured on.

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
app.py                 Flask single-page app (HTML)
api.py                 JSON API for mini-program / mobile clients
web/
  posture.js           the logic and the plain report, ported to JS  <- what a mini-program reuses
  engine.js            loads MediaPipe in the browser from files next to the page
  src/                 the app's markup, styles and interaction
  dist/                build output (wasm and model parts not committed)
posture/
  geometry.py          pure angle maths, no vision deps, exactly testable
  landmarks.py         MediaPipe Tasks wrapper (the mp.solutions.* API is gone in 1.0.x)
  view.py              front/side/oblique classification, facing direction
  metrics.py           the measurements, and uncertainty propagation
  thresholds.py        every cut point, with a provenance tag and its basis
  guards.py            when NOT to answer
  assess.py            orchestration, verdict assembly, advice
  report.py            the plain answer a person reads (issues, exercises, retakes)
  consumer_zh.json     every word of it, shared by API and web
  overlay.py           skeleton drawing
scripts/
  setup.sh
  coco_mine.py               find candidates by annotation, not by eye
  fetch_from_manifest.py     download with provenance recording
  validate.py                pipeline over a folder + acceptance statistics
  repeatability.py           precision: readings under transforms that change nothing
  synthetic_warp.py          accuracy: readings under a KNOWN posture change
  commons_explore.py         Wikimedia Commons candidate search (see Test data)
  reference_distribution.py  norm-reference percentiles for thresholds
  build_web.py               build web/dist/; fails if the JS port disagrees on any test photo
  compare_models.py          can a smaller pose model stand in? (measured: no)
  report_survey.py           what a person would be told, photo by photo
reports/
  RELIABILITY.md             what the numbers are worth  <- read this
  DEPLOYMENT.md              mini-program / on-device constraints
  synthetic_warp.json        known-posture-change results
  HANDOFF.md                 state and next steps
  repeatability.json         raw measurements
  reference_distribution.json
  landmark_noise.json        feeds the ± on every reading
  model_swap.json            heavy vs full pose model on the same photos
```
