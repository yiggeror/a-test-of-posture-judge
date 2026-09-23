"""Decision thresholds, and an honest account of where each one comes from.

Every constant carries a provenance tag. The tags mean exactly this:

  guess                 Picked by hand. No supporting evidence of any kind.
                        Do not present a reading judged by one of these as if
                        it were grounded.
  geometric-estimate    Derived from body proportions or projection geometry.
                        The derivation is stated and can be checked, but it
                        says nothing about what value is clinically notable.
  literature-adjacent   Taken from published work on a RELATED measurement and
                        adapted. The adaptation is unvalidated; the mismatch
                        is spelled out in `basis`.
  population-percentile Derived from the distribution measured over this
                        repo's own reference set. States n. This says where a
                        reading sits among other photographs -- it does NOT
                        say the reading is healthy or unhealthy.

What "calibration" can and cannot mean here
-------------------------------------------
There are no clinical labels anywhere in this project. Nobody has told us
which of the reference photographs show a person a clinician would consider
to have forward head posture. So no threshold in this file can be validated
against a ground truth of "has the condition".

The most that is available is NORM-REFERENCING: given enough photographs,
report where a reading falls in the distribution of readings from comparable
photographs. That converts "18 degrees is notable" (a guess) into "18 degrees
is at the 92nd percentile of n=NN photographs from the reference set" (a
measurement). It is a real improvement and it is still not a clinical finding,
because an unusual value in a population is not the same as a problematic one,
and the reference population here is photographs from a general-purpose
object-detection dataset rather than a screened cohort.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

REFERENCE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports", "reference_distribution.json")

# Verdict band keys. User-facing wording is deliberately tentative: this tool
# reports tendencies for reference, it does not diagnose.
BAND_REFERENCE = "reference"
BAND_SLIGHT = "slight"
BAND_NOTABLE = "notable"

BAND_LABELS_ZH = {
    BAND_REFERENCE: "参考范围内",
    BAND_SLIGHT: "轻度倾向",
    BAND_NOTABLE: "明显倾向",
}


@dataclass
class ThresholdSpec:
    key: str
    label_zh: str
    slight: float           # absolute-value cut for "slight tendency"
    notable: float          # absolute-value cut for "notable tendency"
    direction: str          # "positive_only" or "symmetric"
    provenance: str
    basis: str
    n: int | None = None    # sample size, when provenance is a percentile

    def classify(self, value: float) -> str:
        v = value if self.direction == "positive_only" else abs(value)
        if v >= self.notable:
            return BAND_NOTABLE
        if v >= self.slight:
            return BAND_SLIGHT
        return BAND_REFERENCE


# --- Default thresholds -----------------------------------------------------
# These apply when reports/reference_distribution.json is absent. Read the
# provenance on each one before trusting any verdict built from it.

# Which metrics carry a verdict was decided by MEASUREMENT, not by which ones
# have familiar clinical names. See reports/RELIABILITY.md. The ranking that
# drove it, on the COCO set (gross error = residual over 5 deg after a known
# rotation is removed):
#
#   lateral_head_shift    +/-0.9   0%     -> verdict
#   head_over_hip         +/-1.1   5%     -> verdict
#   shoulder_tilt         +/-1.3   1%     -> verdict
#   shoulder_protraction  +/-1.4   5%     -> verdict
#   trunk_sway            +/-2.0  13%     -> verdict
#   pelvis_tilt           +/-3.5   3%     -> verdict (often unresolved, below)
#   forward_head          +/-2.6  22%     -> diagnostic only
#   head_tilt             +/-3.4  10%     -> diagnostic only
#   head_vs_shoulder_tilt +/-3.7  13%     -> diagnostic only
#   knee_deviation        +/-3.8  25%     -> guard only
#
# Each cut point below is at least 3x the metric's measured uncertainty, so a
# reading can actually land in a band rather than straddling it. That is a
# necessary condition for a threshold to mean anything, not a sufficient one:
# these are still not clinical cutoffs.

DEFAULTS: dict[str, ThresholdSpec] = {
    # --- sagittal (side view) ---
    "head_over_hip": ThresholdSpec(
        key="head_over_hip", label_zh="头部前移",
        slight=5.0, notable=10.0, direction="positive_only",
        provenance="geometric-estimate",
        basis="How far the ear sits in front of the hip, as an angle from "
              "vertical. CRITERION-referenced, not population-referenced: the "
              "reference is ideal plumb-line alignment (ear over shoulder over "
              "hip, the standard sagittal posture reference), which is 0 deg in "
              "this geometry. That zero is checked, not assumed: the three "
              "upright studio side photos in testdata/pexels read -0.7, +0.3 "
              "and +0.6 deg. Measurement noise is +/-1.0 deg (1 sigma). The "
              "cuts sit at 5x and 10x that noise above the zero; the multiples "
              "are a judgment. A user-supplied side photo with an obvious forward "
              "head (not in the repo: its licence is unknown) reads +11.7. The "
              "only other photo seen above 5 deg that passed the guards, "
              "coco_333626, is a cook bent over a counter -- a task posture, "
              "which the side-view arm guard added alongside these cuts now "
              "rejects. So the upper cut rests on one real example. These cuts replaced "
              "measured 80th/95th percentiles (14.6/33.1) that missed the "
              "user-supplied photo entirely: percentiles of unscreened candid "
              "photos say what is common, and a large forward head is common.",
    ),
    "shoulder_protraction": ThresholdSpec(
        key="shoulder_protraction", label_zh="圆肩（肩前移）",
        slight=5.0, notable=12.0, direction="positive_only",
        provenance="geometric-estimate",
        basis="Shoulder joint displacement from the hip, as an angle from "
              "vertical. CRITERION-referenced like head_over_hip. The upright "
              "zero is NOT 0 in this geometry: MediaPipe's shoulder landmark "
              "sits slightly behind its hip landmark in upright stance, and the "
              "three upright studio side photos read -5.0, -5.9 and -3.9 deg. "
              "The cuts sit 10 and 17 deg forward of that zero (measurement "
              "noise +/-1.4 deg); the offsets are a judgment. Replaced measured "
              "percentiles (10.4/35.7) for the same reason as head_over_hip: a "
              "35.7 deg upper-body lean is not a standing posture at all.",
    ),
    "trunk_sway": ThresholdSpec(
        key="trunk_sway", label_zh="躯干前后倾",
        slight=6.0, notable=10.0, direction="symmetric",
        provenance="guess",
        basis="Whole-body forward/backward lean, hip relative to ankle. Quiet "
              "standing sway is well under a degree, so a multi-degree lean in "
              "a still photograph is a real offset. The usable range is narrow: "
              "the neutrality guard rejects the photo entirely above 12 deg, so "
              "the whole scale lives between the 3x-uncertainty floor (~6 deg) "
              "and that ceiling. Two bands is all this metric can support, and where the line between them falls is chosen by hand.",
    ),
    # --- frontal (front view) ---
    "shoulder_tilt": ThresholdSpec(
        key="shoulder_tilt", label_zh="高低肩",
        slight=6.0, notable=12.0, direction="symmetric",
        provenance="guess",
        basis="Camera roll adds directly to this reading and cannot be "
              "separated from a real shoulder difference in a single "
              "uncalibrated photo. Replaced by measured percentiles when the "
              "reference distribution has enough samples; until then these are chosen by hand.",
    ),
    "pelvis_tilt": ThresholdSpec(
        key="pelvis_tilt", label_zh="骨盆侧倾",
        slight=7.0, notable=12.0, direction="symmetric",
        provenance="guess",
        basis="Lateral pelvic obliquity only -- NOT anterior/posterior pelvic "
              "tilt, which these landmarks cannot measure. Measured across the "
              "hips, a narrower span than the shoulders, so it is the least "
              "precise frontal reading (+/-3.5 deg) and frequently comes back "
              "unresolved. Same camera-roll confound as shoulder_tilt, plus "
              "standing on one leg produces several degrees of it. Chosen by hand.",
    ),
    "lateral_head_shift": ThresholdSpec(
        key="lateral_head_shift", label_zh="头部侧偏",
        slight=5.0, notable=9.0, direction="symmetric",
        provenance="guess",
        basis="Horizontal offset of the head from the shoulder midline, as an "
              "angle subtended at the hips. Promoted from diagnostic to a "
              "verdict metric because it measures best of anything here "
              "(+/-0.9 deg, 0% gross error) -- it uses a long span and avoids "
              "the eye landmarks. CAVEAT: a subject turned slightly away from "
              "the camera shifts the nose off the shoulder midline without any "
              "real head shift, and the front-view gate admits up to about 30 "
              "deg of torso yaw, so some of this reading can be stance. The cut points are chosen by hand.",
    ),
}

# Measured and displayed, but never turned into a verdict.
#
#   forward_head           +/-2.6 deg against 10/18 thresholds: cannot resolve
#                          its own bands. Kept because comparing it with
#                          head_over_hip identifies shoulder mislocalisation --
#                          they share the shoulder with opposite sign.
#   head_tilt              measured across the eyes, a 64px span, the shortest
#                          in the set.
#   head_vs_shoulder_tilt  camera roll cancels in it, which is genuinely
#                          useful, but it inherits head_tilt's noise.
#   knee_deviation         25% gross error. Used as a NEUTRALITY GUARD (a bent
#                          knee invalidates the other readings), never reported
#                          as a postural finding.
DIAGNOSTIC_ONLY = {"forward_head", "head_tilt", "head_vs_shoulder_tilt",
                   "knee_deviation"}

# Metrics judged against an ideal alignment rather than against a population.
# load_thresholds() never replaces these with reference-set percentiles.
#
# Why: the sagittal percentiles came from unscreened candid photographs, where
# a forward head and a forward-leaning upper body are common. Their 80th
# percentile therefore marked only the extreme tail as a "slight tendency",
# and a user-supplied photo with an obvious forward head (+11.7 deg, against an
# upright studio zero of about 0) came back as normal. For a posture check, the
# useful question is "how far from upright", not "how rare among photos".
# The frontal metrics keep their percentiles: their ideal is symmetric about 0
# and the reference readings cluster there, so the two framings agree.
CRITERION_REFERENCED = {"head_over_hip", "shoulder_protraction"}


def load_thresholds(path: str | None = None) -> dict[str, ThresholdSpec]:
    """Return thresholds, preferring measured percentiles where available.

    When reports/reference_distribution.json exists, the slight/notable cuts
    for each metric are replaced by the 80th and 95th percentile of the
    absolute readings over the reference set, and the provenance is upgraded to
    population-percentile with n recorded. Metrics missing from the file keep
    their defaults, tags included.
    """
    specs = {k: ThresholdSpec(**vars(v)) for k, v in DEFAULTS.items()}
    path = path or REFERENCE_PATH
    if not os.path.exists(path):
        return specs
    try:
        with open(path) as fh:
            ref = json.load(fh)
    except (OSError, ValueError):
        return specs

    for key, spec in specs.items():
        if key in CRITERION_REFERENCED:
            continue
        entry = (ref.get("metrics") or {}).get(key)
        if not entry:
            continue
        n = int(entry.get("n", 0))
        # Percentiles from a handful of samples are noise. Below this the
        # defaults are kept, with their honest "guess" tag intact.
        # provenance: guess -- a minimum-n rule of thumb, not a power analysis
        if n < 20:
            spec.basis += (f" [reference set has only n={n} for this metric, "
                           "too few for percentile thresholds, so the hand-picked "
                           "values above are still in use]")
            continue
        # A cut point below the metric's own measurement error cannot separate
        # anything, however large the sample behind it. Adopting such a
        # threshold would launder a coin flip into a "measured" tag, which is
        # worse than leaving an honest guess in place.
        if entry.get("usable_for_thresholds") is False:
            reason = entry.get("unusable_reason", "flagged unusable by "
                               "scripts/reference_distribution.py")
            spec.basis += f" [percentile thresholds rejected: {reason}]"
            continue
        p80, p95 = entry.get("abs_p80"), entry.get("abs_p95")
        if p80 is None or p95 is None:
            continue
        spec.slight = round(float(p80), 1)
        spec.notable = round(float(p95), 1)
        spec.n = n
        spec.provenance = "population-percentile"
        spec.basis = (
            f"80th/95th percentile of |reading| over n={n} photographs in this "
            f"repo's reference set ({ref.get('source', 'unspecified source')}). "
            "This locates a reading within a population of photographs. It does "
            "NOT indicate whether the posture is healthy: the reference set is "
            "unscreened general photography, not a clinical cohort, and no "
            "image in it carries a clinical label."
        )
    return specs
