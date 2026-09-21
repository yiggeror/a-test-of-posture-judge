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

DEFAULTS: dict[str, ThresholdSpec] = {
    "forward_head": ThresholdSpec(
        key="forward_head", label_zh="头部前引",
        slight=10.0, notable=18.0, direction="positive_only",
        provenance="guess",
        basis="Published cutoffs exist for the craniovertebral angle (roughly "
              "CVA < 48-50 deg indicates forward head posture), but CVA is "
              "measured from the C7 spinous process and this metric is measured "
              "from the acromion, which sits several centimetres anterior. The "
              "offset between the two has not been characterised here, so "
              "converting the published cutoff would be inventing a mapping. "
              "These numbers are placeholders chosen by hand.",
    ),
    "shoulder_protraction": ThresholdSpec(
        key="shoulder_protraction", label_zh="圆肩（肩前移）",
        slight=6.0, notable=12.0, direction="positive_only",
        provenance="guess",
        basis="No published cutoff is expressed in this geometry (acromion "
              "displacement from the hip, as an angle from vertical). Chosen by "
              "hand.",
    ),
    "trunk_sway": ThresholdSpec(
        key="trunk_sway", label_zh="躯干前后倾",
        slight=4.0, notable=8.0, direction="symmetric",
        provenance="guess",
        basis="Quiet-stance sway in healthy adults is well under a degree, so "
              "a sustained multi-degree lean in a still photograph is a real "
              "postural offset rather than sway. Where the notable/slight line "
              "belongs is still a hand-picked choice.",
    ),
    "knee_deviation": ThresholdSpec(
        key="knee_deviation", label_zh="膝关节角度",
        slight=6.0, notable=12.0, direction="symmetric",
        provenance="guess",
        basis="Genu recurvatum is commonly described from about 5-10 deg of "
              "hyperextension, but that is a goniometric measurement at the "
              "joint, not a three-landmark projection from a photograph. "
              "Chosen by hand.",
    ),
    "shoulder_tilt": ThresholdSpec(
        key="shoulder_tilt", label_zh="高低肩",
        slight=2.0, notable=4.0, direction="symmetric",
        provenance="guess",
        basis="Camera roll adds directly to this reading and is typically a "
              "degree or two in handheld photographs, which is the same size "
              "as the effect being measured. Any cut point below a few degrees "
              "is mostly measuring the photographer.",
    ),
    "pelvis_tilt": ThresholdSpec(
        key="pelvis_tilt", label_zh="骨盆侧倾",
        slight=2.0, notable=4.0, direction="symmetric",
        provenance="guess",
        basis="Lateral pelvic obliquity carries the same camera-roll confound "
              "as shoulder_tilt: a degree or two of handheld camera roll adds "
              "directly to the reading and is the same size as the effect "
              "being measured. Standing with the weight on one leg also "
              "produces several degrees of it, which is a stance rather than a "
              "postural trait. Chosen by hand.",
    ),
    "head_tilt": ThresholdSpec(
        key="head_tilt", label_zh="头部侧倾",
        slight=3.0, notable=6.0, direction="symmetric",
        provenance="guess",
        basis="Measured across the eyes, the shortest span of any metric here, "
              "so the noisiest. Chosen by hand.",
    ),
    "head_vs_shoulder_tilt": ThresholdSpec(
        key="head_vs_shoulder_tilt", label_zh="头肩相对侧倾",
        slight=3.0, notable=6.0, direction="symmetric",
        provenance="guess",
        basis="Camera roll cancels in this difference, so unlike head_tilt and "
              "shoulder_tilt it is not confounded by camera levelness. The cut "
              "points are still hand-picked.",
    ),
}

# Diagnostic metrics are measured and displayed but never produce a verdict.
DIAGNOSTIC_ONLY = {"head_over_hip", "lateral_head_shift"}


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
