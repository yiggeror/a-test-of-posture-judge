"""Threshold constants for posture metrics.

READ THIS BEFORE TRUSTING ANY NUMBER IN THIS FILE.

Every threshold carries an explicit `basis` field with one of three values:

  "literature-adjacent" - a published clinical construct exists for something
      NEARLY this, but the landmarks or the reference axis used here differ, so
      the published cut-off does NOT transfer directly. The number below is
      still ours, not theirs.
  "geometric-estimate"  - derived by reasoning about plausible body dimensions
      (e.g. "a 5 cm forward displacement over a 18 cm ear-to-shoulder rise is
      about 15 degrees"). Arithmetic is sound; the input dimensions are typical
      values, not measured ones.
  "guess"               - chosen so the demo produces a sensible spread of
      outputs. No empirical basis whatsoever.

None of these are validated against a labelled posture dataset. Nothing here
has been calibrated against clinician ratings. They are demo defaults.

Wording rule for anything user-facing: these produce a REFERENCE reading
("参考"/"倾向"), never a diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Threshold:
    """A two-step band: |value| < mild is 'typical', >= marked is 'pronounced'."""

    mild: float
    marked: float
    unit: str
    basis: str          # one of the three labels documented above
    note: str           # shown in the UI next to the number


# --- Forward head -----------------------------------------------------------
# Metric: angle between the acromion->tragus line and the image vertical
# (plumb), measured on a sagittal (side) view. Larger = ear further forward.
#
# Literature anchor: the craniovertebral angle (CVA) is the angle between the
# HORIZONTAL and the C7->tragus line; CVA below roughly 48-50 deg is widely
# used as a forward-head-posture cut-off (e.g. Yip, Chiu & Poon, Manual Therapy
# 2008). Our metric differs on BOTH counts: we use the acromion rather than C7
# (MediaPipe has no C7 landmark at all), and we measure from vertical rather
# than horizontal. So the 48-50 deg figure cannot be reused and is not reused.
#
# The numbers below are geometry: at a typical ear-to-acromion vertical rise of
# ~18 cm, a 5 cm forward displacement is atan(5/18) = 15.5 deg and 8 cm is
# atan(8/18) = 24 deg. Those displacements are themselves typical-case
# assumptions, not measurements.
FORWARD_HEAD = Threshold(
    mild=15.0, marked=25.0, unit="deg", basis="geometric-estimate",
    note="由典型耳/肩位移几何推算而来，并非沿用临床 CVA 阈值——"
         "所用关键点不同（肩峰而非 C7），参考轴也不同（铅垂线而非水平线）。",
)

# --- Rounded shoulders ------------------------------------------------------
# Metric: horizontal offset of the acromion relative to the ear lobe, signed so
# positive = acromion anterior to the ear, normalised by torso length
# (acromion->hip distance in pixels) to remove image scale.
#
# There is no normalised published cut-off we are confident in. Clinical
# practice tends to use absolute measures instead (acromion-to-table distance
# in supine, or a tape measure to a wall), which are not recoverable from an
# uncalibrated photo.
ROUNDED_SHOULDER = Threshold(
    mild=0.08, marked=0.15, unit="x torso length", basis="guess",
    note="纯猜测，未采用任何已发表的归一化阈值。只有正负号、"
         "以及同一个人两张照片之间的相对大小，才具备一定参考价值。",
)

# --- Sagittal trunk alignment ----------------------------------------------
# Metric: 180 deg minus the shoulder-hip-ankle angle, i.e. how far those three
# points depart from a straight line in the sagittal plane.
#
# NAMING WARNING: the brief calls this "pelvic tilt". It is not. Anterior /
# posterior pelvic tilt is defined by the ASIS-PSIS line, and MediaPipe
# provides NEITHER of those landmarks - it emits one hip point per side, near
# the greater trochanter. What follows is a global sagittal ALIGNMENT measure.
# A real pelvic tilt angle cannot be obtained from these landmarks at all.
SAGITTAL_ALIGNMENT = Threshold(
    mild=8.0, marked=15.0, unit="deg", basis="guess",
    note="纯猜测。另需注意：这是躯干矢状面对齐，不是骨盆倾斜——"
         "MediaPipe 不提供 ASIS/PSIS 关键点，真正的骨盆倾斜在此测不了。",
)

# --- Shoulder height asymmetry ---------------------------------------------
# Metric: tilt of the left-right acromion line away from the image horizontal,
# on a frontal view.
#
# A ~1 cm height difference across a ~38 cm biacromial width is about 1.5 deg,
# and mild asymmetry is extremely common in asymptomatic people. This metric is
# also the one most easily destroyed by camera roll: tilting the phone 3 deg
# moves the reading by 3 deg, and nothing in a single photo can separate the
# two.
SHOULDER_TILT = Threshold(
    mild=2.5, marked=5.0, unit="deg", basis="geometric-estimate",
    note="相机翻滚角会 1:1 叠加到该读数上，在单张未标定照片中无法与真实"
         "不对称分离。小数值请当作噪声看待。",
)

# Landmark visibility below this is treated as 'not usable' rather than being
# quietly used anyway. MediaPipe's visibility is a model confidence in [0,1];
# 0.5 is the library's own default detection threshold.
MIN_VISIBILITY = 0.5

# Heuristic split between a frontal and a sagittal view, using shoulder width
# divided by torso length. A frontal adult is ~0.55-0.75; a true side view
# collapses the acromions on top of each other and drops well below this.
# Tuned by eye on a handful of images - a guess, and reported as such.
VIEW_RATIO_SIDE_MAX = 0.25
VIEW_RATIO_FRONT_MIN = 0.40


# --- Plausibility guards ----------------------------------------------------
# Added after Phase 2 validation found MediaPipe emitting a full, confident
# 33-landmark skeleton on a photograph of a HAND (testdata/test.jpg: ear
# visibility 0.996, shoulder visibility 1.000). MediaPipe's `visibility` is a
# per-landmark occlusion estimate, NOT a "this is really a person" score, so it
# cannot be used to catch that case. These are cheap anatomical sanity checks.
#
# Both are geometric estimates from human proportions, and both are deliberately
# loose - they exist to reject nonsense, not to grade posture.

# Angle of the shoulder-midpoint -> hip-midpoint axis away from image vertical.
# Every metric here assumes a STANDING subject; beyond this the assumption is
# simply void (the subject is sitting, lying, bending, or it is not a person).
MAX_TORSO_TILT_DEG = 40.0

# (ear midpoint -> shoulder midpoint) length divided by torso length.
# Adults sit around 0.4-0.6. The hand false-positive scored 0.16.
MIN_HEAD_TORSO_RATIO = 0.22
MAX_HEAD_TORSO_RATIO = 1.10
