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
# simply void (the subject is bending, lying, or it is not a person).
#
# Tightened from 40 after round-2 measurement: across 11 genuinely standing
# photos the torso axis never exceeded 10.2 deg, and 40 was letting people
# bent at the waist through - one produced "forward head -73 deg, pronounced".
# 25 keeps more than double the observed headroom over real standing subjects.
MAX_TORSO_TILT_DEG = 25.0

# Knee angle (hip-knee-ankle), taking the more extended leg. A standing subject
# has near-extended knees; sitting, kneeling and crouching do not.
#
# Measured round 2: standing 175.7-179.4 deg (n=11, including the edge cases),
# sitting 35.2 and 81.2, kneeling 19.0. The two populations are separated by
# roughly 90 deg, so this cut-off sits in a very wide empty gap.
#
# It does NOT catch bending at the waist - knees stay straight there - which is
# what MAX_TORSO_TILT_DEG is for.
MIN_KNEE_EXTENSION_DEG = 150.0

# (ear midpoint -> shoulder midpoint) length divided by torso length.
# Adults sit around 0.4-0.6. The hand false-positive scored 0.16.
MIN_HEAD_TORSO_RATIO = 0.22
MAX_HEAD_TORSO_RATIO = 1.10


# --- View gate (body yaw) ---------------------------------------------------
# Replaces the shoulder-width/torso-length ratio as the primary view test.
#
# Measured on the 12 real photos collected in round 2 (reports/PHASE3.md):
#   true frontal      yaw =  5.2, 14.1, 15.0, 22.1, 24.1 deg
#   rotated ~1/3 turn yaw = 53.2, 58.4 deg
#   true lateral      yaw = 84.9, 86.4, 86.4, 87.6, 87.9 deg
#
# The two rotated shots are the ones the ratio test wrongly called frontal
# (ratio 0.453 and 0.401, just over the old 0.40 cut-off), and on which it then
# computed a shoulder-height reading that perspective had already contaminated.
# Yaw separates them with wide margins on both sides; the ratio does not
# separate them at all.
#
# Caveat: n = 12, and yaw comes from MediaPipe's world-landmark z, the least
# reliable output of a single-image model. Used only as a coarse GATE - reject
# the ambiguous middle - never as a measurement. The wide "oblique" band in
# between is deliberate.
YAW_FRONT_MAX = 30.0
YAW_SIDE_MIN = 70.0

# --- Landmark uncertainty ---------------------------------------------------
# Perturbation budget for the reported +/- on ear-dependent metrics, as a
# fraction of torso length (scale free, so it needs no image dimensions).
#
# Why this exists: on a 1400px-tall photo, displacing the ear landmark by 10px
# swings the forward-head angle by 4.4 deg, while the gap between the "mild"
# and "pronounced" thresholds is only 10 deg. A bare number implies a precision
# the landmark simply does not have, so every ear-dependent reading is shown
# with the swing this budget produces.
#
# 3% of torso length is roughly 10-13px on the collected photos. It is an
# assumption about typical landmark error, NOT a measured error distribution -
# an ear hidden under hair can be off by far more (observed: ~50px, and
# MediaPipe still reported visibility 1.000).
LANDMARK_JITTER_FRACTION = 0.03


# Last-resort output bound for the shoulder-hip-ankle deviation. A standing
# human simply cannot depart 60 deg from a straight sagittal line - at that
# point the landmarks are not on a standing body. Observed: real standing
# subjects -3 to -10 deg, a person bent at the waist -29, and a statue in a
# contrapposto pose 129, which is what this catches.
#
# This is a sanity bound on the OUTPUT, not a posture judgement. It is a guard
# of last resort: everything it catches should ideally have been caught by an
# input check first.
MAX_PLAUSIBLE_SAGITTAL_DEV_DEG = 60.0
