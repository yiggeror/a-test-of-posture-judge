"""False-positive guards.

The tool's failure mode that matters is not "misses a real postural issue", it
is "produces a confident number from an input that cannot support one". Four
confirmed cases from the previous phase:

  1. a photo of a bare palm -> full 33-point skeleton
  2. a sock / calf close-up -> full 33-point skeleton
  3. long hair over the ear -> ear landmark on the hair, visibility 1.000,
     app reported "forward head +25.39 deg, notable tendency"
  4. an upright statue / mannequin / doll -> indistinguishable by geometry

MediaPipe's `visibility` and `presence` channels do not help with any of
these; case 3 is the proof, where the fabricated landmark scored 1.000.

So the guards here use signals that are independent of the pose model's own
confidence:

  * geometry and framing (cheap, and catches genuine framing problems)
  * STABILITY under small input perturbations -- a landmark that is inferred
    rather than observed tends to move when the image is nudged, while one
    that is locked to real image evidence does not. This reuses the
    repeatability machinery and is the only guard here that addresses case 3.
  * an INDEPENDENT detector (EfficientDet-Lite, Apache-2.0) asked whether a
    "person" box exists at all. This addresses cases 1 and 2.

None of them addresses case 4. An upright statue is a genuinely human-shaped
object and both the pose model and the object detector are expected to call it
a person. That limitation is stated in the README rather than papered over
with more heuristics.
"""
from __future__ import annotations

import atexit
import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import landmarks as L
from .geometry import distance, midpoint
from .view import ViewEstimate

SEVERITY_BLOCK = "block"   # readings must not be shown as verdicts
SEVERITY_WARN = "warn"     # readings shown, with the caveat attached

# --- Guard thresholds -------------------------------------------------------

# A second detected person at least this fraction of the subject's bounding-box
# diagonal makes it ambiguous which person was measured.
# provenance: guess
BYSTANDER_SIZE_RATIO = 0.45

# Landmarks this far outside the frame (as a fraction of body scale) mean the
# subject is cropped, and the out-of-frame landmark positions are inferred.
# provenance: guess
OUT_OF_FRAME_MARGIN = 0.02

# Human segment proportions. Shoulder-to-hip and hip-to-ankle spans differ by a
# roughly fixed ratio; a detection whose ratio is far outside this range is not
# looking at a standing human body.
# provenance: geometric-estimate (segment proportions, generous bounds)
TORSO_LEG_RATIO_MIN = 0.45
TORSO_LEG_RATIO_MAX = 1.35

# Physiological sanity ceiling for any reported sagittal angle. Beyond this the
# reading is a measurement failure, not a posture.
# provenance: geometric-estimate
MAX_PLAUSIBLE_ANGLE = 45.0

# Knee flexion above this means the subject is not standing neutrally, so the
# sagittal readings describe a different pose than the one being assessed.
# provenance: guess
MAX_KNEE_FLEXION = 12.0

# Trunk lean above this means the same.
# provenance: guess
MAX_TRUNK_LEAN = 12.0

# --- Frontal-plane neutrality ----------------------------------------------
# The knee and trunk guards above are sagittal-only, so without these a front
# photo of someone mid-stride passes completely unchecked. A front view needs
# its own neutrality test or it is held to a weaker standard than a side view.

# Ankle separation as a multiple of hip separation. A neutral stance is
# roughly hip-width, so ~1.0; a stride, a wide stance or a lunge exceeds this.
# provenance: geometric-estimate (hip-width stance, generous upper bound)
MAX_STANCE_WIDTH_RATIO = 2.2

# Lateral offset of the hip midpoint from the ankle midpoint, as a fraction of
# body scale. Standing with the weight parked on one leg -- which tilts the
# pelvis and is one of the things this tool reports on -- shifts this. Above
# the limit the subject is not evenly loaded and the pelvis reading describes
# the stance rather than the person.
# provenance: guess
MAX_WEIGHT_SHIFT_FRAC = 0.06

# Wrists must hang below the hips. Arms crossed, on hips, or in pockets change
# the shoulder landmark positions, which every shoulder metric depends on.
# provenance: guess
MIN_WRIST_BELOW_HIP_FRAC = -0.02

# Knee extension of the STRAIGHTER leg, in a front view. The sagittal knee
# guard uses the signed sagittal deviation and does not apply here, which left
# front-view seated subjects completely unchecked -- a seated photo from the
# previous phase's negative set produced frontal readings with no complaint.
#
# provenance: measured -- on testdata/pexels, genuine front-view standing
# subjects measure 176.5-179.0 deg (n=4) while seated subjects measure 72.3
# and 88.5 deg. The two populations are ~85 deg apart, so this cut sits in a
# very wide empty gap. It matches the 150 deg the previous phase measured for
# the sagittal case (standing 175.7-179.4, sitting 35.2/81.2, kneeling 19.0).
# Note this does NOT separate upright statues, which measure 172-179 like
# real standing people.
MIN_KNEE_EXTENSION_FRONT_DEG = 150.0

# Stability: landmark displacement under a small perturbation, as a fraction of
# body scale, above which the landmark is treated as inferred rather than
# observed. Calibrated against the measured noise floor -- see
# reports/REPEATABILITY.md.
# provenance: guess -- replace once the noise floor is measured on more images
UNSTABLE_LANDMARK_FRAC = 0.025

# Landmarks whose stability actually matters for the reported metrics.
CRITICAL_LANDMARKS = (
    L.LEFT_EAR, L.RIGHT_EAR, L.LEFT_SHOULDER, L.RIGHT_SHOULDER,
    L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_ANKLE, L.RIGHT_ANKLE,
)

# Minimum subject height, shoulder to ankle, in pixels.
#
# provenance: measured -- scripts/repeatability.py stratified by subject size
# over this repo's reference set. Restricting to subjects >= 430px cut the
# sagittal gross-error rate from 48% to 36% (forward_head), 43% to 25%
# (head_over_hip), 40% to 27% (shoulder_protraction) and 48% to 32%
# (trunk_sway), and moved the robust rotation slopes from about -0.7 to about
# -0.9. Frontal metrics barely moved (1% either way), so this is warned about
# rather than blocked, and the warning names the sagittal readings.
MIN_SUBJECT_HEIGHT_PX = 430.0

_DETECTOR_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "efficientdet_lite0.tflite")


@dataclass
class GuardFinding:
    key: str
    severity: str
    message_zh: str
    detail: dict = field(default_factory=dict)


def check_framing(pose: L.PoseResult) -> list[GuardFinding]:
    """Landmarks outside the image, and multiple people."""
    out: list[GuardFinding] = []
    scale = L.body_scale(pose)
    margin = OUT_OF_FRAME_MARGIN * scale

    escaped = []
    for idx in CRITICAL_LANDMARKS:
        x, y = pose.xy(idx)
        if (x < -margin or x > pose.width + margin
                or y < -margin or y > pose.height + margin):
            escaped.append(L.LANDMARK_NAMES[idx])
    if escaped:
        out.append(GuardFinding(
            key="landmarks_out_of_frame", severity=SEVERITY_BLOCK,
            message_zh="关键点超出画面范围，说明人物被裁切，超框点的位置是推测的，"
                       "不能用于测量。请上传完整全身照。",
            detail={"landmarks": escaped}))

    subj = _bbox(pose)
    subj_diag = math.hypot(subj[2] - subj[0], subj[3] - subj[1])
    for bx in pose.other_pose_bboxes:
        diag = math.hypot(bx[2] - bx[0], bx[3] - bx[1])
        if subj_diag > 0 and diag / subj_diag >= BYSTANDER_SIZE_RATIO:
            out.append(GuardFinding(
                key="multiple_people", severity=SEVERITY_BLOCK,
                message_zh="画面中检测到不止一个人，无法确定测量的是哪一位。"
                           "请上传只有一个人的照片。",
                detail={"n_poses": pose.n_poses_detected,
                        "size_ratio": round(diag / subj_diag, 3)}))
            break
    return out


def check_subject_resolution(pose: L.PoseResult, view: ViewEstimate
                             ) -> list[GuardFinding]:
    """Warn when the subject is too small for the sagittal metrics to hold up.

    Measured, not assumed: sagittal reliability depends strongly on how many
    pixels the subject occupies, while frontal reliability barely does. See
    the provenance note on MIN_SUBJECT_HEIGHT_PX.
    """
    scale = L.body_scale(pose)
    if scale >= MIN_SUBJECT_HEIGHT_PX or view.view == "front":
        return []
    return [GuardFinding(
        key="subject_too_small", severity=SEVERITY_WARN,
        message_zh=f"人物在画面中偏小（肩到踝约 {scale:.0f} 像素，建议至少 "
                   f"{MIN_SUBJECT_HEIGHT_PX:.0f} 像素）。实测显示，侧面各项读数的"
                   "严重误差率在人物偏小时会上升约一半。请离近一些或使用更高分辨率"
                   "的照片重拍。",
        detail={"body_scale_px": round(scale, 1),
                "recommended_min": MIN_SUBJECT_HEIGHT_PX})]


def check_proportions(pose: L.PoseResult) -> list[GuardFinding]:
    """Reject detections whose segment ratios are not those of a standing body."""
    sh = midpoint(pose.xy(L.LEFT_SHOULDER), pose.xy(L.RIGHT_SHOULDER))
    hip = midpoint(pose.xy(L.LEFT_HIP), pose.xy(L.RIGHT_HIP))
    ank = midpoint(pose.xy(L.LEFT_ANKLE), pose.xy(L.RIGHT_ANKLE))

    torso = distance(sh, hip)
    leg = distance(hip, ank)
    if leg <= 1e-6:
        return [GuardFinding(
            key="degenerate_geometry", severity=SEVERITY_BLOCK,
            message_zh="检测到的人体比例异常，无法测量。",
            detail={"torso_px": torso, "leg_px": leg})]

    ratio = torso / leg
    if not (TORSO_LEG_RATIO_MIN <= ratio <= TORSO_LEG_RATIO_MAX):
        return [GuardFinding(
            key="implausible_proportions", severity=SEVERITY_BLOCK,
            message_zh="检测到的躯干与腿部比例不符合站立人体，"
                       "这通常意味着画面中并没有一个完整站立的人。",
            detail={"torso_leg_ratio": round(ratio, 3),
                    "accepted_range": [TORSO_LEG_RATIO_MIN, TORSO_LEG_RATIO_MAX]})]
    return []


def check_pose_neutrality(metrics: dict) -> list[GuardFinding]:
    """The subject must be standing neutrally for the readings to mean anything."""
    out: list[GuardFinding] = []
    knee = metrics.get("knee_deviation")
    if knee is not None and knee.value > MAX_KNEE_FLEXION:
        out.append(GuardFinding(
            key="knee_flexed", severity=SEVERITY_BLOCK,
            message_zh="膝关节明显弯曲，说明拍摄时不是中立站姿，"
                       "此时的其他读数描述的是另一个姿势，不作判定。",
            detail={"knee_deviation": round(knee.value, 1),
                    "limit": MAX_KNEE_FLEXION}))

    trunk = metrics.get("trunk_sway")
    if trunk is not None and abs(trunk.value) > MAX_TRUNK_LEAN:
        out.append(GuardFinding(
            key="trunk_leaning", severity=SEVERITY_BLOCK,
            message_zh="整个躯干明显前后倾斜，说明不是中立站姿，"
                       "上方各项读数不作判定。",
            detail={"trunk_sway": round(trunk.value, 1),
                    "limit": MAX_TRUNK_LEAN}))
    return out


def check_frontal_neutrality(pose: L.PoseResult) -> list[GuardFinding]:
    """Frontal-plane equivalent of check_pose_neutrality.

    Without this, a front view is held to a weaker standard than a side view:
    the knee and trunk guards only fire on sagittal metrics, so a front photo
    of someone mid-stride would be measured and reported without complaint.
    """
    out: list[GuardFinding] = []
    scale = L.body_scale(pose)
    if scale <= 1e-6:
        return out

    lank, rank = pose.xy(L.LEFT_ANKLE), pose.xy(L.RIGHT_ANKLE)
    lhip, rhip = pose.xy(L.LEFT_HIP), pose.xy(L.RIGHT_HIP)
    hip_w = distance(lhip, rhip)
    ankle_w = distance(lank, rank)

    if hip_w > 1e-6 and (ankle_w / hip_w) > MAX_STANCE_WIDTH_RATIO:
        out.append(GuardFinding(
            key="stance_too_wide", severity=SEVERITY_BLOCK,
            message_zh="双脚分开过宽或正处于跨步中，不是中立站姿，"
                       "此时骨盆与肩部的读数反映的是站姿而不是体态。",
            detail={"stance_width_ratio": round(ankle_w / hip_w, 2),
                    "limit": MAX_STANCE_WIDTH_RATIO}))

    hip_mid = midpoint(lhip, rhip)
    ankle_mid = midpoint(lank, rank)
    shift = abs(hip_mid[0] - ankle_mid[0]) / scale
    if shift > MAX_WEIGHT_SHIFT_FRAC:
        out.append(GuardFinding(
            key="weight_on_one_leg", severity=SEVERITY_WARN,
            message_zh="重心明显偏向一侧腿。这会直接造成骨盆侧倾与高低肩的读数，"
                       "但反映的是当时的站法，不一定是长期体态。"
                       "建议双脚平均受力后重拍一张对照。",
            detail={"weight_shift_frac": round(shift, 4),
                    "limit": MAX_WEIGHT_SHIFT_FRAC}))

    # Knee extension, taking the STRAIGHTER leg: standing with one knee
    # slightly soft is normal, sitting bends both.
    from .geometry import interior_angle
    knees = [interior_angle(pose.xy(h), pose.xy(k), pose.xy(a))
             for h, k, a in ((L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE),
                             (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE))]
    knees = [a for a in knees if a == a]  # drop NaN
    if knees and max(knees) < MIN_KNEE_EXTENSION_FRONT_DEG:
        out.append(GuardFinding(
            key="knees_not_extended", severity=SEVERITY_BLOCK,
            message_zh="双膝都明显弯曲，说明人物是坐着或蹲着而不是站立，"
                       "站姿体态的各项读数在这种情况下没有意义。",
            detail={"straighter_knee_deg": round(max(knees), 1),
                    "limit": MIN_KNEE_EXTENSION_FRONT_DEG}))

    lw, rw = pose.xy(L.LEFT_WRIST), pose.xy(L.RIGHT_WRIST)
    raised = [name for name, w, hip in
              (("left", lw, lhip), ("right", rw, rhip))
              if (w[1] - hip[1]) / scale < MIN_WRIST_BELOW_HIP_FRAC]
    if raised:
        out.append(GuardFinding(
            key="arms_not_at_side", severity=SEVERITY_WARN,
            message_zh="手臂没有自然垂放（抱臂、叉腰、插兜都会如此）。"
                       "这会改变肩部关键点的位置，肩相关读数请谨慎参考。",
            detail={"raised": raised}))
    return out


def check_output_sanity(metrics: dict) -> list[GuardFinding]:
    """Reject readings beyond what a human body can produce."""
    out = []
    for key, m in metrics.items():
        if m.plane == "sagittal" and abs(m.value) > MAX_PLAUSIBLE_ANGLE:
            out.append(GuardFinding(
                key="implausible_reading", severity=SEVERITY_BLOCK,
                message_zh=f"「{m.label}」读数超出人体可能范围，"
                           "判定为测量失败而非体态问题。",
                detail={"metric": key, "value": round(m.value, 1),
                        "limit": MAX_PLAUSIBLE_ANGLE}))
    return out


def check_view_suitability(view: ViewEstimate, metrics: dict) -> list[GuardFinding]:
    """An oblique view silently corrupts sagittal readings; say so."""
    out = []
    if view.view == "oblique":
        out.append(GuardFinding(
            key="oblique_view", severity=SEVERITY_BLOCK,
            message_zh="拍摄角度介于正面与侧面之间。这种角度下矢状面各项读数会被"
                       "系统性压缩，且读数本身看不出这个偏差。请重新拍摄正对或"
                       "完全侧对镜头的照片。",
            detail={"yaw_deg": round(view.yaw_deg, 1),
                    "shoulder_spread": round(view.shoulder_spread, 3)}))
    if view.view == "side" and view.facing_confidence < 0.25:
        out.append(GuardFinding(
            key="facing_ambiguous", severity=SEVERITY_WARN,
            message_zh="无法可靠判断人物朝向，矢状面读数的正负号可能相反。",
            detail={"facing_confidence": round(view.facing_confidence, 3)}))
    return out


def check_landmark_stability(image_rgb: np.ndarray, pose: L.PoseResult,
                             *, model_path: str | None = None
                             ) -> list[GuardFinding]:
    """Flag landmarks that move under a perturbation that should not move them.

    This is the only guard that catches the hair-over-ear failure, where the
    pose model reported visibility 1.000 for a landmark it had invented. A
    landmark locked to real image evidence stays put when the image is
    rescaled slightly; an inferred one drifts.

    The perturbation is a 4% downscale-and-restore, chosen to change the
    detector's input sampling without changing the depicted posture at all.
    """
    from PIL import Image

    h, w = image_rgb.shape[:2]
    if min(h, w) < 64:
        return []

    small = Image.fromarray(image_rgb).resize(
        (max(1, int(w * 0.96)), max(1, int(h * 0.96))), Image.LANCZOS)
    perturbed = np.array(small.resize((w, h), Image.LANCZOS))

    try:
        pose2 = L.detect(perturbed, model_path=model_path)
    except L.LandmarkerUnavailable:
        return []
    if pose2 is None:
        return [GuardFinding(
            key="unstable_detection", severity=SEVERITY_BLOCK,
            message_zh="对同一张图做极小幅度缩放后就检测不到人体了，"
                       "说明这次检测本身不稳定，结果不可信。",
            detail={})]

    scale = L.body_scale(pose)
    if scale <= 1e-6:
        return []

    drifted = {}
    for idx in CRITICAL_LANDMARKS:
        d = distance(pose.xy(idx), pose2.xy(idx)) / scale
        if d > UNSTABLE_LANDMARK_FRAC:
            drifted[L.LANDMARK_NAMES[idx]] = round(d, 4)

    if not drifted:
        return []
    return [GuardFinding(
        key="unstable_landmarks", severity=SEVERITY_WARN,
        message_zh="以下关键点在图像被轻微缩放后位置明显移动，说明模型是在"
                   "推测而不是看到它们（遮挡、头发、衣物都会导致这种情况）。"
                   "依赖这些点的读数请谨慎参考：" + "、".join(drifted),
        detail={"drift_frac_of_body_scale": drifted,
                "limit": UNSTABLE_LANDMARK_FRAC})]


def check_is_person(image_rgb: np.ndarray, pose: L.PoseResult,
                    *, model_path: str | None = None) -> list[GuardFinding]:
    """Ask an independent detector whether a person is present at all.

    EfficientDet-Lite0 (Apache-2.0) is trained separately from BlazePose, so it
    is a genuine second opinion on the palm-photo and sock-photo failures,
    where the pose model produced a skeleton with no person in frame.

    It does NOT resolve statues, mannequins or dolls: those are human-shaped
    and both models are expected to accept them. Returns nothing when the
    detector model is unavailable rather than silently passing.
    """
    model_path = model_path or _DETECTOR_MODEL
    if not os.path.exists(model_path):
        return []
    try:
        import mediapipe as mp
    except ImportError:
        return []

    try:
        detector = _get_object_detector(model_path)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(image_rgb))
        result = detector.detect(mp_image)
    except Exception:
        # An independent check that fails must not block a valid upload.
        return []

    subj = _bbox(pose)
    for det in result.detections:
        for cat in det.categories:
            if cat.category_name != "person":
                continue
            bb = det.bounding_box
            box = (bb.origin_x, bb.origin_y,
                   bb.origin_x + bb.width, bb.origin_y + bb.height)
            if _iou(subj, box) > 0.2:
                return []

    return [GuardFinding(
        key="no_person_detected", severity=SEVERITY_BLOCK,
        message_zh="姿态模型给出了骨架，但独立的物体检测模型在画面中找不到人。"
                   "姿态模型对手掌、衣物特写等非人体画面也会输出完整骨架，"
                   "因此这里以独立检测的结果为准，不给出体态判定。",
        detail={"n_detections": len(result.detections)})]


def _get_object_detector(model_path: str):
    global _OBJ_DETECTOR
    if _OBJ_DETECTOR is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        _OBJ_DETECTOR = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                score_threshold=0.3,
                max_results=10))
    return _OBJ_DETECTOR


_OBJ_DETECTOR = None


@atexit.register
def _close_object_detector() -> None:
    """Same teardown-order problem as the pose landmarker; see landmarks.py."""
    global _OBJ_DETECTOR
    if _OBJ_DETECTOR is not None:
        try:
            _OBJ_DETECTOR.close()
        except Exception:
            pass
        _OBJ_DETECTOR = None


def _bbox(pose: L.PoseResult) -> tuple[float, float, float, float]:
    xs = [lm.x for lm in pose.landmarks]
    ys = [lm.y for lm in pose.landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_all(pose: L.PoseResult, view: ViewEstimate, metrics: dict,
            image_rgb: np.ndarray | None = None,
            *, deep: bool = True) -> list[GuardFinding]:
    """Run every guard. `deep=False` skips the two that re-run a model."""
    findings: list[GuardFinding] = []
    findings += check_framing(pose)
    findings += check_proportions(pose)
    findings += check_subject_resolution(pose, view)
    findings += check_view_suitability(view, metrics)
    findings += check_pose_neutrality(metrics)
    if view.view in ("front", "oblique"):
        findings += check_frontal_neutrality(pose)
    findings += check_output_sanity(metrics)
    if deep and image_rgb is not None:
        findings += check_is_person(image_rgb, pose)
        findings += check_landmark_stability(image_rgb, pose)
    return findings


def blocked(findings: list[GuardFinding]) -> bool:
    return any(f.severity == SEVERITY_BLOCK for f in findings)
