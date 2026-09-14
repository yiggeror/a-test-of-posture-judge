"""Posture geometry.

All computation happens in PIXEL space. MediaPipe normalises x by image width
and y by image height, so an angle taken from raw normalised coordinates is
wrong by the image aspect ratio. `Pt` values reaching this module are already
in pixels.

Image convention: x grows right, y grows DOWN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .landmarks import (
    LEFT_ANKLE, LEFT_EAR, LEFT_HIP, LEFT_SHOULDER, NOSE,
    RIGHT_ANKLE, RIGHT_EAR, RIGHT_HIP, RIGHT_SHOULDER, Pt,
)
from .thresholds import (
    FORWARD_HEAD, MIN_VISIBILITY, ROUNDED_SHOULDER, SAGITTAL_ALIGNMENT,
    SHOULDER_TILT, VIEW_RATIO_FRONT_MIN, VIEW_RATIO_SIDE_MAX, Threshold,
)

Band = Literal["typical", "mild", "pronounced", "unavailable"]
View = Literal["front", "side", "oblique", "unknown"]


@dataclass
class Metric:
    key: str
    label_zh: str
    label_en: str
    value: float | None
    unit: str
    band: Band
    threshold: Threshold | None
    detail: str = ""
    advice: str = ""
    caveats: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.value is not None


def _dist(a: Pt, b: Pt) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _mid(a: Pt, b: Pt) -> Pt:
    return Pt((a.x + b.x) / 2, (a.y + b.y) / 2,
              min(a.visibility, b.visibility))


def _band(value: float, t: Threshold) -> Band:
    v = abs(value)
    if v >= t.marked:
        return "pronounced"
    if v >= t.mild:
        return "mild"
    return "typical"


# --------------------------------------------------------------------------
# View and facing
# --------------------------------------------------------------------------

def detect_view(pts: list[Pt]) -> tuple[View, float]:
    """Classify frontal vs sagittal from shoulder width / torso length.

    Returns (view, ratio). The ratio is reported so the UI can show why.
    """
    ls, rs = pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER]
    lh, rh = pts[LEFT_HIP], pts[RIGHT_HIP]
    shoulder_w = _dist(ls, rs)
    torso = _dist(_mid(ls, rs), _mid(lh, rh))
    if torso <= 1e-6:
        return "unknown", 0.0
    ratio = shoulder_w / torso
    if ratio <= VIEW_RATIO_SIDE_MAX:
        return "side", ratio
    if ratio >= VIEW_RATIO_FRONT_MIN:
        return "front", ratio
    return "oblique", ratio


def facing_sign(pts: list[Pt]) -> float:
    """+1 if the subject faces image-right, -1 if image-left.

    Uses the nose relative to the ear midpoint: the nose is always anterior to
    the ears. On a frontal view the two nearly coincide and the sign is close
    to meaningless, which is why anterior/posterior metrics are only reported
    for sagittal views.
    """
    ears = [p for p in (pts[LEFT_EAR], pts[RIGHT_EAR])
            if p.visibility >= MIN_VISIBILITY]
    if not ears:
        return 1.0
    ear_x = sum(p.x for p in ears) / len(ears)
    dx = pts[NOSE].x - ear_x
    return 1.0 if dx >= 0 else -1.0


def _sagittal_side(pts: list[Pt]) -> tuple[int, int, int, int, str]:
    """Pick the body side with the more reliable landmarks for a side view."""
    left_score = min(pts[LEFT_EAR].visibility, pts[LEFT_SHOULDER].visibility,
                     pts[LEFT_HIP].visibility)
    right_score = min(pts[RIGHT_EAR].visibility, pts[RIGHT_SHOULDER].visibility,
                      pts[RIGHT_HIP].visibility)
    if left_score >= right_score:
        return LEFT_EAR, LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE, "left"
    return RIGHT_EAR, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE, "right"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def forward_head(pts: list[Pt], view: View) -> Metric:
    """Angle of the acromion->tragus line from vertical, sagittal view.

    NOTE the landmark substitution: MediaPipe's `ear` landmark sits near the
    tragus but is not a tragus, and there is no C7 landmark at all.
    """
    ear_i, sh_i, _, _, side = _sagittal_side(pts)
    ear, sh = pts[ear_i], pts[sh_i]
    m = Metric("forward_head", "头前引角", "Forward head angle", None, "deg",
               "unavailable", FORWARD_HEAD)
    m.advice = (
        "参考方向：若数值偏高，倾向提示头部相对肩峰前移。常见对应做法是下巴回收"
        "（chin tuck）、调整屏幕到视线高度、以及减少长时间低头。"
    )
    if view not in ("side", "oblique"):
        m.detail = "需要侧面照。正面照下耳与肩峰几乎重叠，该角度没有意义。"
        m.caveats.append("正面视角，未计算。")
        return m
    if ear.visibility < MIN_VISIBILITY or sh.visibility < MIN_VISIBILITY:
        m.detail = f"耳({ear.visibility:.2f})或肩峰({sh.visibility:.2f})可见度不足。"
        return m

    sign = facing_sign(pts)
    forward = (ear.x - sh.x) * sign      # + = ear anterior to acromion
    rise = sh.y - ear.y                  # + = ear above shoulder (image y down)
    if rise <= 1e-6:
        m.detail = "耳未位于肩峰上方，几何关系异常，不出数。"
        return m
    m.value = math.degrees(math.atan2(forward, rise))
    m.band = _band(m.value, FORWARD_HEAD)
    m.detail = (f"使用{'左' if side=='left' else '右'}侧；耳相对肩峰前移 "
                f"{forward:.0f}px，垂直高差 {rise:.0f}px。")
    if view == "oblique":
        m.caveats.append("视角判定为斜侧，角度会被透视压缩，数值偏低。")
    m.caveats.append("以图像竖直方向为铅垂线；相机若有俯仰/翻滚，误差直接进入该角度。")
    return m


def rounded_shoulder(pts: list[Pt], view: View) -> Metric:
    """Horizontal offset of the acromion relative to the ear lobe.

    Implemented exactly as specified (acromion measured against the ear), and
    normalised by torso length so it is scale free.

    IMPORTANT, and reported in the UI: along the anterior/posterior axis this
    quantity is the NEGATIVE of the forward-head displacement, because both use
    the same two points. With only the ear and the acromion, "head forward" and
    "shoulder forward" are one measurement, not two. `plumb_offsets` below adds
    the independent version measured against the ankle plumb line.
    """
    ear_i, sh_i, hip_i, _, side = _sagittal_side(pts)
    ear, sh, hip = pts[ear_i], pts[sh_i], pts[hip_i]
    m = Metric("rounded_shoulder", "圆肩（肩峰相对耳垂水平偏移）",
               "Rounded shoulder offset", None, "x torso length",
               "unavailable", ROUNDED_SHOULDER)
    m.advice = (
        "参考方向：正值倾向提示肩峰位于耳垂前方。常见对应做法是加强胸椎伸展与"
        "肩胛后缩（如 wall slide、face pull），并放松胸小肌。"
    )
    if view not in ("side", "oblique"):
        m.detail = "需要侧面照。"
        m.caveats.append("正面视角，未计算。")
        return m
    if min(ear.visibility, sh.visibility, hip.visibility) < MIN_VISIBILITY:
        m.detail = "耳、肩峰或髋可见度不足。"
        return m
    torso = _dist(sh, hip)
    if torso <= 1e-6:
        m.detail = "躯干长度为零，无法归一化。"
        return m
    sign = facing_sign(pts)
    offset_px = (sh.x - ear.x) * sign     # + = acromion anterior to ear
    m.value = offset_px / torso
    m.band = _band(m.value, ROUNDED_SHOULDER)
    m.detail = (f"使用{'左' if side=='left' else '右'}侧；肩峰相对耳垂前移 "
                f"{offset_px:+.0f}px，躯干长 {torso:.0f}px。正值=肩峰在前。")
    m.caveats.append(
        "与「头前引角」共用耳-肩峰两点，二者沿前后轴互为反号，并非独立指标。"
        "下方的铅垂线偏移给出相互独立的版本。")
    return m


def sagittal_alignment(pts: list[Pt], view: View) -> Metric:
    """180 deg minus the shoulder-hip-ankle angle.

    The brief labels this "pelvic tilt"; it is not, and the UI says so.
    """
    _, sh_i, hip_i, ank_i, side = _sagittal_side(pts)
    sh, hip, ank = pts[sh_i], pts[hip_i], pts[ank_i]
    m = Metric("sagittal_alignment", "躯干矢状面对齐（肩-髋-踝）",
               "Sagittal trunk alignment", None, "deg", "unavailable",
               SAGITTAL_ALIGNMENT)
    m.advice = (
        "参考方向：偏差越大，倾向提示站立时肩-髋-踝偏离一条直线（髋前推或臀后坐）。"
        "常见对应做法是练习中立位站姿与髋屈肌/臀肌的平衡。"
    )
    if view not in ("side", "oblique"):
        m.detail = "需要侧面照。"
        m.caveats.append("正面视角，未计算。")
        return m
    if min(sh.visibility, hip.visibility, ank.visibility) < MIN_VISIBILITY:
        m.detail = (f"肩({sh.visibility:.2f})/髋({hip.visibility:.2f})/"
                    f"踝({ank.visibility:.2f})可见度不足，需要全身入镜。")
        return m
    v1 = (sh.x - hip.x, sh.y - hip.y)
    v2 = (ank.x - hip.x, ank.y - hip.y)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 <= 1e-6 or n2 <= 1e-6:
        m.detail = "关键点重合，无法成角。"
        return m
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    interior = math.degrees(math.acos(cos))
    deviation = 180.0 - interior
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    sign = facing_sign(pts)
    m.value = deviation * (1.0 if cross * sign >= 0 else -1.0)
    m.band = _band(m.value, SAGITTAL_ALIGNMENT)
    m.detail = (f"使用{'左' if side=='left' else '右'}侧；肩-髋-踝夹角 "
                f"{interior:.1f}°，偏离直线 {deviation:.1f}°。")
    m.caveats.append("这是躯干整体矢状面对齐，不是骨盆前后倾；"
                     "MediaPipe 无 ASIS/PSIS 关键点，真正的骨盆倾斜测不了。")
    return m


def shoulder_tilt(pts: list[Pt], view: View) -> Metric:
    """Tilt of the biacromial line from the image horizontal, frontal view."""
    ls, rs = pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER]
    m = Metric("shoulder_tilt", "高低肩", "Shoulder height asymmetry",
               None, "deg", "unavailable", SHOULDER_TILT)
    m.advice = (
        "参考方向：数值偏大倾向提示左右肩峰高度不等。先排除相机翻滚与站姿重心"
        "偏移；持续存在可关注单侧负重习惯。"
    )
    if view not in ("front", "oblique"):
        m.detail = "需要正面照。侧面照下两侧肩峰前后重叠，高差不可读。"
        m.caveats.append("侧面视角，未计算。")
        return m
    if min(ls.visibility, rs.visibility) < MIN_VISIBILITY:
        m.detail = "双侧肩峰可见度不足。"
        return m
    dx = abs(ls.x - rs.x)
    if dx <= 1e-6:
        m.detail = "两肩峰水平投影重合。"
        return m
    dy = ls.y - rs.y     # + = left shoulder lower (image y down)
    m.value = math.degrees(math.atan2(dy, dx))
    m.band = _band(m.value, SHOULDER_TILT)
    higher = "右肩偏高" if dy > 0 else "左肩偏高"
    m.detail = (f"左右肩峰垂直高差 {abs(dy):.0f}px，肩宽 {dx:.0f}px；{higher}。"
                f"正值=左肩较低。")
    if view == "oblique":
        m.caveats.append("视角判定为斜侧，一侧肩峰更靠近相机会被透视放大。")
    m.caveats.append("相机翻滚角会 1:1 叠加到该数值上，单张照片无法分离。")
    return m


def plumb_offsets(pts: list[Pt], view: View) -> Metric:
    """Anterior offset of ear and acromion from a vertical line at the ankle.

    This is the clinical plumb-line convention (the line passes near the
    lateral malleolus, the greater trochanter, the acromion and the ear lobe).
    Because both offsets share one external reference instead of referencing
    each other, forward head and rounded shoulder become independent here.

    Reported as supplementary context. No threshold - we have no defensible
    cut-off for it, so it is shown as a number only.
    """
    ear_i, sh_i, hip_i, ank_i, side = _sagittal_side(pts)
    ear, sh, hip, ank = pts[ear_i], pts[sh_i], pts[hip_i], pts[ank_i]
    m = Metric("plumb", "铅垂线偏移（以踝为基准）", "Plumb-line offsets",
               None, "x torso length", "unavailable", None)
    m.advice = ("参考用途：三个偏移共享同一条基准线，可用来区分究竟是头更靠前"
                "还是肩更靠前。无阈值判定。")
    if view not in ("side", "oblique"):
        m.detail = "需要侧面照。"
        return m
    if min(ear.visibility, sh.visibility, hip.visibility,
           ank.visibility) < MIN_VISIBILITY:
        m.detail = "需要耳、肩、髋、踝同时可见（全身侧面入镜）。"
        return m
    torso = _dist(sh, hip)
    if torso <= 1e-6:
        m.detail = "躯干长度为零。"
        return m
    sign = facing_sign(pts)
    ear_off = (ear.x - ank.x) * sign / torso
    sh_off = (sh.x - ank.x) * sign / torso
    hip_off = (hip.x - ank.x) * sign / torso
    m.value = ear_off
    m.band = "typical"
    m.detail = (f"相对踝部铅垂线（正=偏前，单位=躯干长）："
                f"耳 {ear_off:+.3f}，肩峰 {sh_off:+.3f}，髋 {hip_off:+.3f}。")
    m.caveats.append("无阈值：未找到可信的归一化参考值，故只给数不判定。")
    return m


def compute_for_view(pts: list[Pt], view: View) -> list[Metric]:
    """All metrics for an explicitly chosen view."""
    return [
        forward_head(pts, view),
        rounded_shoulder(pts, view),
        sagittal_alignment(pts, view),
        shoulder_tilt(pts, view),
        plumb_offsets(pts, view),
    ]


def compute_all(pts: list[Pt]) -> tuple[View, float, list[Metric], Plausibility]:
    """Detect the view, sanity-check the pose, then compute every metric.

    If the pose fails the plausibility check, every metric is returned as
    `unavailable` rather than as a number. Reporting "shoulder tilt -56.8 deg,
    pronounced" for a photograph of a hand - which is exactly what this code
    used to do - is worse than reporting nothing.
    """
    view, ratio = detect_view(pts)
    plaus = check_plausibility(pts)
    if not plaus.ok:
        blocked = []
        for m in compute_for_view(pts, view):
            m.value = None
            m.band = "unavailable"
            m.detail = "姿态合理性检查未通过，不出数。"
            # Reasons are shown once in the page-level banner, not repeated on
            # each of the five metric cards.
            m.caveats = []
            blocked.append(m)
        return view, ratio, blocked, plaus
    return view, ratio, compute_for_view(pts, view), plaus


# --------------------------------------------------------------------------
# Plausibility
# --------------------------------------------------------------------------

@dataclass
class Plausibility:
    ok: bool
    torso_tilt_deg: float | None
    head_torso_ratio: float | None
    reasons: list[str] = field(default_factory=list)


def check_plausibility(pts: list[Pt]) -> Plausibility:
    """Reject poses that cannot be a standing person.

    Motivated by a real Phase 2 failure: MediaPipe returned a complete,
    high-confidence 33-landmark skeleton for a photo of a hand, and every
    downstream metric duly produced a number. `visibility` does not catch this
    - it estimates per-landmark occlusion, not whether a person is present - so
    these anatomical checks do the job instead.

    This reduces obvious false positives. It is not a person detector, and it
    will not catch a subtly wrong skeleton.
    """
    from .thresholds import (MAX_HEAD_TORSO_RATIO, MAX_TORSO_TILT_DEG,
                             MIN_HEAD_TORSO_RATIO)

    reasons: list[str] = []
    sh_mid = _mid(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER])
    hip_mid = _mid(pts[LEFT_HIP], pts[RIGHT_HIP])
    ear_mid = _mid(pts[LEFT_EAR], pts[RIGHT_EAR])

    torso = _dist(sh_mid, hip_mid)
    if torso <= 1e-6:
        return Plausibility(False, None, None, ["躯干长度为零，关键点退化。"])

    # Torso axis vs image vertical.
    dx = hip_mid.x - sh_mid.x
    dy = hip_mid.y - sh_mid.y
    tilt = math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 90.0
    if tilt > MAX_TORSO_TILT_DEG:
        reasons.append(
            f"躯干轴线偏离铅垂线 {tilt:.0f}°（上限 {MAX_TORSO_TILT_DEG:.0f}°）。"
            "本工具的全部指标都以「直立站姿」为前提，该前提在此不成立。")

    ratio = _dist(ear_mid, sh_mid) / torso
    if not (MIN_HEAD_TORSO_RATIO <= ratio <= MAX_HEAD_TORSO_RATIO):
        reasons.append(
            f"头颈段/躯干长 = {ratio:.2f}，落在人体常见范围 "
            f"{MIN_HEAD_TORSO_RATIO}–{MAX_HEAD_TORSO_RATIO} 之外，"
            "关键点很可能不是落在真实人体上。")

    return Plausibility(not reasons, tilt, ratio, reasons)
