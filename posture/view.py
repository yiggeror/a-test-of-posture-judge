"""Camera-view estimation and facing direction.

Both are purely geometric, computed from landmark positions rather than from
MediaPipe's `z` channel. The `z` values are a weak monocular depth guess and
are not accurate enough to decide torso yaw; the projected shoulder separation
is.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import landmarks as L
from .geometry import distance, midpoint

# A fully front-facing adult projects a shoulder separation of ~0.29 of their
# shoulder-to-ankle height (biacromial width ~0.40 m over a ~1.38 m span). A
# true lateral view projects it to ~0. These cut points sit either side of that
# range, leaving a deliberately wide "oblique" band because a metric computed
# from an oblique view is wrong in a way the reading itself cannot reveal.
# provenance: geometric-estimate (anthropometric ratio, derived here)
SIDE_MAX_SPREAD = 0.14
FRONT_MIN_SPREAD = 0.22

VIEW_FRONT = "front"
VIEW_SIDE = "side"
VIEW_OBLIQUE = "oblique"

FACING_RIGHT = +1
FACING_LEFT = -1


@dataclass(frozen=True)
class ViewEstimate:
    view: str
    shoulder_spread: float   # normalised by body scale
    hip_spread: float
    facing: int              # +1 subject faces +x, -1 faces -x
    facing_confidence: float  # 0..1, how clearly the nose leads the ears
    yaw_deg: float           # 0 = full lateral, 90 = full frontal


def estimate_view(pose: L.PoseResult) -> ViewEstimate:
    """Classify the camera view and which way the subject faces."""
    scale = L.body_scale(pose)
    if scale <= 1.0:
        scale = 1.0

    lsh, rsh = pose.xy(L.LEFT_SHOULDER), pose.xy(L.RIGHT_SHOULDER)
    lhip, rhip = pose.xy(L.LEFT_HIP), pose.xy(L.RIGHT_HIP)

    shoulder_spread = abs(lsh[0] - rsh[0]) / scale
    hip_spread = abs(lhip[0] - rhip[0]) / scale

    # Use the shoulders as the primary cue; the hips agree but are noisier
    # because clothing hides the landmark.
    spread = shoulder_spread
    if spread <= SIDE_MAX_SPREAD:
        view = VIEW_SIDE
    elif spread >= FRONT_MIN_SPREAD:
        view = VIEW_FRONT
    else:
        view = VIEW_OBLIQUE

    # Torso yaw: asin of the projected spread over the expected frontal spread.
    # 90 deg means square to the camera, 0 means fully lateral.
    import math
    ratio = min(1.0, spread / 0.29)
    yaw_deg = math.degrees(math.asin(ratio))

    facing, conf = _facing(pose, scale)
    return ViewEstimate(view=view, shoulder_spread=shoulder_spread,
                        hip_spread=hip_spread, facing=facing,
                        facing_confidence=conf, yaw_deg=yaw_deg)


def _facing(pose: L.PoseResult, scale: float) -> tuple[int, float]:
    """Decide which way the subject faces, from nose-versus-ears offset.

    In any non-frontal view the nose projects anterior to the ear midpoint.
    The confidence is that offset normalised by the ear separation, clipped to
    1.0; near-frontal views give a small offset and therefore low confidence,
    which is correct -- facing direction is genuinely ambiguous head-on.
    """
    nose = pose.xy(L.NOSE)
    lear, rear = pose.xy(L.LEFT_EAR), pose.xy(L.RIGHT_EAR)
    ear_mid = midpoint(lear, rear)

    dx = nose[0] - ear_mid[0]
    # Head size reference: ear separation in a frontal view, but it collapses
    # laterally, so fall back to a fraction of body scale.
    head_ref = max(distance(lear, rear), scale * 0.06)
    conf = min(1.0, abs(dx) / head_ref) if head_ref > 0 else 0.0
    facing = FACING_RIGHT if dx >= 0 else FACING_LEFT
    return facing, conf


def near_ear_index(pose: L.PoseResult, facing: int) -> int:
    """Pick the ear landmark closest to the camera in a lateral view.

    In a side view the far ear is occluded and MediaPipe hallucinates its
    position, so metrics must use the near ear. The near ear is the one
    displaced toward the camera; with the subject facing `facing`, that is the
    ear whose own visibility is higher, with the position offset as a
    tiebreaker.
    """
    lv = pose.visibility(L.LEFT_EAR)
    rv = pose.visibility(L.RIGHT_EAR)
    if abs(lv - rv) > 0.15:
        return L.LEFT_EAR if lv > rv else L.RIGHT_EAR
    # Visibility was uninformative. Fall back to the ear further from the nose
    # horizontally, i.e. the one on the camera side of the head.
    nose_x = pose.xy(L.NOSE)[0]
    ldx = abs(pose.xy(L.LEFT_EAR)[0] - nose_x)
    rdx = abs(pose.xy(L.RIGHT_EAR)[0] - nose_x)
    return L.LEFT_EAR if ldx > rdx else L.RIGHT_EAR


def near_side_indices(pose: L.PoseResult, facing: int) -> dict[str, int]:
    """Return the camera-side landmark index for each bilateral joint.

    Same reasoning as `near_ear_index`: in a lateral view the far-side joints
    are occluded and their positions are inferred, so sagittal metrics should
    be built from the near side wherever possible.
    """
    ear = near_ear_index(pose, facing)
    is_left = ear == L.LEFT_EAR
    if is_left:
        return {"ear": L.LEFT_EAR, "shoulder": L.LEFT_SHOULDER, "hip": L.LEFT_HIP,
                "knee": L.LEFT_KNEE, "ankle": L.LEFT_ANKLE, "heel": L.LEFT_HEEL,
                "foot": L.LEFT_FOOT_INDEX}
    return {"ear": L.RIGHT_EAR, "shoulder": L.RIGHT_SHOULDER, "hip": L.RIGHT_HIP,
            "knee": L.RIGHT_KNEE, "ankle": L.RIGHT_ANKLE, "heel": L.RIGHT_HEEL,
            "foot": L.RIGHT_FOOT_INDEX}
