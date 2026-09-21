"""Synthetic pose builder.

Lets the metric, view, guard and threshold layers be tested exactly, with no
detector involved. A synthetic skeleton has a KNOWN forward-head angle, so
these tests check that the measurement recovers the number it was given --
which is the one thing real photographs can never be used to check, because
nobody knows the true angle of a person in a photograph.
"""
from __future__ import annotations

import math

import pytest

from posture import landmarks as L


def _rot_about(px, py, cx, cy, deg):
    r = math.radians(deg)
    dx, dy = px - cx, py - cy
    return (cx + dx * math.cos(r) - dy * math.sin(r),
            cy + dx * math.sin(r) + dy * math.cos(r))


def make_pose(*, view: str = "side",
              forward_head_deg: float = 0.0,
              shoulder_protraction_deg: float = 0.0,
              trunk_sway_deg: float = 0.0,
              knee_deviation_deg: float = 0.0,
              shoulder_tilt_deg: float = 0.0,
              pelvis_tilt_deg: float = 0.0,
              head_tilt_deg: float = 0.0,
              facing: int = +1,
              width: int = 800, height: int = 1100,
              visibility: float = 0.99) -> L.PoseResult:
    """Build a standing figure with the given angles baked in.

    Segment lengths are in pixels and roughly proportional to a real adult:
    ankle->hip 430, hip->shoulder 300, shoulder->ear 90.

    Angles are applied by placing each upper joint at the requested angle from
    vertical above its lower joint, which is exactly the inverse of what
    metrics.py computes, so a correct implementation round-trips.
    """
    cx = width / 2.0
    ankle_y, hip_len, torso_len, neck_len = 980.0, 430.0, 300.0, 90.0

    ank = (cx, ankle_y)
    # hip sits `trunk_sway_deg` from vertical above the ankle
    t = math.radians(trunk_sway_deg)
    hip = (ank[0] + facing * hip_len * math.sin(t), ank[1] - hip_len * math.cos(t))
    s = math.radians(shoulder_protraction_deg)
    sh = (hip[0] + facing * torso_len * math.sin(s), hip[1] - torso_len * math.cos(s))
    f = math.radians(forward_head_deg)
    ear = (sh[0] + facing * neck_len * math.sin(f), sh[1] - neck_len * math.cos(f))

    # knee: on the hip-ankle line, then pushed off it by the requested angle
    kx, ky = (hip[0] + ank[0]) / 2.0, (hip[1] + ank[1]) / 2.0
    if knee_deviation_deg:
        # deviation is the supplement of the interior angle; displace the knee
        # perpendicular to the hip-ankle line by the matching amount
        half = hip_len / 2.0
        off = half * math.tan(math.radians(knee_deviation_deg) / 2.0)
        ux, uy = ank[0] - hip[0], ank[1] - hip[1]
        n = math.hypot(ux, uy) or 1.0
        # Unit normal pointing ANTERIORLY for this subject: with the hip above
        # the ankle, u is (0, +L), so this gives (+1, 0), i.e. toward +x, which
        # is the front of a subject facing +x. Positive knee_deviation means
        # flexed, so the knee must move to the front.
        px, py = uy / n, -ux / n
        kx += facing * off * px
        ky += facing * off * py

    # In an ideal lateral projection a symmetric body's left and right
    # landmarks fall on the same point, so the half-offsets are exactly zero.
    # Any non-zero value here would displace whichever side `near_side_indices`
    # picks away from the sagittal axis and make the round-trip inexact for a
    # reason that has nothing to do with the code under test.
    half_shoulder = 110.0 if view == "front" else 0.0
    half_hip = 78.0 if view == "front" else 0.0
    half_ankle = 60.0 if view == "front" else 0.0
    half_eye = 32.0 if view == "front" else 0.0

    pts = [(0.0, 0.0)] * L.N_LANDMARKS

    def put(idx, xy):
        pts[idx] = xy

    # Bilateral pairs, tilted where a tilt was requested. Subject's LEFT is on
    # the viewer's right in a front view, and a positive tilt means the
    # subject's left is higher (smaller y).
    def pair(left_idx, right_idx, cxy, half, tilt_deg):
        lx, ly = cxy[0] + half, cxy[1]
        rx, ry = cxy[0] - half, cxy[1]
        if tilt_deg:
            lx, ly = _rot_about(lx, ly, cxy[0], cxy[1], -tilt_deg)
            rx, ry = _rot_about(rx, ry, cxy[0], cxy[1], -tilt_deg)
        put(left_idx, (lx, ly))
        put(right_idx, (rx, ry))

    pair(L.LEFT_SHOULDER, L.RIGHT_SHOULDER, sh, half_shoulder, shoulder_tilt_deg)
    pair(L.LEFT_HIP, L.RIGHT_HIP, hip, half_hip, pelvis_tilt_deg)
    pair(L.LEFT_ANKLE, L.RIGHT_ANKLE, ank, half_ankle, 0.0)
    pair(L.LEFT_KNEE, L.RIGHT_KNEE, (kx, ky), half_ankle, 0.0)
    pair(L.LEFT_EAR, L.RIGHT_EAR, ear, half_eye * 1.4, head_tilt_deg)
    pair(L.LEFT_EYE, L.RIGHT_EYE, (ear[0] + facing * 22, ear[1] - 8),
         half_eye, head_tilt_deg)
    pair(L.LEFT_EYE_INNER, L.RIGHT_EYE_INNER,
         (ear[0] + facing * 22, ear[1] - 8), half_eye * 0.6, head_tilt_deg)
    pair(L.LEFT_EYE_OUTER, L.RIGHT_EYE_OUTER,
         (ear[0] + facing * 22, ear[1] - 8), half_eye * 1.3, head_tilt_deg)
    pair(L.MOUTH_LEFT, L.MOUTH_RIGHT, (ear[0] + facing * 30, ear[1] + 22),
         half_eye * 0.7, head_tilt_deg)
    # wrists hang beside the hips, below hip level
    pair(L.LEFT_WRIST, L.RIGHT_WRIST, (hip[0], hip[1] + 95), half_hip * 1.2, 0.0)
    pair(L.LEFT_ELBOW, L.RIGHT_ELBOW, (hip[0], hip[1] - 60), half_hip * 1.15, 0.0)
    pair(L.LEFT_PINKY, L.RIGHT_PINKY, (hip[0], hip[1] + 120), half_hip * 1.2, 0.0)
    pair(L.LEFT_INDEX, L.RIGHT_INDEX, (hip[0], hip[1] + 122), half_hip * 1.2, 0.0)
    pair(L.LEFT_THUMB, L.RIGHT_THUMB, (hip[0], hip[1] + 112), half_hip * 1.15, 0.0)
    pair(L.LEFT_HEEL, L.RIGHT_HEEL, (ank[0] - facing * 18, ank[1] + 18),
         half_ankle, 0.0)
    pair(L.LEFT_FOOT_INDEX, L.RIGHT_FOOT_INDEX,
         (ank[0] + facing * 55, ank[1] + 26), half_ankle, 0.0)

    put(L.NOSE, (ear[0] + facing * 44, ear[1] + 6))

    lms = [L.Landmark(x=x, y=y, z=0.0, visibility=visibility, presence=visibility)
           for x, y in pts]
    return L.PoseResult(landmarks=lms, width=width, height=height,
                        n_poses_detected=1)


def scale_pose(pose: L.PoseResult, factor: float) -> L.PoseResult:
    """Shrink or enlarge a pose and its canvas together.

    `make_pose` lays the figure out at fixed pixel coordinates, so passing a
    smaller canvas alone does not make the subject smaller. This scales the
    landmarks too, which is what a lower-resolution photograph of the same
    person actually looks like.
    """
    lms = [L.Landmark(x=lm.x * factor, y=lm.y * factor, z=lm.z,
                      visibility=lm.visibility, presence=lm.presence)
           for lm in pose.landmarks]
    return L.PoseResult(landmarks=lms,
                        width=max(1, int(pose.width * factor)),
                        height=max(1, int(pose.height * factor)),
                        n_poses_detected=pose.n_poses_detected,
                        other_pose_bboxes=list(pose.other_pose_bboxes))


@pytest.fixture
def side_pose():
    return make_pose(view="side")


@pytest.fixture
def front_pose():
    return make_pose(view="front")
