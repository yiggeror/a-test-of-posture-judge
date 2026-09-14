"""Overlay drawing: skeleton, landmarks, and the measurement construction lines."""
from __future__ import annotations

import base64

import cv2
import numpy as np

from .landmarks import (
    LEFT_ANKLE, LEFT_EAR, LEFT_HIP, LEFT_SHOULDER, RIGHT_ANKLE, RIGHT_EAR,
    RIGHT_HIP, RIGHT_SHOULDER, SKELETON, Pt,
)
from .metrics import View, _sagittal_side
from .thresholds import MIN_VISIBILITY

BONE = (232, 180, 74)      # BGR
JOINT = (255, 255, 255)
JOINT_EDGE = (40, 40, 40)
PLUMB = (120, 220, 120)
MEASURE = (90, 120, 250)
LOW_VIS = (110, 110, 110)


def _p(pt: Pt) -> tuple[int, int]:
    return int(round(pt.x)), int(round(pt.y))


def draw(bgr: np.ndarray, pts: list[Pt], view: View) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    thick = max(2, int(round(min(h, w) / 400)))
    r = max(3, int(round(min(h, w) / 220)))

    for a, b in SKELETON:
        pa, pb = pts[a], pts[b]
        lo = min(pa.visibility, pb.visibility) < MIN_VISIBILITY
        cv2.line(img, _p(pa), _p(pb), LOW_VIS if lo else BONE,
                 max(1, thick - 1) if lo else thick, cv2.LINE_AA)

    for pt in pts:
        col = LOW_VIS if pt.visibility < MIN_VISIBILITY else JOINT
        cv2.circle(img, _p(pt), r, col, -1, cv2.LINE_AA)
        cv2.circle(img, _p(pt), r, JOINT_EDGE, 1, cv2.LINE_AA)

    # Construction lines for whichever metrics were actually computed.
    if view in ("side", "oblique"):
        ear_i, sh_i, hip_i, ank_i, _ = _sagittal_side(pts)
        ear, sh, ank = pts[ear_i], pts[sh_i], pts[ank_i]
        if min(ear.visibility, sh.visibility) >= MIN_VISIBILITY:
            # vertical (plumb) through the acromion + the acromion->ear line
            cv2.line(img, (int(sh.x), 0), (int(sh.x), h), PLUMB, 1, cv2.LINE_AA)
            cv2.line(img, _p(sh), _p(ear), MEASURE, thick, cv2.LINE_AA)
        if ank.visibility >= MIN_VISIBILITY:
            cv2.line(img, (int(ank.x), 0), (int(ank.x), h), PLUMB, 1,
                     cv2.LINE_AA)
    if view in ("front", "oblique"):
        ls, rs = pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER]
        if min(ls.visibility, rs.visibility) >= MIN_VISIBILITY:
            cv2.line(img, _p(ls), _p(rs), MEASURE, thick, cv2.LINE_AA)
            y = int(round((ls.y + rs.y) / 2))
            cv2.line(img, (int(min(ls.x, rs.x)), y), (int(max(ls.x, rs.x)), y),
                     PLUMB, 1, cv2.LINE_AA)
    return img


def to_png_data_uri(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
