"""Pure geometry helpers.

Everything here operates on plain (x, y) tuples in *image* coordinates, where
+x points right and +y points DOWN. That sign convention is the single most
common source of angle bugs in this codebase, so every function that cares
about it says so in its docstring.

No MediaPipe imports live in this module: it is exactly testable with
synthetic points, which is what `tests/test_geometry.py` does.
"""
from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def angle_from_vertical(lower: Point, upper: Point, *, anterior: int = +1) -> float:
    """Signed angle in degrees between the `lower -> upper` vector and straight up.

    Image coordinates (+y down), so "straight up" is (0, -1).

    Returns 0.0 when `upper` sits directly above `lower`. The sign is set so a
    POSITIVE result means `upper` is displaced toward the subject's ANTERIOR
    (front) side, given `anterior`:

        anterior = +1  the subject faces +x (to the right of frame)
        anterior = -1  the subject faces -x (to the left of frame)

    This is the workhorse for every sagittal metric: forward-head, shoulder
    protraction and trunk sway are all "how far is the upper point tipped in
    front of the lower point".
    """
    dx = (upper[0] - lower[0]) * anterior
    dy = upper[1] - lower[1]
    # atan2(horizontal, -vertical): -dy because up is negative in image coords.
    return math.degrees(math.atan2(dx, -dy))


def angle_from_horizontal(left: Point, right: Point, *, invert_y: bool = True) -> float:
    """Signed angle in degrees of the `left -> right` vector away from horizontal.

    With `invert_y` (the default, and correct for image coordinates), a POSITIVE
    result means the `right` point is HIGHER in the picture than the `left`
    point. Used for the frontal-plane levelness metrics (shoulder tilt, pelvis
    tilt, head tilt).
    """
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    if invert_y:
        dy = -dy
    if dx == 0 and dy == 0:
        return 0.0
    # Fold to (-90, 90]: left/right label order must not flip the sign.
    ang = math.degrees(math.atan2(dy, dx))
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    return ang


def interior_angle(a: Point, vertex: Point, c: Point) -> float:
    """Unsigned interior angle ABC in degrees, in [0, 180].

    Used for joint flexion/extension, e.g. hip-knee-ankle. 180 deg is a fully
    straight joint.
    """
    v1 = (a[0] - vertex[0], a[1] - vertex[1])
    v2 = (c[0] - vertex[0], c[1] - vertex[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def rotate_point(p: Point, center: Point, degrees: float) -> Point:
    """Rotate `p` about `center` by `degrees`, CLOCKWISE on screen.

    In image coordinates (+y down) a positive mathematical rotation appears
    clockwise, which is what an image-rotation filter of the same sign
    produces. The repeatability harness relies on this matching PIL/OpenCV,
    and `tests/test_geometry.py` pins the relationship.
    """
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    dx, dy = p[0] - center[0], p[1] - center[1]
    return (center[0] + dx * cos - dy * sin,
            center[1] + dx * sin + dy * cos)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100].

    Implemented here rather than pulled from numpy so the geometry layer stays
    dependency-free and the reference-distribution code can be unit tested
    without the vision stack installed.
    """
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation (n-1). NaN for fewer than two values."""
    n = len(values)
    if n < 2:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
