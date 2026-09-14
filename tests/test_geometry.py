"""Geometry unit tests.

These use SYNTHETIC coordinates with analytically known answers. They verify
the arithmetic only. They say nothing about whether the model detects poses
well, and nothing about whether the thresholds are clinically meaningful.
"""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture.landmarks import Pt, NAMES, IDX
from posture import metrics as M


def blank(n=33, vis=1.0):
    return [Pt(0.0, 0.0, vis) for _ in range(n)]


def put(pts, name, x, y, vis=1.0):
    pts[IDX[name]] = Pt(float(x), float(y), vis)


def side_subject(ear_dx=0.0, rise=180.0):
    """Subject facing image-right, seen from the left side.

    Acromion at (500, 500); ear `ear_dx` px in front (+x) and `rise` px above.
    """
    p = blank(vis=0.99)
    put(p, "left_shoulder", 500, 500)
    put(p, "right_shoulder", 505, 500)      # collapsed -> side view
    put(p, "left_hip", 500, 900)
    put(p, "right_hip", 503, 900)
    put(p, "left_ankle", 500, 1500)
    put(p, "right_ankle", 502, 1500)
    put(p, "left_ear", 500 + ear_dx, 500 - rise)
    put(p, "right_ear", 500 + ear_dx, 500 - rise)
    put(p, "nose", 500 + ear_dx + 60, 500 - rise)   # nose anterior -> faces +x
    return p


def test_view_detection():
    assert M.detect_view(side_subject())[0] == "side"
    p = blank(vis=0.99)
    put(p, "left_shoulder", 400, 500); put(p, "right_shoulder", 700, 500)
    put(p, "left_hip", 430, 950);      put(p, "right_hip", 670, 950)
    assert M.detect_view(p)[0] == "front"


def test_facing_sign():
    assert M.facing_sign(side_subject()) == 1.0
    p = side_subject()
    p[IDX["nose"]] = Pt(p[IDX["left_ear"]].x - 60, 320, 0.99)
    assert M.facing_sign(p) == -1.0


def test_forward_head_zero_when_ear_above_shoulder():
    m = M.forward_head(side_subject(ear_dx=0.0), "side")
    assert m.available and abs(m.value) < 1e-9
    assert m.band == "typical"


def test_forward_head_known_angle():
    # ear 100px forward over a 100px rise -> exactly 45 degrees
    m = M.forward_head(side_subject(ear_dx=100.0, rise=100.0), "side")
    assert abs(m.value - 45.0) < 1e-6
    assert m.band == "pronounced"
    # 30 degrees: dx = rise * tan(30)
    rise = 180.0
    m = M.forward_head(side_subject(ear_dx=rise * math.tan(math.radians(30)),
                                    rise=rise), "side")
    assert abs(m.value - 30.0) < 1e-6


def test_forward_head_sign_flips_with_facing():
    p = side_subject(ear_dx=80.0)
    p[IDX["nose"]] = Pt(p[IDX["left_ear"]].x - 60, 320, 0.99)  # now faces -x
    m = M.forward_head(p, "side")
    assert m.value < 0     # ear is now BEHIND the acromion


def test_forward_head_unavailable_on_front_view():
    p = blank(vis=0.99)
    put(p, "left_shoulder", 400, 500); put(p, "right_shoulder", 700, 500)
    put(p, "left_hip", 430, 950);      put(p, "right_hip", 670, 950)
    m = M.forward_head(p, "front")
    assert not m.available and m.band == "unavailable"


def test_forward_head_needs_visible_landmarks():
    p = side_subject(ear_dx=80.0)
    p[IDX["left_ear"]] = Pt(580, 320, 0.1)
    p[IDX["right_ear"]] = Pt(580, 320, 0.1)
    assert not M.forward_head(p, "side").available


def test_rounded_shoulder_sign_and_scale():
    # acromion 40px BEHIND the ear -> negative; torso is 400px
    m = M.rounded_shoulder(side_subject(ear_dx=40.0), "side")
    assert abs(m.value - (-40.0 / 400.0)) < 1e-9
    # acromion in front of the ear -> positive
    m2 = M.rounded_shoulder(side_subject(ear_dx=-60.0), "side")
    assert m2.value > 0


def test_rounded_shoulder_is_scale_invariant():
    a = M.rounded_shoulder(side_subject(ear_dx=40.0), "side").value
    p = side_subject(ear_dx=40.0)
    scaled = [Pt(q.x * 3, q.y * 3, q.visibility) for q in p]
    b = M.rounded_shoulder(scaled, "side").value
    assert abs(a - b) < 1e-9


def test_sagittal_alignment_straight_is_zero():
    m = M.sagittal_alignment(side_subject(), "side")
    assert m.available and abs(m.value) < 1e-6
    assert m.band == "typical"


def test_sagittal_alignment_known_deviation():
    p = side_subject()
    # hip pushed 400px forward of the shoulder-ankle column at mid height
    put(p, "left_hip", 900, 900); put(p, "right_hip", 900, 900)
    m = M.sagittal_alignment(p, "side")
    v1 = (500 - 900, 500 - 900)
    v2 = (500 - 900, 1500 - 900)
    cos = (v1[0]*v2[0] + v1[1]*v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    expected = 180.0 - math.degrees(math.acos(cos))
    assert abs(abs(m.value) - expected) < 1e-6


def test_shoulder_tilt_known_angle():
    p = blank(vis=0.99)
    put(p, "left_shoulder", 400, 600); put(p, "right_shoulder", 700, 600)
    put(p, "left_hip", 430, 1000);     put(p, "right_hip", 670, 1000)
    m = M.shoulder_tilt(p, "front")
    assert abs(m.value) < 1e-9 and m.band == "typical"
    # left shoulder 300px lower across a 300px span -> 45 deg, positive
    put(p, "left_shoulder", 400, 900)
    m = M.shoulder_tilt(p, "front")
    assert abs(m.value - 45.0) < 1e-6
    # mirror -> negative
    put(p, "left_shoulder", 400, 600); put(p, "right_shoulder", 700, 900)
    assert abs(M.shoulder_tilt(p, "front").value + 45.0) < 1e-6


def test_shoulder_tilt_unavailable_on_side_view():
    assert not M.shoulder_tilt(side_subject(), "side").available


def test_aspect_ratio_is_not_baked_into_angles():
    """The regression this guards: computing angles on normalised coords.

    Stretching only the x axis MUST change the angle (it is a real shape
    change). Scaling both axes equally must NOT. If someone reintroduces
    normalised-coordinate maths, the uniform case starts drifting.
    """
    base = M.forward_head(side_subject(ear_dx=100.0, rise=100.0), "side").value
    p = side_subject(ear_dx=100.0, rise=100.0)
    uniform = [Pt(q.x * 2.5, q.y * 2.5, q.visibility) for q in p]
    assert abs(M.forward_head(uniform, "side").value - base) < 1e-9
    stretched = [Pt(q.x * 2.0, q.y, q.visibility) for q in p]
    assert abs(M.forward_head(stretched, "side").value - base) > 5.0


def test_plumb_offsets_share_one_reference():
    p = side_subject(ear_dx=120.0)
    m = M.plumb_offsets(p, "side")
    assert m.available
    assert "耳" in m.detail and "肩峰" in m.detail


def test_band_boundaries():
    from posture.thresholds import FORWARD_HEAD as T
    assert M._band(T.mild - 0.01, T) == "typical"
    assert M._band(T.mild, T) == "mild"
    assert M._band(T.marked - 0.01, T) == "mild"
    assert M._band(T.marked, T) == "pronounced"
    assert M._band(-T.marked, T) == "pronounced"   # banding uses magnitude


def test_landmark_table_is_the_documented_33():
    assert len(NAMES) == 33 and len(set(NAMES)) == 33
    assert NAMES[0] == "nose" and NAMES[32] == "right_foot_index"


def test_detect_does_not_deadlock_on_the_landmarker_lock():
    """Regression: detect() used to hold the same non-reentrant lock that
    get_landmarker() then tried to acquire, hanging the first request."""
    import threading
    from posture import engine
    assert engine._create_lock is not engine._infer_lock
    assert engine._create_lock.acquire(timeout=1)
    try:
        done = threading.Event()
        threading.Thread(
            target=lambda: (engine._infer_lock.acquire(timeout=2), done.set()),
            daemon=True).start()
        assert done.wait(3), "inference lock is entangled with the creation lock"
    finally:
        engine._create_lock.release()


def test_plausibility_rejects_a_non_vertical_torso():
    """Regression: a photo of a hand produced a confident skeleton whose torso
    axis lay 61 deg off vertical, and every metric reported a number anyway."""
    p = blank(vis=0.99)
    put(p, "left_shoulder", 534, 767); put(p, "right_shoulder", 622, 900)
    put(p, "left_hip", 324, 919);      put(p, "right_hip", 347, 1014)
    put(p, "left_ear", 539, 750);      put(p, "right_ear", 585, 836)
    put(p, "nose", 560, 800)
    c = M.check_plausibility(p)
    assert not c.ok
    assert c.torso_tilt_deg > 40.0
    _, _, ms, plaus = M.compute_all(p)
    assert not plaus.ok
    assert all(m.value is None and m.band == "unavailable" for m in ms)


def test_plausibility_accepts_an_upright_subject():
    p = blank(vis=0.99)
    put(p, "left_shoulder", 407, 271); put(p, "right_shoulder", 255, 273)
    put(p, "left_hip", 360, 509);      put(p, "right_hip", 274, 506)
    put(p, "left_ear", 356, 154);      put(p, "right_ear", 300, 154)
    put(p, "nose", 328, 170)
    c = M.check_plausibility(p)
    assert c.ok and c.reasons == [] and c.torso_tilt_deg < 10.0


def test_plausibility_rejects_bad_head_torso_proportion():
    p = blank(vis=0.99)
    put(p, "left_shoulder", 400, 500); put(p, "right_shoulder", 600, 500)
    put(p, "left_hip", 420, 900);      put(p, "right_hip", 580, 900)
    put(p, "left_ear", 495, 480);      put(p, "right_ear", 505, 480)  # far too close
    put(p, "nose", 500, 470)
    assert not M.check_plausibility(p).ok
