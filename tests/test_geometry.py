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
    # hip pushed 150px forward of the shoulder-ankle column at mid height.
    # Kept modest on purpose: a larger offset trips the sanity bound in
    # thresholds.MAX_PLAUSIBLE_SAGITTAL_DEV_DEG, which is a separate test.
    put(p, "left_hip", 650, 900); put(p, "right_hip", 650, 900)
    m = M.sagittal_alignment(p, "side")
    v1 = (500 - 650, 500 - 900)
    v2 = (500 - 650, 1500 - 900)
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


def test_triage_targets_prioritise_side_views():
    """The collection brief and the triage script must not drift apart:
    side views are the blocking gap, so they carry the largest target."""
    import importlib.util, pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "triage_images.py"
    spec = importlib.util.spec_from_file_location("triage", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod.TARGETS) == {"side", "front", "negative"}
    assert mod.TARGETS["side"] > mod.TARGETS["front"] > mod.TARGETS["negative"]
    assert mod.MIN_BODY_FRACTION > 0.5 and mod.MIN_SHORT_SIDE >= 480


def _world(yaw_deg, r=0.2):
    """Synthetic world landmarks with a known biacromial yaw."""
    from posture.landmarks import Pt3
    import math as _m
    w = [Pt3(0.0, 0.0, 0.0, 1.0) for _ in range(33)]
    a = _m.radians(yaw_deg)
    w[IDX["left_shoulder"]] = Pt3(r * _m.cos(a), 0.0, r * _m.sin(a), 1.0)
    w[IDX["right_shoulder"]] = Pt3(-r * _m.cos(a), 0.0, -r * _m.sin(a), 1.0)
    return w


def test_body_yaw_recovers_known_rotation():
    for want in (0.0, 15.0, 45.0, 90.0):
        got = M.body_yaw(_world(want))
        assert abs(got - want) < 1e-6, (want, got)
    # a mirrored rotation reads the same magnitude
    assert abs(M.body_yaw(_world(-30.0)) - 30.0) < 1e-6


def test_yaw_gate_beats_the_ratio_on_a_rotated_subject():
    """Regression for a real miss: two photos rotated ~55 deg scored ratio
    0.453 and 0.401, just over the old frontal cut-off, so the ratio test
    called them frontal and computed a shoulder-height reading on them."""
    p = blank(vis=0.99)
    put(p, "left_shoulder", 400, 500); put(p, "right_shoulder", 690, 500)
    put(p, "left_hip", 430, 1150);     put(p, "right_hip", 660, 1150)
    ratio = M.shoulder_torso_ratio(p)
    assert 0.40 <= ratio <= 0.47, ratio
    assert M.detect_view(p)[0] == "front"                 # ratio fallback: wrong
    assert M.detect_view(p, _world(55.0))[0] == "oblique"  # yaw gate: caught
    # and the gate still passes a genuine frontal and a genuine lateral
    assert M.detect_view(p, _world(12.0))[0] == "front"
    assert M.detect_view(p, _world(86.0))[0] == "side"


def test_yaw_falls_back_to_ratio_when_world_is_missing():
    p = side_subject()
    assert M.body_yaw(None) is None
    assert M.body_yaw([]) is None
    assert M.detect_view(p, None)[0] == "side"


def test_forward_head_reports_its_own_uncertainty():
    m = M.forward_head(side_subject(ear_dx=60.0), "side")
    assert m.uncertainty is not None and m.uncertainty > 0
    # the swing must be on the order of degrees, not negligible or absurd
    assert 0.5 < m.uncertainty < 30.0
    # and it must not recurse
    bare = M.forward_head(side_subject(ear_dx=60.0), "side",
                          _estimate_uncertainty=False)
    assert bare.uncertainty is None
    assert abs(bare.value - m.value) < 1e-9


def test_facing_sign_is_stable_when_nose_sits_over_the_ear():
    """Regression: with the nose almost directly above the ear, a few pixels
    of noise flipped the facing sign and every A/P metric flipped with it."""
    p = blank(vis=0.99)
    put(p, "left_shoulder", 560, 500); put(p, "right_shoulder", 440, 500)
    put(p, "left_hip", 550, 900);      put(p, "right_hip", 450, 900)
    put(p, "left_ear", 520, 320);      put(p, "right_ear", 480, 320)
    signs = set()
    for jitter in (-3, -1, 0, 1, 3):
        q = list(p)
        q[IDX["nose"]] = Pt(500.0 + jitter, 340.0, 0.99)
        signs.add(M.facing_sign(q))
    assert len(signs) == 1, f"facing sign flipped under +/-3px of noise: {signs}"


def test_extrapolated_ankle_is_refused():
    """MediaPipe returns an ankle for a photo cropped at the shins - hundreds
    of pixels below the last row that exists. That point is a guess."""
    p = side_subject()
    put(p, "left_ankle", 500, 1500); put(p, "right_ankle", 502, 1500)
    size = (1000, 1000)          # ankle y=1500 lies outside a 1000px-tall image
    assert M.sagittal_alignment(p, "side", size).value is None
    assert M.plumb_offsets(p, "side", size).value is None
    # with the real height it computes normally
    assert M.sagittal_alignment(p, "side", (1000, 1600)).value is not None
    # and with no size supplied the check cannot run, so it does not block
    assert M.sagittal_alignment(p, "side", None).value is not None


def test_in_frame_bounds():
    assert M._in_frame(Pt(0.0, 0.0, 1.0), (100, 200))
    assert M._in_frame(Pt(100.0, 200.0, 1.0), (100, 200))
    assert not M._in_frame(Pt(-1.0, 50.0, 1.0), (100, 200))
    assert not M._in_frame(Pt(50.0, 201.0, 1.0), (100, 200))
    assert M._in_frame(Pt(-999.0, 999.0, 1.0), None)


def test_oblique_view_withholds_every_metric():
    """Regression: a subject rotated ~53 deg was classed oblique and still got
    "forward head +32.1 deg, pronounced" - a confident pathological reading on
    a view where perspective contaminates both the sagittal and the frontal
    geometry."""
    p = side_subject(ear_dx=80.0)
    ms = M.compute_for_view(p, "oblique")
    assert all(m.value is None for m in ms), \
        [(m.key, m.value) for m in ms if m.value is not None]
    assert all("斜侧" in m.detail for m in ms)
    # the same landmarks still produce readings when the view is unambiguous
    assert any(m.value is not None for m in M.compute_for_view(p, "side"))


def test_yaw_in_the_ambiguous_band_maps_to_oblique():
    p = side_subject()
    for yaw in (31.0, 45.0, 69.0):
        assert M.detect_view(p, _world(yaw))[0] == "oblique"
    assert M.detect_view(p, _world(29.0))[0] == "front"
    assert M.detect_view(p, _world(71.0))[0] == "side"


def test_knee_flexion_rejects_a_seated_subject():
    """Measured round 2: standing knees 175.7-179.4 deg, sitting 35.2 and 81.2,
    kneeling 19.0 - so this gate sits in a ~90 deg empty gap."""
    p = side_subject()
    put(p, "left_knee", 700, 1100); put(p, "right_knee", 700, 1100)
    assert M.knee_extension(p) < 150.0
    assert not M.check_plausibility(p).ok
    # a straight leg passes
    q = side_subject()
    put(q, "left_knee", 500, 1200); put(q, "right_knee", 502, 1200)
    assert M.knee_extension(q) > 170.0
    assert M.check_plausibility(q).ok


def test_multiple_people_are_refused():
    p = side_subject()
    put(p, "left_knee", 500, 1200); put(p, "right_knee", 502, 1200)
    assert M.check_plausibility(p, None, 1).ok
    c = M.check_plausibility(p, None, 2)
    assert not c.ok and any("2 个人" in r for r in c.reasons)


def test_span_taller_than_the_image_is_refused():
    """A half-body crop scored an ear-to-ankle span of 1.02x the image height;
    real full-body photos in the set run 0.61-0.74."""
    p = side_subject()
    put(p, "left_knee", 500, 1200); put(p, "right_knee", 502, 1200)
    # ear y=320, ankle y=1500 -> span 1180px
    assert M.check_plausibility(p, (1000, 1600)).ok       # span 0.74
    assert not M.check_plausibility(p, (1000, 1100)).ok   # span 1.07


def test_shins_cropped_photo_keeps_its_head_metrics():
    """A photo cut off at the shins still has a real ear and shoulder, so the
    head/shoulder metrics stand; only the ankle-dependent ones are withheld."""
    p = side_subject(ear_dx=60.0)
    put(p, "left_knee", 500, 1200); put(p, "right_knee", 502, 1200)
    size = (1000, 1450)          # ankle y=1500 is just outside
    assert M.check_plausibility(p, size).ok
    ms = {m.key: m for m in M.compute_for_view(p, "side", size)}
    assert ms["forward_head"].value is not None
    assert ms["rounded_shoulder"].value is not None
    assert ms["sagittal_alignment"].value is None
    assert ms["plumb"].value is None


def test_absurd_sagittal_deviation_is_refused():
    """A statue in contrapposto produced a 129 deg shoulder-hip-ankle
    deviation, which no standing human can reach."""
    p = side_subject()
    put(p, "left_hip", 1400, 700); put(p, "right_hip", 1400, 700)
    m = M.sagittal_alignment(p, "side")
    if m.value is not None:
        assert abs(m.value) <= 60.0
    put(p, "left_ankle", 480, 760); put(p, "right_ankle", 480, 760)
    assert M.sagittal_alignment(p, "side").value is None


def test_knee_check_ignores_an_extrapolated_knee():
    """Regression: a side-view photo cropped at mid-calf yielded a knee angle
    of 135 deg from an extrapolated knee, and the standing check rejected a
    subject who was plainly standing."""
    p = side_subject()
    put(p, "left_knee", 640, 1250); put(p, "right_knee", 640, 1250)   # bogus
    assert M.knee_extension(p, None) < 150.0            # no size: cannot tell
    assert M.knee_extension(p, (1000, 1100)) is None    # knee/ankle out of frame
    assert M.check_plausibility(p, (1000, 1100)).knee_deg is None
