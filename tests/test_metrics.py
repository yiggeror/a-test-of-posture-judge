"""Round-trip tests: a synthetic pose built with a known angle must measure it.

This is the one accuracy check available anywhere in the project. Real
photographs cannot support one, because nobody knows the true angle of a
person in a photograph. A synthetic skeleton does.
"""
from __future__ import annotations

import math

import pytest

from posture import landmarks as L
from posture.metrics import NoiseModel, compute_frontal, compute_sagittal
from posture.view import estimate_view

from .conftest import make_pose

NOISE = NoiseModel(frac_of_body_scale=0.004, provenance="test")


def sag(**kw):
    p = make_pose(view="side", **kw)
    return compute_sagittal(p, estimate_view(p), NOISE)


def front(**kw):
    p = make_pose(view="front", **kw)
    return compute_frontal(p, NOISE)


class TestSagittalRoundTrip:
    @pytest.mark.parametrize("angle", [0.0, 3.0, 7.5, 15.0, 25.0])
    def test_forward_head_recovered(self, angle):
        m = sag(forward_head_deg=angle)
        assert m["forward_head"].value == pytest.approx(angle, abs=0.05)

    @pytest.mark.parametrize("angle", [0.0, 4.0, 12.0, 20.0])
    def test_shoulder_protraction_recovered(self, angle):
        m = sag(shoulder_protraction_deg=angle)
        assert m["shoulder_protraction"].value == pytest.approx(angle, abs=0.05)

    @pytest.mark.parametrize("angle", [-8.0, -3.0, 0.0, 5.0, 9.0])
    def test_trunk_sway_recovered(self, angle):
        m = sag(trunk_sway_deg=angle)
        assert m["trunk_sway"].value == pytest.approx(angle, abs=0.05)

    def test_angles_compose_along_the_chain(self):
        # Each joint angle is measured relative to its own lower joint, so a
        # figure built with trunk sway AND forward head must report each
        # independently rather than one absorbing the other.
        m = sag(trunk_sway_deg=5.0, shoulder_protraction_deg=8.0,
                forward_head_deg=12.0)
        assert m["trunk_sway"].value == pytest.approx(5.0, abs=0.05)
        assert m["shoulder_protraction"].value == pytest.approx(8.0, abs=0.05)
        assert m["forward_head"].value == pytest.approx(12.0, abs=0.05)

    def test_facing_left_gives_same_reading(self):
        # The same anatomy photographed from the other side must read the same.
        r = make_pose(view="side", forward_head_deg=14.0, facing=+1)
        l = make_pose(view="side", forward_head_deg=14.0, facing=-1)
        vr, vl = estimate_view(r), estimate_view(l)
        assert vr.facing == +1 and vl.facing == -1
        a = compute_sagittal(r, vr, NOISE)["forward_head"].value
        b = compute_sagittal(l, vl, NOISE)["forward_head"].value
        assert a == pytest.approx(b, abs=0.05)

    def test_knee_deviation_sign_and_magnitude(self):
        flexed = sag(knee_deviation_deg=10.0)["knee_deviation"]
        assert flexed.value == pytest.approx(10.0, abs=0.6)
        hyper = sag(knee_deviation_deg=-8.0)["knee_deviation"]
        assert hyper.value == pytest.approx(-8.0, abs=0.6)

    def test_straight_leg_is_zero(self):
        assert sag()["knee_deviation"].value == pytest.approx(0.0, abs=0.05)


class TestMetricIndependence:
    def test_head_over_hip_ignores_the_shoulder(self):
        # Moving ONLY the shoulder must not change head_over_hip. This is the
        # property that makes it useful for separating a real head position
        # from a mislocalised shoulder.
        base = make_pose(view="side", forward_head_deg=10.0)
        v = estimate_view(base)
        before = compute_sagittal(base, v, NOISE)["head_over_hip"].value

        moved = make_pose(view="side", forward_head_deg=10.0)
        for idx in (L.LEFT_SHOULDER, L.RIGHT_SHOULDER):
            lm = moved.landmarks[idx]
            moved.landmarks[idx] = L.Landmark(lm.x + 40, lm.y, lm.z,
                                              lm.visibility, lm.presence)
        after = compute_sagittal(moved, estimate_view(moved), NOISE)["head_over_hip"].value
        assert before == pytest.approx(after, abs=0.05)

    def test_shoulder_error_pushes_the_two_metrics_opposite_ways(self):
        # A shoulder placed too far anterior increases shoulder_protraction and
        # decreases forward_head. Their coupling through shared measurement
        # error is why they must not be read as independent findings.
        base = make_pose(view="side", forward_head_deg=12.0,
                         shoulder_protraction_deg=6.0)
        b = compute_sagittal(base, estimate_view(base), NOISE)

        moved = make_pose(view="side", forward_head_deg=12.0,
                          shoulder_protraction_deg=6.0)
        for idx in (L.LEFT_SHOULDER, L.RIGHT_SHOULDER):
            lm = moved.landmarks[idx]
            moved.landmarks[idx] = L.Landmark(lm.x + 25, lm.y, lm.z,
                                              lm.visibility, lm.presence)
        a = compute_sagittal(moved, estimate_view(moved), NOISE)

        assert a["shoulder_protraction"].value > b["shoulder_protraction"].value
        assert a["forward_head"].value < b["forward_head"].value


class TestUncertainty:
    def test_short_span_is_noisier_than_long_span(self):
        # The structural point: ear-shoulder is a short segment, hip-ankle is a
        # long one, so the same landmark noise hurts forward_head far more.
        m = sag()
        assert m["forward_head"].uncertainty > m["trunk_sway"].uncertainty
        ratio = m["forward_head"].uncertainty / m["trunk_sway"].uncertainty
        assert ratio > 2.5

    def test_uncertainty_scales_with_noise(self):
        p = make_pose(view="side")
        v = estimate_view(p)
        lo = compute_sagittal(p, v, NoiseModel(0.002, "test"))["forward_head"]
        hi = compute_sagittal(p, v, NoiseModel(0.008, "test"))["forward_head"]
        assert hi.uncertainty == pytest.approx(4 * lo.uncertainty, rel=1e-6)

    def test_uncertainty_is_scale_invariant(self):
        # Photographing the same person from twice as far must not change the
        # uncertainty, because the noise model is expressed relative to body
        # size rather than in absolute pixels.
        a = make_pose(view="side", width=800, height=1100)
        b = make_pose(view="side", width=1600, height=2200)
        ua = compute_sagittal(a, estimate_view(a), NOISE)["forward_head"].uncertainty
        ub = compute_sagittal(b, estimate_view(b), NOISE)["forward_head"].uncertainty
        assert ua == pytest.approx(ub, rel=1e-6)


class TestFrontal:
    @pytest.mark.parametrize("angle", [0.0, 2.0, 5.0, -4.0])
    def test_shoulder_tilt_recovered(self, angle):
        assert front(shoulder_tilt_deg=angle)["shoulder_tilt"].value == \
            pytest.approx(angle, abs=0.05)

    @pytest.mark.parametrize("angle", [0.0, 3.0, -6.0])
    def test_pelvis_tilt_recovered(self, angle):
        assert front(pelvis_tilt_deg=angle)["pelvis_tilt"].value == \
            pytest.approx(angle, abs=0.05)

    def test_head_vs_shoulder_tilt_cancels_common_rotation(self):
        # Tilting head and shoulders by the same amount is what a rolled camera
        # does, and the roll-invariant metric must read zero for it.
        m = front(shoulder_tilt_deg=5.0, head_tilt_deg=5.0)
        assert m["head_vs_shoulder_tilt"].value == pytest.approx(0.0, abs=0.1)

    def test_head_vs_shoulder_tilt_keeps_real_difference(self):
        m = front(shoulder_tilt_deg=2.0, head_tilt_deg=8.0)
        assert m["head_vs_shoulder_tilt"].value == pytest.approx(6.0, abs=0.1)

    def test_pelvic_sagittal_tilt_is_not_offered(self):
        # The metric set must not contain anything claiming to be
        # anterior/posterior pelvic tilt, which these landmarks cannot measure.
        keys = set(front()) | set(sag())
        assert not any("pelvic_sagittal" in k or k == "pelvic_tilt" for k in keys)
        assert "pelvis_tilt" in front()  # lateral obliquity only, clearly named


class TestNoiseModel:
    def test_missing_file_falls_back_and_says_so(self):
        n = NoiseModel.load("/nonexistent/path/noise.json")
        assert n.provenance == "guess"
        assert n.frac_of_body_scale > 0

    def test_sigma_scales_with_body(self):
        n = NoiseModel(0.005, "test")
        assert n.sigma_px(1000.0) == pytest.approx(5.0)
