"""Guard tests.

The guards are the part of this project that decides when NOT to answer, which
matters more than the readings themselves: the failure mode that damages trust
is a confident number produced from an input that cannot support one.
"""
from __future__ import annotations

import numpy as np
import pytest

from posture import guards as G
from posture import landmarks as L
from posture.metrics import NoiseModel, compute_all
from posture.view import estimate_view

from .conftest import make_pose

NOISE = NoiseModel(frac_of_body_scale=0.004, provenance="test")


def metrics_for(pose):
    return compute_all(pose, estimate_view(pose), NOISE)


def keys(findings):
    return {f.key for f in findings}


class TestFraming:
    def test_clean_pose_passes(self, side_pose):
        assert G.check_framing(side_pose) == []

    def test_landmark_outside_frame_blocks(self):
        p = make_pose(view="side")
        lm = p.landmarks[L.LEFT_ANKLE]
        p.landmarks[L.LEFT_ANKLE] = L.Landmark(lm.x, p.height + 200, lm.z, 1.0, 1.0)
        f = G.check_framing(p)
        assert "landmarks_out_of_frame" in keys(f)
        assert all(x.severity == G.SEVERITY_BLOCK for x in f)

    def test_comparable_second_person_blocks(self, side_pose):
        side_pose.n_poses_detected = 2
        side_pose.other_pose_bboxes = [(10, 10, 300, 1000)]
        assert "multiple_people" in keys(G.check_framing(side_pose))

    def test_small_bystander_is_ignored(self, side_pose):
        # A distant person in the background must not block a valid upload.
        side_pose.n_poses_detected = 2
        side_pose.other_pose_bboxes = [(10, 10, 40, 70)]
        assert "multiple_people" not in keys(G.check_framing(side_pose))


class TestProportions:
    def test_normal_body_passes(self, side_pose):
        assert G.check_proportions(side_pose) == []

    def test_squashed_body_blocks(self):
        # Approximates the palm-photo and sock-photo failures: a skeleton whose
        # segments are not in human proportion.
        p = make_pose(view="side")
        for idx in (L.LEFT_ANKLE, L.RIGHT_ANKLE):
            lm = p.landmarks[idx]
            hip_y = p.landmarks[L.LEFT_HIP].y
            p.landmarks[idx] = L.Landmark(lm.x, hip_y + 20, lm.z, 1.0, 1.0)
        assert "implausible_proportions" in keys(G.check_proportions(p))


class TestNeutrality:
    def test_neutral_stance_passes(self, side_pose):
        assert G.check_pose_neutrality(metrics_for(side_pose)) == []

    def test_flexed_knee_blocks(self):
        p = make_pose(view="side", knee_deviation_deg=25.0)
        assert "knee_flexed" in keys(G.check_pose_neutrality(metrics_for(p)))

    def test_leaning_trunk_blocks(self):
        p = make_pose(view="side", trunk_sway_deg=20.0)
        assert "trunk_leaning" in keys(G.check_pose_neutrality(metrics_for(p)))

    def test_frontal_neutrality_is_checked_too(self, front_pose):
        # Regression test: the knee and trunk guards are sagittal-only, so
        # without a frontal equivalent a front photo would be held to a weaker
        # standard than a side photo.
        assert G.check_frontal_neutrality(front_pose) == []

    def test_wide_stance_blocks_in_front_view(self):
        p = make_pose(view="front")
        hip_half = abs(p.landmarks[L.LEFT_HIP].x - p.landmarks[L.RIGHT_HIP].x) / 2
        cx = p.width / 2
        for idx, sign in ((L.LEFT_ANKLE, +1), (L.RIGHT_ANKLE, -1)):
            lm = p.landmarks[idx]
            p.landmarks[idx] = L.Landmark(cx + sign * hip_half * 2.5, lm.y,
                                          lm.z, 1.0, 1.0)
        assert "stance_too_wide" in keys(G.check_frontal_neutrality(p))

    def test_weight_on_one_leg_warns_not_blocks(self):
        p = make_pose(view="front")
        for idx in (L.LEFT_HIP, L.RIGHT_HIP):
            lm = p.landmarks[idx]
            p.landmarks[idx] = L.Landmark(lm.x + 90, lm.y, lm.z, 1.0, 1.0)
        f = G.check_frontal_neutrality(p)
        assert "weight_on_one_leg" in keys(f)
        assert all(x.severity == G.SEVERITY_WARN for x in f
                   if x.key == "weight_on_one_leg")

    def test_raised_arms_warn(self):
        p = make_pose(view="front")
        for idx in (L.LEFT_WRIST, L.RIGHT_WRIST):
            lm = p.landmarks[idx]
            p.landmarks[idx] = L.Landmark(lm.x, p.landmarks[L.LEFT_SHOULDER].y,
                                          lm.z, 1.0, 1.0)
        assert "arms_not_at_side" in keys(G.check_frontal_neutrality(p))


class TestViewSuitability:
    def test_side_view_passes(self, side_pose):
        v = estimate_view(side_pose)
        assert "oblique_view" not in keys(G.check_view_suitability(v, {}))

    def test_oblique_view_blocks(self):
        # Oblique is blocked rather than warned because the distortion is
        # invisible in the reading itself.
        p = make_pose(view="side")
        cx = p.width / 2
        for idx, sign in ((L.LEFT_SHOULDER, +1), (L.RIGHT_SHOULDER, -1)):
            lm = p.landmarks[idx]
            p.landmarks[idx] = L.Landmark(cx + sign * 60, lm.y, lm.z, 1.0, 1.0)
        v = estimate_view(p)
        assert v.view == "oblique"
        assert "oblique_view" in keys(G.check_view_suitability(v, {}))


class TestOutputSanity:
    def test_normal_readings_pass(self, side_pose):
        assert G.check_output_sanity(metrics_for(side_pose)) == []

    def test_impossible_angle_blocks(self):
        from posture.metrics import Measurement
        m = {"forward_head": Measurement(
            key="forward_head", label="x", value=87.0, uncertainty=1.0,
            plane="sagittal", span_px=100.0)}
        assert "implausible_reading" in keys(G.check_output_sanity(m))


class TestVisibilityIsNotTrusted:
    def test_guards_do_not_consult_visibility(self):
        # Confirmed failure from the previous phase: a landmark placed on hair
        # reported visibility 1.000. Any guard that trusted that channel would
        # have passed it. This pins that no guard reads it.
        import inspect
        src = inspect.getsource(G)
        body = "\n".join(line for line in src.splitlines()
                         if not line.strip().startswith("#")
                         and not line.strip().startswith("*"))
        assert ".visibility" not in body
        assert ".presence" not in body


class TestBlockedHelper:
    def test_blocked_true_when_any_block(self):
        f = [G.GuardFinding("a", G.SEVERITY_WARN, "w"),
             G.GuardFinding("b", G.SEVERITY_BLOCK, "b")]
        assert G.blocked(f) is True

    def test_blocked_false_for_warnings_only(self):
        assert G.blocked([G.GuardFinding("a", G.SEVERITY_WARN, "w")]) is False

    def test_blocked_false_for_empty(self):
        assert G.blocked([]) is False
