"""View classification and facing direction."""
from __future__ import annotations

import pytest

from posture import landmarks as L
from posture.view import (FACING_LEFT, FACING_RIGHT, VIEW_FRONT, VIEW_OBLIQUE,
                          VIEW_SIDE, estimate_view, near_side_indices)

from .conftest import make_pose


class TestViewClassification:
    def test_side_view_detected(self, side_pose):
        assert estimate_view(side_pose).view == VIEW_SIDE

    def test_front_view_detected(self, front_pose):
        assert estimate_view(front_pose).view == VIEW_FRONT

    def test_oblique_between_the_two(self):
        p = make_pose(view="side")
        cx = p.width / 2
        for idx, sign in ((L.LEFT_SHOULDER, +1), (L.RIGHT_SHOULDER, -1)):
            lm = p.landmarks[idx]
            p.landmarks[idx] = L.Landmark(cx + sign * 62, lm.y, lm.z, 1.0, 1.0)
        assert estimate_view(p).view == VIEW_OBLIQUE

    def test_yaw_near_zero_for_side(self, side_pose):
        assert estimate_view(side_pose).yaw_deg < 20

    def test_yaw_near_90_for_front(self, front_pose):
        assert estimate_view(front_pose).yaw_deg > 70

    def test_view_is_scale_invariant(self):
        small = make_pose(view="side", width=400, height=550)
        large = make_pose(view="side", width=1600, height=2200)
        assert estimate_view(small).view == estimate_view(large).view
        assert estimate_view(small).shoulder_spread == pytest.approx(
            estimate_view(large).shoulder_spread, rel=1e-6)


class TestFacing:
    def test_facing_right(self):
        assert estimate_view(make_pose(view="side", facing=+1)).facing == FACING_RIGHT

    def test_facing_left(self):
        assert estimate_view(make_pose(view="side", facing=-1)).facing == FACING_LEFT

    def test_side_view_facing_is_confident(self, side_pose):
        assert estimate_view(side_pose).facing_confidence > 0.5

    def test_front_view_facing_is_not_confident(self, front_pose):
        # Facing direction is genuinely ambiguous head-on, so low confidence is
        # the correct answer rather than a failure.
        assert estimate_view(front_pose).facing_confidence < 0.5


class TestNearSide:
    def test_returns_a_consistent_set(self, side_pose):
        v = estimate_view(side_pose)
        side = near_side_indices(side_pose, v.facing)
        left = {L.LEFT_EAR, L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE}
        right = {L.RIGHT_EAR, L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE,
                 L.RIGHT_ANKLE}
        chosen = {side["ear"], side["shoulder"], side["hip"], side["knee"],
                  side["ankle"]}
        # All five must come from the same side of the body; a mixed set would
        # measure an angle between two different sides of the person.
        assert chosen <= left or chosen <= right

    def test_all_expected_joints_present(self, side_pose):
        v = estimate_view(side_pose)
        side = near_side_indices(side_pose, v.facing)
        assert set(side) == {"ear", "shoulder", "hip", "knee", "ankle", "heel",
                             "foot"}
