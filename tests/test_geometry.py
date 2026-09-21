"""Exact tests for the geometry layer.

These use synthetic points, so they test the maths with no detector involved
and no tolerance for "close enough". Every sign convention the rest of the
codebase depends on is pinned here, because a sign error in image coordinates
(+y down) produces plausible-looking numbers with the wrong meaning.
"""
from __future__ import annotations

import math

import pytest

from posture.geometry import (angle_from_horizontal, angle_from_vertical,
                              distance, interior_angle, mean, midpoint,
                              percentile, rotate_point, stdev)


class TestAngleFromVertical:
    def test_straight_up_is_zero(self):
        assert angle_from_vertical((0, 100), (0, 0)) == pytest.approx(0.0)

    def test_upper_point_forward_is_positive_when_facing_right(self):
        # Facing +x, upper point displaced to +x -> anterior -> positive.
        a = angle_from_vertical((0, 100), (10, 0), anterior=+1)
        assert a > 0

    def test_upper_point_forward_is_positive_when_facing_left(self):
        # Same anatomy mirrored: facing -x, upper point displaced to -x.
        a = angle_from_vertical((0, 100), (-10, 0), anterior=-1)
        assert a > 0

    def test_facing_flips_sign(self):
        right = angle_from_vertical((0, 100), (10, 0), anterior=+1)
        left = angle_from_vertical((0, 100), (10, 0), anterior=-1)
        assert right == pytest.approx(-left)

    def test_45_degrees(self):
        assert angle_from_vertical((0, 100), (100, 0)) == pytest.approx(45.0)

    def test_known_triangle(self):
        # 3-4-5: horizontal 30, vertical 40 -> atan(30/40) = 36.87 deg
        assert angle_from_vertical((0, 40), (30, 0)) == pytest.approx(36.8699, abs=1e-3)

    def test_scale_invariant(self):
        small = angle_from_vertical((0, 10), (3, 0))
        large = angle_from_vertical((0, 1000), (300, 0))
        assert small == pytest.approx(large)

    def test_translation_invariant(self):
        a = angle_from_vertical((0, 100), (10, 0))
        b = angle_from_vertical((500, 700), (510, 600))
        assert a == pytest.approx(b)


class TestAngleFromHorizontal:
    def test_level_is_zero(self):
        assert angle_from_horizontal((0, 50), (100, 50)) == pytest.approx(0.0)

    def test_right_point_higher_is_positive(self):
        # +y is down, so a smaller y means higher in the picture.
        assert angle_from_horizontal((0, 50), (100, 40)) > 0

    def test_right_point_lower_is_negative(self):
        assert angle_from_horizontal((0, 50), (100, 60)) < 0

    def test_argument_order_does_not_flip_sign(self):
        # Swapping the two endpoints describes the same physical tilt, so the
        # reported angle must not change sign.
        a = angle_from_horizontal((0, 50), (100, 40))
        b = angle_from_horizontal((100, 40), (0, 50))
        assert a == pytest.approx(b)

    def test_45_degrees(self):
        assert angle_from_horizontal((0, 100), (100, 0)) == pytest.approx(45.0)

    def test_camera_roll_adds_directly(self):
        # Rolling the camera by r degrees rotates every point, and a level
        # segment then reads as exactly r. This is why shoulder_tilt cannot be
        # separated from camera roll.
        left, right = (0.0, 50.0), (100.0, 50.0)
        for roll in (-7.0, -1.5, 3.0, 11.0):
            c = midpoint(left, right)
            lr = rotate_point(left, c, roll)
            rr = rotate_point(right, c, roll)
            assert angle_from_horizontal(lr, rr) == pytest.approx(-roll, abs=1e-6)


class TestInteriorAngle:
    def test_straight_line_is_180(self):
        assert interior_angle((0, 0), (0, 50), (0, 100)) == pytest.approx(180.0)

    def test_right_angle(self):
        assert interior_angle((0, 0), (0, 50), (50, 50)) == pytest.approx(90.0)

    def test_degenerate_returns_nan(self):
        assert math.isnan(interior_angle((0, 0), (0, 0), (1, 1)))

    def test_symmetric_in_endpoints(self):
        a = interior_angle((10, 0), (0, 50), (40, 90))
        b = interior_angle((40, 90), (0, 50), (10, 0))
        assert a == pytest.approx(b)


class TestRotatePoint:
    def test_rotating_about_self_is_identity(self):
        p = (13.0, 47.0)
        assert rotate_point(p, p, 37.0) == pytest.approx(p)

    def test_full_turn_is_identity(self):
        p, c = (10.0, 20.0), (5.0, 5.0)
        assert rotate_point(p, c, 360.0) == pytest.approx(p)

    def test_preserves_distance(self):
        p, c = (10.0, 20.0), (5.0, 5.0)
        assert distance(rotate_point(p, c, 53.0), c) == pytest.approx(distance(p, c))

    def test_rotation_adds_to_vertical_angle(self):
        # The invariance harness depends on this exactly: rotating the points by
        # theta changes every angle-from-vertical by exactly theta. That is what
        # makes a rotation sweep a ground-truth test -- the true delta is known
        # without anyone labelling the photograph.
        #
        # The sign relationship between this function and PIL's Image.rotate is
        # NOT assumed anywhere; scripts/repeatability.py measures it and
        # asserts it, so a convention mismatch shows up as a failed self-check
        # rather than as a silently mirrored result.
        lower, upper = (100.0, 300.0), (120.0, 100.0)
        c = (150.0, 200.0)
        base = angle_from_vertical(lower, upper)
        for theta in (-12.0, -3.0, 5.0, 9.5):
            lr = rotate_point(lower, c, theta)
            ur = rotate_point(upper, c, theta)
            assert angle_from_vertical(lr, ur) == pytest.approx(base + theta, abs=1e-6)


class TestStats:
    def test_percentile_endpoints(self):
        xs = [1, 2, 3, 4, 5]
        assert percentile(xs, 0) == 1
        assert percentile(xs, 100) == 5

    def test_percentile_median(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3

    def test_percentile_interpolates(self):
        assert percentile([0, 10], 25) == pytest.approx(2.5)

    def test_percentile_single_value(self):
        assert percentile([7], 50) == 7

    def test_percentile_empty_is_nan(self):
        assert math.isnan(percentile([], 50))

    def test_stdev_known(self):
        assert stdev([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.1381, abs=1e-3)

    def test_stdev_needs_two(self):
        assert math.isnan(stdev([1]))

    def test_mean(self):
        assert mean([1, 2, 3, 4]) == pytest.approx(2.5)


class TestBasics:
    def test_midpoint(self):
        assert midpoint((0, 0), (10, 20)) == (5, 10)

    def test_distance_345(self):
        assert distance((0, 0), (3, 4)) == pytest.approx(5.0)
