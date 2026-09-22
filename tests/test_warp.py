"""Geometry of the known-posture-change harness.

The warp is what makes `scripts/synthetic_warp.py` a ground truth, so its
geometry is pinned here rather than trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.synthetic_warp import ramp_for, warp, warped_pose

from .conftest import make_pose


class TestRamp:
    def test_zero_at_the_hips(self):
        assert ramp_for(np.array([500.0]), y_hip=500.0, y_ear=100.0)[0] == \
            pytest.approx(0.0)

    def test_one_at_the_ears(self):
        assert ramp_for(np.array([100.0]), y_hip=500.0, y_ear=100.0)[0] == \
            pytest.approx(1.0)

    def test_monotonic_between(self):
        ys = np.linspace(500.0, 100.0, 20)
        r = ramp_for(ys, y_hip=500.0, y_ear=100.0)
        assert np.all(np.diff(r) >= -1e-9)

    def test_clamped_outside(self):
        # Below the hips nothing moves; above the ears the head stays rigid.
        assert ramp_for(np.array([900.0]), 500.0, 100.0)[0] == pytest.approx(0.0)
        assert ramp_for(np.array([10.0]), 500.0, 100.0)[0] == pytest.approx(1.0)

    def test_degenerate_span_is_safe(self):
        assert ramp_for(np.array([100.0]), 100.0, 100.0)[0] == pytest.approx(0.0)


class TestWarp:
    def test_zero_shift_is_identity(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (40, 30, 3), dtype=np.uint8)
        assert np.array_equal(warp(img, 0.0, 35.0, 5.0), img)

    def test_shape_preserved(self):
        img = np.zeros((40, 30, 3), dtype=np.uint8)
        assert warp(img, 7.0, 35.0, 5.0).shape == img.shape

    def test_rows_below_the_hips_are_untouched(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (40, 30, 3), dtype=np.uint8)
        out = warp(img, 9.0, y_hip=20.0, y_ear=2.0)
        assert np.array_equal(out[25:], img[25:])

    def test_head_row_actually_moves(self):
        img = np.zeros((40, 30, 3), dtype=np.uint8)
        img[:, 10:12] = 255                      # a vertical bar
        out = warp(img, 6.0, y_hip=35.0, y_ear=2.0)
        # near the top the bar should have shifted right
        assert out[1, 10:12].max() < 255
        assert out[1, 16:18].max() > 0


class TestWarpedPose:
    def test_hip_does_not_move(self):
        p = make_pose(view="side")
        from posture import landmarks as L
        y_hip = p.landmarks[L.LEFT_HIP].y
        y_ear = p.landmarks[L.LEFT_EAR].y
        w = warped_pose(p, 50.0, y_hip, y_ear)
        assert w.landmarks[L.LEFT_HIP].x == pytest.approx(
            p.landmarks[L.LEFT_HIP].x, abs=1e-6)

    def test_ear_moves_by_the_full_shift(self):
        p = make_pose(view="side")
        from posture import landmarks as L
        y_hip = p.landmarks[L.LEFT_HIP].y
        y_ear = p.landmarks[L.LEFT_EAR].y
        w = warped_pose(p, 50.0, y_hip, y_ear)
        assert w.landmarks[L.LEFT_EAR].x == pytest.approx(
            p.landmarks[L.LEFT_EAR].x + 50.0, abs=1e-6)

    def test_y_is_never_changed(self):
        p = make_pose(view="side")
        w = warped_pose(p, 33.0, 500.0, 100.0)
        for a, b in zip(p.landmarks, w.landmarks):
            assert a.y == pytest.approx(b.y)

    def test_shift_changes_the_metric_in_the_right_direction(self):
        # A forward shear must increase head_over_hip for a right-facing
        # subject. This is the sign check the whole harness rests on.
        from posture import landmarks as L
        from posture.metrics import NoiseModel, compute_sagittal
        from posture.view import estimate_view
        p = make_pose(view="side", facing=+1)
        v = estimate_view(p)
        n = NoiseModel(0.004, "test")
        y_hip = p.landmarks[L.LEFT_HIP].y
        y_ear = p.landmarks[L.LEFT_EAR].y
        before = compute_sagittal(p, v, n)["head_over_hip"].value
        after = compute_sagittal(warped_pose(p, 60.0, y_hip, y_ear), v, n)["head_over_hip"].value
        assert after > before
