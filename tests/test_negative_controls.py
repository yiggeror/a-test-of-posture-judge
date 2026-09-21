"""Regression tests for the confirmed false-positive failures.

The previous phase documented that MediaPipe produces a full 33-point skeleton
for a photo of a bare palm and for a sock/calf close-up, with high `visibility`
scores on every point. These tests pin that such inputs never reach a verdict.

They run against real images in testdata/negative/ (COCO images containing no
person at all, under redistributable licenses) and against synthetic inputs.
They are skipped rather than failed when the models are not installed, so the
suite still runs after a bare checkout.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from posture import assess
from posture import landmarks as L

NEG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "testdata", "negative")


def models_available() -> bool:
    return os.path.exists(L.DEFAULT_MODEL)


requires_model = pytest.mark.skipif(not models_available(),
                                    reason="models not installed; run scripts/setup.sh")


def negative_images() -> list[str]:
    return sorted(glob.glob(os.path.join(NEG_DIR, "*.jpg")))


@requires_model
class TestNonPersonImages:
    def test_negative_set_is_present(self):
        assert negative_images(), (
            "testdata/negative/ is empty; the non-person regression set is "
            "part of the repo, see testdata/negative/sources.csv")

    @pytest.mark.parametrize("path", negative_images() or ["<none>"])
    def test_no_verdict_from_a_non_person_image(self, path):
        if path == "<none>":
            pytest.skip("no negative images present")
        a = assess(L.load_rgb(path))
        # Either no pose at all, or a pose that every verdict is withheld from.
        assert not a.ok, f"{os.path.basename(path)} produced a verdict"
        assert not a.verdicts, f"{os.path.basename(path)} leaked verdicts"


@requires_model
class TestSyntheticNonImages:
    def test_flat_colour_produces_no_verdict(self):
        rgb = np.full((600, 400, 3), 128, dtype=np.uint8)
        a = assess(rgb)
        assert not a.ok and not a.verdicts

    def test_random_noise_produces_no_verdict(self):
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, (600, 400, 3), dtype=np.uint8)
        a = assess(rgb)
        assert not a.ok and not a.verdicts

    def test_tiny_image_produces_no_verdict(self):
        rgb = np.full((24, 16, 3), 200, dtype=np.uint8)
        a = assess(rgb)
        assert not a.ok and not a.verdicts


@requires_model
class TestCroppedBodyParts:
    """A close crop of one body part is the palm/sock failure mode."""

    def _crop_region(self, path, box):
        rgb = L.load_rgb(path)
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = (int(box[0] * w), int(box[1] * h),
                          int(box[2] * w), int(box[3] * h))
        return np.ascontiguousarray(rgb[y0:y1, x0:x1])

    def test_head_crop_does_not_yield_full_body_verdict(self):
        imgs = sorted(glob.glob(os.path.join(
            os.path.dirname(NEG_DIR), "front", "*.jpg")))
        if not imgs:
            pytest.skip("no front testdata")
        # Top fifth of the frame: a head-and-shoulders crop has no legs, so a
        # full-body reading from it would be fabricated.
        crop = self._crop_region(imgs[0], (0.2, 0.0, 0.8, 0.2))
        if min(crop.shape[:2]) < 32:
            pytest.skip("crop too small")
        a = assess(crop)
        assert not a.ok or not a.verdicts
