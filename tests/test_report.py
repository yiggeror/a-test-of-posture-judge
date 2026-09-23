"""The plain-language report: what a person who uploaded a photo reads.

The rule under test is that caution survives translation. A reading whose
error bar reaches a milder band is reported at the milder band; an
unresolvable reading is reported as not measured, never as fine; a photo on
which nothing could be judged is a retake, never "no problems found".
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

from posture import landmarks as L
from posture.assess import Assessment, MetricVerdict, _below_noise_floor, _resolve_band
from posture.metrics import Measurement, NoiseModel
from posture.report import (COPY, LEVEL_BORDERLINE, LEVEL_GOOD, LEVEL_NOTABLE,
                            LEVEL_NOT_MEASURED, LEVEL_SLIGHT, consumer_report,
                            level_for)
from posture.thresholds import BAND_LABELS_ZH, ThresholdSpec

from .conftest import make_pose

POS = ThresholdSpec("head_over_hip", "x", slight=10.0, notable=20.0,
                    direction="positive_only", provenance="guess", basis="b")
SYM = ThresholdSpec("shoulder_tilt", "x", slight=6.0, notable=12.0,
                    direction="symmetric", provenance="guess", basis="b")


def verdict(value, unc, spec=POS):
    m = Measurement(key=spec.key, label="x", value=value, uncertainty=unc,
                    plane="sagittal", span_px=100.0)
    band, resolved, alt = _resolve_band(m, spec)
    floored = _below_noise_floor(m, spec)
    return MetricVerdict(measurement=m, band=band, band_label_zh=BAND_LABELS_ZH[band],
                         spec=spec, resolved=resolved and not floored,
                         alternative_band=alt, below_noise_floor=floored)


class TestLevels:
    def test_clearly_normal_is_good(self):
        assert level_for(verdict(3.0, 1.0)) == LEVEL_GOOD

    def test_straddling_the_first_cut_is_borderline_not_good(self):
        assert level_for(verdict(9.5, 1.0)) == LEVEL_BORDERLINE
        assert level_for(verdict(10.5, 1.0)) == LEVEL_BORDERLINE

    def test_clearly_slight_is_slight(self):
        assert level_for(verdict(15.0, 1.0)) == LEVEL_SLIGHT

    def test_straddling_slight_and_notable_is_reported_as_slight(self):
        # The person is told the least severe thing the reading cannot rule
        # out. Never rounded up.
        assert level_for(verdict(20.5, 1.0)) == LEVEL_SLIGHT
        assert level_for(verdict(19.5, 1.0)) == LEVEL_SLIGHT

    def test_clearly_notable_is_notable(self):
        assert level_for(verdict(30.0, 1.0)) == LEVEL_NOTABLE

    def test_interval_spanning_all_three_bands_is_borderline(self):
        # alternative_band keeps only the most severe neighbour, so a mapping
        # built on it would say "slight" here. The interval reaches normal.
        v = verdict(15.0, 6.0, ThresholdSpec("k", "x", slight=10.0, notable=17.0,
                                             direction="positive_only",
                                             provenance="guess", basis="b"))
        assert not v.below_noise_floor
        assert level_for(v) == LEVEL_BORDERLINE

    def test_below_noise_floor_is_not_measured_never_good(self):
        v = verdict(3.0, 11.0)
        assert v.below_noise_floor
        assert level_for(v) == LEVEL_NOT_MEASURED

    def test_symmetric_metric_uses_distance_from_zero(self):
        assert level_for(verdict(-15.0, 1.0, SYM)) == LEVEL_NOTABLE
        assert level_for(verdict(-8.0, 1.0, SYM)) == LEVEL_SLIGHT
        # interval crosses zero -> nearest point is 0 -> can't be worse than
        # borderline, and here the far end stays normal
        assert level_for(verdict(0.5, 1.0, SYM)) == LEVEL_GOOD


def _assessment(view, verdicts, measurements, *, blocked=False, findings=(),
                unavailable=None, unstable=None, pose=None):
    from posture.view import ViewEstimate
    ve = ViewEstimate(view=view, yaw_deg=10.0, shoulder_spread=0.1,
                      hip_spread=0.1, facing=1, facing_confidence=1.0) \
        if view else None
    return Assessment(ok=not blocked, view=ve, measurements=measurements,
                      verdicts=verdicts, findings=list(findings), blocked=blocked,
                      noise=NoiseModel.load(), unavailable=unavailable or {},
                      unstable=unstable or {}, pose=pose)


class TestReport:
    def test_no_person_is_a_retake(self):
        r = consumer_report(_assessment(None, {}, {}))
        assert r["status"] == "retake" and r["retake"][0]["key"] == "no_pose"

    def test_blocked_photo_names_what_to_change(self):
        from posture.guards import GuardFinding
        f = GuardFinding("oblique_view", "block", "x")
        r = consumer_report(_assessment("oblique", {}, {}, blocked=True, findings=[f]))
        assert r["status"] == "retake"
        assert r["retake"][0]["title"] == COPY["retake"]["oblique_view"]["title"]

    def test_issue_carries_plain_copy_and_exercises(self):
        v = verdict(30.0, 1.0)
        r = consumer_report(_assessment("side", {"head_over_hip": v},
                                        {"head_over_hip": v.measurement}))
        assert r["status"] == "ok"
        (i,) = r["issues"]
        assert i["name"] == "头部前伸" and i["level"] == LEVEL_NOTABLE
        assert len(i["tips"]) == 3 and all(t["name"] and t["how"] for t in i["tips"])

    def test_nothing_judged_is_a_retake_not_all_clear(self):
        v = verdict(3.0, 11.0)          # below noise floor
        r = consumer_report(_assessment("side", {"head_over_hip": v},
                                        {"head_over_hip": v.measurement}))
        assert r["status"] == "retake"
        assert r["issues"] == [] and r["good"] == []

    def test_withheld_metrics_are_listed_with_a_reason(self):
        good = verdict(3.0, 1.0)
        m2 = Measurement("trunk_sway", "x", 3.0, 1.0, "sagittal", 100.0,
                         (L.LEFT_ANKLE, L.LEFT_HIP))
        r = consumer_report(_assessment(
            "side", {"head_over_hip": good},
            {"head_over_hip": good.measurement, "trunk_sway": m2},
            unavailable={"trunk_sway": ["left_ankle"]}))
        (nm,) = r["not_measured"]
        assert nm["key"] == "trunk_sway" and "脚" in nm["reason"]

    def test_trunk_sway_direction_picks_the_copy(self):
        spec = ThresholdSpec("trunk_sway", "x", 6.0, 10.0, "symmetric", "guess", "b")
        fwd, back = verdict(11.0, 0.5, spec), verdict(-11.0, 0.5, spec)
        rf = consumer_report(_assessment("side", {"trunk_sway": fwd}, {"trunk_sway": fwd.measurement}))
        rb = consumer_report(_assessment("side", {"trunk_sway": back}, {"trunk_sway": back.measurement}))
        assert rf["issues"][0]["name"] == "重心前移"
        assert rb["issues"][0]["name"] == "重心后移"

    def test_shoulder_side_is_in_image_terms(self):
        # Raise the landmark that sits on the image-left; the report must say
        # "画面左侧" regardless of which anatomical side that landmark is.
        pose = make_pose(view="front")
        l, r_ = pose.landmarks[L.LEFT_SHOULDER], pose.landmarks[L.RIGHT_SHOULDER]
        left_idx = L.LEFT_SHOULDER if l.x < r_.x else L.RIGHT_SHOULDER
        lm = pose.landmarks[left_idx]
        pose.landmarks[left_idx] = L.Landmark(lm.x, lm.y - 60, lm.z, 1.0, 1.0)
        v = verdict(15.0, 1.0, SYM)
        rep = consumer_report(_assessment("front", {"shoulder_tilt": v},
                                          {"shoulder_tilt": v.measurement}, pose=pose))
        assert rep["issues"][0]["side"] == "left"
        assert "画面左侧" in rep["issues"][0]["summary"]


class TestCopy:
    def test_every_judged_metric_has_copy(self):
        from posture.thresholds import DEFAULTS, DIAGNOSTIC_ONLY
        for key in DEFAULTS:
            if key not in DIAGNOSTIC_ONLY:
                assert key in COPY["issues"], key

    def test_every_blocking_guard_has_retake_copy(self):
        import inspect
        from posture import assess as A
        from posture import guards as G
        src = inspect.getsource(G) + inspect.getsource(A)
        import re
        keys = set(re.findall(r'key="([a-z_]+)",\s*severity=(?:G\.)?SEVERITY_BLOCK', src))
        assert keys, "found no blocking guard keys; the pattern is stale"
        for k in keys:
            assert k in COPY["retake"], k

    def test_copy_never_claims_to_diagnose(self):
        # Project rule: 参考 / 倾向, never 诊断 -- not even as a denial here.
        text = json.dumps(COPY, ensure_ascii=False)
        assert "诊断" not in text

    def test_disclaimer_is_one_sentence(self):
        d = COPY["disclaimer"]
        assert d.count("。") == 1 and d.endswith("。")


requires_model = pytest.mark.skipif(not os.path.exists(L.DEFAULT_MODEL),
                                    reason="models not installed")


@requires_model
class TestOnRealPhotos:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_studio_side_photo_gives_an_ok_report(self):
        from posture.assess import assess_file
        a = assess_file(os.path.join(self.ROOT, "testdata/pexels/side/side_pexels_30033728.jpg"),
                        deep_guards=False)
        r = consumer_report(a)
        assert r["status"] == "ok" and r["view"] == "side"
        assert r["other_view"]["view"] == "front"
        assert len(r["issues"]) + len(r["borderline"]) + len(r["good"]) >= 2

    def test_oblique_photo_is_a_retake(self):
        from posture.assess import assess_file
        a = assess_file(os.path.join(self.ROOT, "testdata/pexels/edge/edge_rotated53_16135927.jpg"),
                        deep_guards=False)
        assert consumer_report(a)["status"] == "retake"


class TestExifOrientation:
    def test_portrait_phone_photo_is_loaded_upright(self, tmp_path):
        # 600x400 landscape pixels tagged "rotate 90 CW" (orientation 6), the
        # way a phone stores a portrait shot. Loaded upright it is 400x600.
        from PIL import Image
        im = Image.fromarray(np.zeros((400, 600, 3), dtype=np.uint8))
        exif = im.getexif()
        exif[0x0112] = 6
        p = tmp_path / "phone.jpg"
        im.save(p, exif=exif)
        arr = L.load_rgb(str(p))
        assert arr.shape[:2] == (600, 400)
