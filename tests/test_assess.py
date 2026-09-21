"""Verdict assembly, including the uncertainty-versus-threshold rule."""
from __future__ import annotations

import pytest

from posture.assess import DISCLAIMER_ZH, _resolve_band
from posture.metrics import Measurement
from posture.thresholds import (BAND_NOTABLE, BAND_REFERENCE, BAND_SLIGHT,
                                ThresholdSpec)

SPEC = ThresholdSpec("k", "测试项", slight=10.0, notable=20.0,
                     direction="positive_only", provenance="guess", basis="b")


def meas(value, unc):
    return Measurement(key="k", label="x", value=value, uncertainty=unc,
                       plane="sagittal", span_px=100.0)


class TestBandResolution:
    def test_clearly_inside_a_band_is_resolved(self):
        band, resolved, alt = _resolve_band(meas(3.0, 1.0), SPEC)
        assert band == BAND_REFERENCE and resolved is True and alt is None

    def test_error_bar_crossing_a_threshold_is_unresolved(self):
        # 9.5 +/- 1.0 straddles the 10.0 cut, so the honest answer is that the
        # band cannot be determined, not whichever side the point landed on.
        band, resolved, alt = _resolve_band(meas(9.5, 1.0), SPEC)
        assert resolved is False
        assert {band, alt} == {BAND_REFERENCE, BAND_SLIGHT}

    def test_unresolved_reports_the_more_severe_alternative(self):
        band, resolved, alt = _resolve_band(meas(19.5, 1.5), SPEC)
        assert resolved is False
        assert alt == BAND_NOTABLE

    def test_tight_uncertainty_resolves_near_a_threshold(self):
        band, resolved, _ = _resolve_band(meas(9.5, 0.1), SPEC)
        assert band == BAND_REFERENCE and resolved is True

    def test_notable_well_clear_is_resolved(self):
        band, resolved, _ = _resolve_band(meas(30.0, 2.0), SPEC)
        assert band == BAND_NOTABLE and resolved is True

    def test_nan_uncertainty_is_never_claimed_resolved(self):
        band, resolved, _ = _resolve_band(meas(3.0, float("nan")), SPEC)
        assert resolved is False

    def test_wide_uncertainty_is_unresolved_even_mid_band(self):
        # A reading whose error bar spans two thresholds must not be presented
        # as a finding.
        _, resolved, _ = _resolve_band(meas(15.0, 8.0), SPEC)
        assert resolved is False


class TestWording:
    def test_disclaimer_avoids_diagnostic_language(self):
        # Project rule: 参考 / 倾向, never 诊断.
        assert "诊断" in DISCLAIMER_ZH  # appears only as a denial
        assert "不是医学诊断" in DISCLAIMER_ZH

    def test_user_facing_strings_never_claim_to_diagnose(self):
        # 诊断 means medical diagnosis in Chinese, so it may appear ONLY inside
        # an explicit denial. Everything else says 参考 or 倾向. This caught a
        # real slip: a diagnostic-only metric was once labelled
        # 「仅供诊断参考」, which reads as "for diagnostic reference".
        import os

        from posture.assess import ADVICE_ZH
        from posture.thresholds import BAND_LABELS_ZH, DEFAULTS

        strings = list(BAND_LABELS_ZH.values())
        strings += [s.label_zh for s in DEFAULTS.values()]
        strings += [line for lines in ADVICE_ZH.values() for line in lines]

        tpl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "templates", "index.html")
        with open(tpl, encoding="utf-8") as fh:
            html = fh.read()
        # Strip the sanctioned denials, then nothing may be left.
        stripped = html.replace("不是医学诊断", "")
        strings.append(stripped)

        for s in strings:
            assert "诊断" not in s, f"diagnostic wording leaked: {s[:80]!r}"

    def test_band_labels_are_tentative(self):
        from posture.thresholds import BAND_LABELS_ZH
        # Every band reads as a tendency or a range, never as a finding.
        for band, label in BAND_LABELS_ZH.items():
            assert any(w in label for w in ("参考", "倾向")), (band, label)

    def test_advice_exists_for_every_judged_metric(self):
        from posture.assess import ADVICE_ZH
        from posture.thresholds import DEFAULTS
        for key in DEFAULTS:
            assert key in ADVICE_ZH, f"no advice text for {key}"
            assert ADVICE_ZH[key], key


class TestNoiseFloor:
    def test_uncertainty_wider_than_the_middle_band_is_unresolvable(self):
        from posture.assess import _below_noise_floor
        # SPEC bands are 10/20, so `slight` is 10 deg wide.
        assert _below_noise_floor(meas(12.0, 11.0), SPEC) is True

    def test_precise_measurement_is_resolvable(self):
        from posture.assess import _below_noise_floor
        assert _below_noise_floor(meas(12.0, 2.0), SPEC) is False

    def test_nan_uncertainty_is_not_called_noise_floored(self):
        from posture.assess import _below_noise_floor
        assert _below_noise_floor(meas(12.0, float("nan")), SPEC) is False

    def test_display_explains_rather_than_offering_two_bands(self):
        from posture.assess import MetricVerdict
        v = MetricVerdict(measurement=meas(12.0, 11.0), band=BAND_SLIGHT,
                          band_label_zh="轻度倾向", spec=SPEC, resolved=False,
                          alternative_band=BAND_NOTABLE, below_noise_floor=True)
        text = v.display_band_zh
        assert "测量精度不足以判定" in text
        # It must not present a band the reader could latch onto.
        assert "/" not in text

    def test_noise_floored_metric_offers_no_advice(self):
        from posture.assess import MetricVerdict
        v = MetricVerdict(measurement=meas(12.0, 11.0), band=BAND_SLIGHT,
                          band_label_zh="轻度倾向", spec=SPEC, resolved=False,
                          alternative_band=BAND_NOTABLE, below_noise_floor=True,
                          advice_zh=[])
        assert v.advice_zh == []
