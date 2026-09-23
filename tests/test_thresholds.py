"""Threshold provenance and band logic.

The project rule is that no threshold may present itself as grounded when it
is not. These tests enforce that mechanically, so it cannot rot.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from posture.thresholds import (BAND_NOTABLE, BAND_REFERENCE, BAND_SLIGHT,
                                DEFAULTS, ThresholdSpec, load_thresholds)

VALID_PROVENANCE = {"guess", "geometric-estimate", "literature-adjacent",
                    "population-percentile"}


class TestProvenanceDiscipline:
    def test_every_default_declares_provenance(self):
        for key, spec in DEFAULTS.items():
            assert spec.provenance in VALID_PROVENANCE, key

    def test_every_default_explains_its_basis(self):
        # A provenance tag with no explanation is not auditable.
        for key, spec in DEFAULTS.items():
            assert len(spec.basis) > 60, f"{key} basis is too thin to audit"

    def test_guesses_admit_they_are_guesses(self):
        # The wording must not dress a hand-picked number up as derived.
        for key, spec in DEFAULTS.items():
            if spec.provenance != "guess":
                continue
            text = spec.basis.lower()
            assert any(w in text for w in
                       ("chosen by hand", "hand-picked", "placeholder",
                        "no published cutoff", "hand")), key

    def test_thresholds_are_ordered(self):
        for key, spec in DEFAULTS.items():
            assert 0 < spec.slight < spec.notable, key

    def test_direction_is_valid(self):
        for key, spec in DEFAULTS.items():
            assert spec.direction in ("positive_only", "symmetric"), key


class TestClassification:
    def test_symmetric_uses_absolute_value(self):
        s = ThresholdSpec("k", "x", slight=2.0, notable=4.0,
                          direction="symmetric", provenance="guess", basis="b")
        assert s.classify(0.5) == BAND_REFERENCE
        assert s.classify(-3.0) == BAND_SLIGHT
        assert s.classify(-5.0) == BAND_NOTABLE
        assert s.classify(5.0) == BAND_NOTABLE

    def test_positive_only_ignores_negative_side(self):
        s = ThresholdSpec("k", "x", slight=5.0, notable=10.0,
                          direction="positive_only", provenance="guess", basis="b")
        assert s.classify(-20.0) == BAND_REFERENCE
        assert s.classify(7.0) == BAND_SLIGHT
        assert s.classify(12.0) == BAND_NOTABLE

    def test_boundary_is_inclusive(self):
        s = ThresholdSpec("k", "x", slight=2.0, notable=4.0,
                          direction="symmetric", provenance="guess", basis="b")
        assert s.classify(2.0) == BAND_SLIGHT
        assert s.classify(4.0) == BAND_NOTABLE


class TestReferenceDistributionLoading:
    def test_missing_file_keeps_defaults_and_tags(self):
        specs = load_thresholds("/nonexistent/reference.json")
        assert specs["lateral_head_shift"].provenance == "guess"
        assert specs["lateral_head_shift"].slight == DEFAULTS["lateral_head_shift"].slight

    def test_percentiles_upgrade_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                json.dump({"source": "unit test",
                           "metrics": {"lateral_head_shift": {
                               "n": 120, "abs_p80": 14.2, "abs_p95": 23.9}}}, fh)
            specs = load_thresholds(path)
        s = specs["lateral_head_shift"]
        assert s.provenance == "population-percentile"
        assert s.slight == 14.2 and s.notable == 23.9
        assert s.n == 120

    def test_small_sample_does_not_upgrade(self):
        # Percentiles from a handful of images are noise, and silently adopting
        # them would replace an honest guess with a dishonest measurement.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                json.dump({"metrics": {"lateral_head_shift": {
                    "n": 9, "abs_p80": 14.2, "abs_p95": 23.9}}}, fh)
            specs = load_thresholds(path)
        s = specs["lateral_head_shift"]
        assert s.provenance == "guess"
        assert s.slight == DEFAULTS["lateral_head_shift"].slight
        assert "n=9" in s.basis

    def test_percentile_basis_disclaims_clinical_meaning(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                json.dump({"metrics": {"shoulder_tilt": {
                    "n": 99, "abs_p80": 5.0, "abs_p95": 9.0}}}, fh)
            specs = load_thresholds(path)
        basis = specs["shoulder_tilt"].basis
        assert "NOT" in basis and "clinical" in basis.lower()

    def test_corrupt_file_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                fh.write("{not json")
            specs = load_thresholds(path)
        assert specs["lateral_head_shift"].provenance == "guess"

    def test_loading_does_not_mutate_defaults(self):
        before = DEFAULTS["lateral_head_shift"].slight
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                json.dump({"metrics": {"lateral_head_shift": {
                    "n": 99, "abs_p80": 99.0, "abs_p95": 111.0}}}, fh)
            load_thresholds(path)
        assert DEFAULTS["lateral_head_shift"].slight == before


class TestCriterionReferenced:
    """Sagittal verdicts are judged against upright, not against a crowd."""

    def test_percentiles_never_replace_criterion_cuts(self):
        from posture.thresholds import CRITERION_REFERENCED
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                json.dump({"metrics": {k: {"n": 500, "abs_p80": 14.6, "abs_p95": 33.1}
                                       for k in CRITERION_REFERENCED}}, fh)
            specs = load_thresholds(path)
        for k in CRITERION_REFERENCED:
            assert specs[k].slight == DEFAULTS[k].slight, k
            assert specs[k].provenance != "population-percentile", k

    def test_the_forward_head_photo_the_percentiles_missed_is_notable(self):
        # Regression for a real miss. A user-supplied side photo with an
        # obvious forward head read head_over_hip = +11.66 +/- 0.99 deg, and
        # the percentile cut (14.6) called it normal. Only the numbers are
        # kept here; the photo itself is not in the repo (unknown licence).
        from posture.assess import MetricVerdict, _below_noise_floor, _resolve_band
        from posture.metrics import Measurement
        from posture.report import LEVEL_NOTABLE, level_for
        spec = load_thresholds()["head_over_hip"]
        m = Measurement("head_over_hip", "x", 11.66, 0.99, "sagittal", 500.0)
        band, resolved, alt = _resolve_band(m, spec)
        v = MetricVerdict(measurement=m, band=band, band_label_zh="", spec=spec,
                          resolved=resolved, alternative_band=alt,
                          below_noise_floor=_below_noise_floor(m, spec))
        assert level_for(v) == LEVEL_NOTABLE

    def test_upright_studio_readings_stay_normal(self):
        # The three upright studio side photos the zero was checked against.
        spec = load_thresholds()["head_over_hip"]
        for reading in (-0.7, 0.3, 0.6):
            assert spec.classify(reading + 1.0) == BAND_REFERENCE
