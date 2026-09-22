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
        assert specs["head_over_hip"].provenance == "guess"
        assert specs["head_over_hip"].slight == DEFAULTS["head_over_hip"].slight

    def test_percentiles_upgrade_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                json.dump({"source": "unit test",
                           "metrics": {"head_over_hip": {
                               "n": 120, "abs_p80": 14.2, "abs_p95": 23.9}}}, fh)
            specs = load_thresholds(path)
        s = specs["head_over_hip"]
        assert s.provenance == "population-percentile"
        assert s.slight == 14.2 and s.notable == 23.9
        assert s.n == 120

    def test_small_sample_does_not_upgrade(self):
        # Percentiles from a handful of images are noise, and silently adopting
        # them would replace an honest guess with a dishonest measurement.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                json.dump({"metrics": {"head_over_hip": {
                    "n": 9, "abs_p80": 14.2, "abs_p95": 23.9}}}, fh)
            specs = load_thresholds(path)
        s = specs["head_over_hip"]
        assert s.provenance == "guess"
        assert s.slight == DEFAULTS["head_over_hip"].slight
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
        assert specs["head_over_hip"].provenance == "guess"

    def test_loading_does_not_mutate_defaults(self):
        before = DEFAULTS["head_over_hip"].slight
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                json.dump({"metrics": {"head_over_hip": {
                    "n": 99, "abs_p80": 99.0, "abs_p95": 111.0}}}, fh)
            load_thresholds(path)
        assert DEFAULTS["head_over_hip"].slight == before
