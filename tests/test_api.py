"""JSON API contract tests.

The API is what a mini-program or mobile client talks to, so its shape is a
contract rather than an implementation detail. In particular the provenance of
every threshold ships with the reading: a client that displays a band without
it is misrepresenting a hand-picked number as a measured one.
"""
from __future__ import annotations

import glob
import io
import json
import os

import numpy as np
import pytest

from api import api as flask_api
from posture import landmarks as L


def models_available() -> bool:
    return os.path.exists(L.DEFAULT_MODEL)


requires_model = pytest.mark.skipif(not models_available(),
                                    reason="models not installed")


@pytest.fixture
def client():
    flask_api.config["TESTING"] = True
    return flask_api.test_client()


class TestHealth:
    def test_health_reports_model_presence(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        d = r.get_json()
        assert "pose_model_present" in d
        assert d["api_version"] == "1"


class TestErrors:
    def test_missing_image_is_a_400(self, client):
        r = client.post("/api/v1/assess", json={})
        assert r.status_code == 400
        assert r.get_json()["ok"] is False

    def test_bad_base64_is_a_400_not_a_crash(self, client):
        r = client.post("/api/v1/assess", json={"image_base64": "!!!not base64!!!"})
        assert r.status_code == 400
        assert "base64" in r.get_json()["error"]

    def test_tiny_payload_rejected(self, client):
        r = client.post("/api/v1/assess", json={"image_base64": "aGk="})
        assert r.status_code == 400


@requires_model
class TestAssess:
    def _post(self, client, path):
        with open(path, "rb") as fh:
            return client.post("/api/v1/assess",
                               data={"photo": (io.BytesIO(fh.read()), "p.jpg")},
                               content_type="multipart/form-data")

    def test_front_photo_returns_metrics(self, client):
        imgs = sorted(glob.glob("testdata/pexels/front/*.jpg"))
        if not imgs:
            pytest.skip("no front testdata")
        d = self._post(client, imgs[0]).get_json()
        assert d["api_version"] == "1"
        assert d["view"] in ("front", "side", "oblique")
        assert isinstance(d["metrics"], list) and d["metrics"]

    def test_every_metric_ships_its_threshold_provenance(self, client):
        imgs = sorted(glob.glob("testdata/pexels/front/*.jpg"))
        if not imgs:
            pytest.skip("no front testdata")
        d = self._post(client, imgs[0]).get_json()
        for m in d["metrics"]:
            if m["diagnostic_only"]:
                continue
            assert m["threshold_provenance"] in (
                "guess", "geometric-estimate", "literature-adjacent",
                "population-percentile")

    def test_band_is_null_when_not_resolved(self, client):
        # A client must never be handed a band the tool did not commit to.
        imgs = sorted(glob.glob("testdata/pexels/front/*.jpg"))
        if not imgs:
            pytest.skip("no front testdata")
        d = self._post(client, imgs[0]).get_json()
        for m in d["metrics"]:
            if not m["resolved"]:
                assert m["band"] is None

    def test_non_person_returns_ok_false_with_reasons(self, client):
        imgs = sorted(glob.glob("testdata/pexels/negative/neg_animal*.jpg"))
        if not imgs:
            pytest.skip("no negative testdata")
        d = self._post(client, imgs[0]).get_json()
        assert d["ok"] is False

    def test_cropped_photo_lists_unavailable_metrics(self, client):
        imgs = sorted(glob.glob("testdata/pexels/side/side_feetcropped*.jpg"))
        if not imgs:
            pytest.skip("no cropped testdata")
        d = self._post(client, imgs[0]).get_json()
        keys = {u["key"] for u in d["unavailable"]}
        assert "trunk_sway" in keys
        assert "head_over_hip" not in keys

    def test_disclaimer_always_present(self, client):
        imgs = sorted(glob.glob("testdata/pexels/front/*.jpg"))
        if not imgs:
            pytest.skip("no front testdata")
        d = self._post(client, imgs[0]).get_json()
        assert "不是医学诊断" in d["disclaimer"]

    def test_response_is_json_serialisable_end_to_end(self, client):
        imgs = sorted(glob.glob("testdata/pexels/side/*.jpg"))
        if not imgs:
            pytest.skip("no side testdata")
        r = self._post(client, imgs[0])
        json.dumps(r.get_json())   # would raise on a stray numpy type
