#!/usr/bin/env python3
"""JSON API over the assessment pipeline.

Separated from app.py because the HTML page is one client, not the contract.
A mini-program, a mobile app or a batch job all need the same structured
result, and none of them can parse a Jinja template.

Deployment shape this assumes: the client uploads a photo, the server runs
MediaPipe and returns JSON. The alternative -- running the model on-device --
is constrained by model size and is discussed in reports/DEPLOYMENT.md.

Privacy: a full-body photograph is sensitive. This service holds the image in
memory for the duration of the request and never writes it to disk. Nothing
here logs image bytes. That is a property of this file and callers should not
add storage without deciding the retention question first.
"""
from __future__ import annotations

import base64
import binascii
import io
import os

from flask import Flask, jsonify, request

from posture import assess
from posture import guards as G
from posture import landmarks as L
from posture.assess import DISCLAIMER_ZH
from posture.thresholds import DIAGNOSTIC_ONLY

api = Flask(__name__)
api.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024

API_VERSION = "1"

# User-visible labels for metrics that never reach a ThresholdSpec.
METRIC_LABELS = {
    "head_over_hip": "头部前移", "shoulder_protraction": "圆肩（肩前移）",
    "trunk_sway": "躯干前后倾", "shoulder_tilt": "高低肩",
    "pelvis_tilt": "骨盆侧倾", "lateral_head_shift": "头部侧偏",
    "forward_head": "耳肩角", "head_tilt": "头部侧倾",
    "head_vs_shoulder_tilt": "头肩相对侧倾", "knee_deviation": "膝关节角度",
}

VIEW_ZH = {"front": "正面", "side": "侧面", "oblique": "斜侧"}


def _read_image():
    """Accept either multipart `photo` or JSON {"image_base64": "..."}."""
    f = request.files.get("photo")
    if f and f.filename:
        return L.load_rgb(io.BytesIO(f.read())), None

    body = request.get_json(silent=True) or {}
    b64 = body.get("image_base64")
    if not b64:
        return None, "missing image: send multipart 'photo' or JSON 'image_base64'"
    if "," in b64[:64]:           # tolerate a data: URL prefix
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return None, "image_base64 is not valid base64"
    if len(raw) < 128:
        return None, "image_base64 decoded to almost nothing"
    return L.load_rgb(io.BytesIO(raw)), None


def _metric_payload(a) -> list[dict]:
    out = []
    for key, m in a.measurements.items():
        if key in a.unavailable:
            continue
        v = a.verdicts.get(key)
        out.append({
            "key": key,
            "label": METRIC_LABELS.get(key, key),
            "value_deg": round(m.value, 1),
            "uncertainty_deg": round(m.uncertainty, 1),
            "plane": m.plane,
            "diagnostic_only": key in DIAGNOSTIC_ONLY,
            # `band` is null for diagnostics and for anything the tool declines
            # to judge. Clients must not invent one.
            "band": (v.band if (v and v.resolved) else None),
            "band_label": (v.display_band_zh if v else None),
            "resolved": (v.resolved if v else None),
            "below_noise_floor": (v.below_noise_floor if v else None),
            "threshold_slight": (v.spec.slight if v else None),
            "threshold_notable": (v.spec.notable if v else None),
            # Ship the provenance. A client that hides it is misrepresenting
            # the number: some cuts are measured percentiles and one is a guess.
            "threshold_provenance": (v.spec.provenance if v else None),
            "threshold_n": (v.spec.n if v else None),
            "advice": (v.advice_zh if v else []),
        })
    out.sort(key=lambda r: (r["diagnostic_only"], r["plane"] != "sagittal"))
    return out


@api.route("/api/v1/assess", methods=["POST"])
def do_assess():
    try:
        rgb, err = _read_image()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"unreadable image: {exc}"}), 400
    if err:
        return jsonify({"ok": False, "error": err}), 400

    a = assess(rgb, deep_guards=request.args.get("deep", "1") != "0")

    if a.view is None:
        return jsonify({
            "ok": False, "api_version": API_VERSION,
            "error": a.error_zh or "no person detected",
            "disclaimer": DISCLAIMER_ZH,
        }), 200

    return jsonify({
        "ok": a.ok,
        "api_version": API_VERSION,
        "view": a.view.view,
        "view_label": VIEW_ZH.get(a.view.view, a.view.view),
        "yaw_deg": round(a.view.yaw_deg, 1),
        # Everything that stopped a verdict, as structured data the client can
        # turn into shooting guidance rather than a wall of text.
        "blocked_by": [{"key": f.key, "message": f.message_zh, "detail": f.detail}
                       for f in a.findings if f.severity == G.SEVERITY_BLOCK],
        "warnings": [{"key": f.key, "message": f.message_zh, "detail": f.detail}
                     for f in a.findings if f.severity == G.SEVERITY_WARN],
        "unavailable": [{"key": k, "label": METRIC_LABELS.get(k, k),
                         "missing_landmarks": names}
                        for k, names in a.unavailable.items()],
        "metrics": _metric_payload(a),
        "landmark_noise": {
            "frac_of_body_scale": a.noise.frac_of_body_scale,
            "provenance": a.noise.provenance,
        },
        "disclaimer": DISCLAIMER_ZH,
    }), 200


@api.route("/api/v1/health", methods=["GET"])
def health():
    """Model availability, so a deploy can be checked without a photo."""
    return jsonify({
        "ok": os.path.exists(L.DEFAULT_MODEL),
        "api_version": API_VERSION,
        "pose_model": os.path.basename(L.DEFAULT_MODEL),
        "pose_model_present": os.path.exists(L.DEFAULT_MODEL),
        "person_detector_present": os.path.exists(G._DETECTOR_MODEL),
    }), 200


if __name__ == "__main__":
    api.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", 5001)),
            debug=bool(os.environ.get("DEBUG")))
