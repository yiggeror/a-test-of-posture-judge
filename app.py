"""Posture assessment demo - Flask server.

Single page: upload one full-body photo (side view preferred), get the skeleton
overlay plus reference readings for four posture metrics.

This is a DEMO. Thresholds are demo defaults, not validated cut-offs - see
posture/thresholds.py. Output wording is deliberately "参考"/"倾向", never
diagnostic.
"""
from __future__ import annotations

import io
import os
import traceback

from flask import Flask, jsonify, render_template, request

from posture import engine, render
from posture.landmarks import NAMES
from posture.metrics import compute_all, compute_for_view
from posture.thresholds import Threshold

MAX_UPLOAD = 12 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD

VIEW_LABEL = {
    "front": "正面",
    "side": "侧面",
    "oblique": "斜侧（介于正面与侧面之间）",
    "unknown": "无法判定",
}


def _threshold_json(t: Threshold | None):
    if t is None:
        return None
    return {"mild": t.mild, "marked": t.marked, "unit": t.unit,
            "basis": t.basis, "note": t.note}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    try:
        path = engine.model_path("full")
        return jsonify({"ok": True, "model": os.path.basename(path),
                        "bytes": os.path.getsize(path)})
    except engine.ModelMissing as e:
        return jsonify({"ok": False, "error": str(e)}), 503


@app.post("/api/analyze")
def analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify({"ok": False, "error": "没有收到图片。"}), 400
    data = f.read()
    if not data:
        return jsonify({"ok": False, "error": "图片为空。"}), 400

    img = engine.read_image(data)
    if img is None:
        return jsonify({"ok": False,
                        "error": "无法解码该图片（支持 JPG / PNG / WebP）。"}), 400

    variant = request.form.get("variant", "full")
    if variant not in engine.VARIANTS:
        variant = "full"

    try:
        pts = engine.detect(img, variant)
    except engine.ModelMissing as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    except Exception:
        traceback.print_exc()
        return jsonify({"ok": False, "error": "推理失败，详见服务端日志。"}), 500

    if pts is None:
        # No pose -> no numbers. We never fall back to invented landmarks.
        return jsonify({
            "ok": False,
            "error": "未检测到人体。请使用全身入镜、光线充足、背景简单的照片。",
        }), 200

    view, ratio, metrics, plaus = compute_all(pts)
    # The view heuristic is only a heuristic, so the UI can override it.
    forced = request.form.get("view")
    if forced in ("front", "side") and plaus.ok:
        view = forced
        metrics = compute_for_view(pts, view)

    overlay = render.to_png_data_uri(render.draw(img, pts, view))
    h, w = img.shape[:2]

    return jsonify({
        "ok": True,
        "overlay": overlay,
        "image": {"width": w, "height": h},
        "plausible": plaus.ok,
        "plausibility": {
            "ok": plaus.ok,
            "torso_tilt_deg": None if plaus.torso_tilt_deg is None
                              else round(plaus.torso_tilt_deg, 1),
            "head_torso_ratio": None if plaus.head_torso_ratio is None
                                else round(plaus.head_torso_ratio, 3),
            "reasons": plaus.reasons,
        },
        "view": {"kind": view, "label": VIEW_LABEL[view], "ratio": round(ratio, 3),
                 "forced": forced in ("front", "side")},
        "landmarks": [
            {"i": i, "name": n, "x": round(p.x, 1), "y": round(p.y, 1),
             "visibility": round(p.visibility, 3)}
            for i, (n, p) in enumerate(zip(NAMES, pts))
        ],
        "metrics": [
            {"key": m.key, "label": m.label_zh, "label_en": m.label_en,
             "value": None if m.value is None else round(m.value, 2),
             "unit": m.unit, "band": m.band, "detail": m.detail,
             "advice": m.advice, "caveats": m.caveats,
             "threshold": _threshold_json(m.threshold)}
            for m in metrics
        ],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)),
            debug=False)
