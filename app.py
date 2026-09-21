#!/usr/bin/env python3
"""Flask single-page app: upload a full-body photo, get reference readings.

Presentation rules enforced here, not left to the template:
  * every reading is shown with its uncertainty, never bare
  * every verdict is shown with the provenance of the threshold behind it
  * a verdict whose error bar crosses a threshold is shown as undetermined
  * guard messages appear ABOVE the readings, not as a footnote
"""
from __future__ import annotations

import base64
import io
import os

from flask import Flask, render_template, request

from posture import assess
from posture import guards as G
from posture import landmarks as L
from posture.assess import DISCLAIMER_ZH
from posture.overlay import draw, draw_for_metric
from posture.thresholds import DIAGNOSTIC_ONLY

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024  # 24 MB

ALLOWED = {"png", "jpg", "jpeg", "webp", "bmp"}

PROVENANCE_ZH = {
    "guess": "凭经验设定，没有依据",
    "geometric-estimate": "由人体比例/投影几何推导",
    "literature-adjacent": "改编自相关文献，换算未经验证",
    "population-percentile": "由本仓库参考图像集的分位数得出",
}


def _b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, disclaimer=DISCLAIMER_ZH)


@app.route("/assess", methods=["POST"])
def do_assess():
    file = request.files.get("photo")
    if not file or not file.filename:
        return render_template("index.html", result=None, error="请先选择一张照片。",
                               disclaimer=DISCLAIMER_ZH)
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED:
        return render_template("index.html", result=None,
                               error=f"不支持的文件格式：.{ext}",
                               disclaimer=DISCLAIMER_ZH)

    try:
        rgb = L.load_rgb(io.BytesIO(file.read()))
    except Exception as exc:
        return render_template("index.html", result=None,
                               error=f"无法读取图片：{exc}", disclaimer=DISCLAIMER_ZH)

    a = assess(rgb)

    unstable: set[str] = set()
    for f in a.findings:
        if f.key == "unstable_landmarks":
            unstable |= set(f.detail.get("drift_frac_of_body_scale", {}))

    overlay = None
    per_metric_images: dict[str, str] = {}
    pose = None
    if a.view is not None:
        try:
            pose = L.detect(rgb)
        except L.LandmarkerUnavailable:
            pose = None
    if pose is not None:
        overlay = _b64(draw(rgb, pose, unstable=unstable))
        for key, m in a.measurements.items():
            if key in a.verdicts:
                per_metric_images[key] = _b64(draw_for_metric(rgb, pose, m, unstable))

    rows = []
    for key, m in a.measurements.items():
        v = a.verdicts.get(key)
        rows.append({
            "key": key,
            "label": m.label,
            "label_zh": v.spec.label_zh if v else key,
            "value": f"{m.value:+.1f}",
            "uncertainty": f"{m.uncertainty:.1f}",
            "plane": m.plane,
            "note": m.note,
            "diagnostic": key in DIAGNOSTIC_ONLY,
            "band": v.band if v else None,
            "band_text": v.display_band_zh if v else "",
            "resolved": v.resolved if v else None,
            "provenance": v.spec.provenance if v else "",
            "provenance_zh": PROVENANCE_ZH.get(v.spec.provenance, "") if v else "",
            "basis": v.spec.basis if v else "",
            "slight": v.spec.slight if v else None,
            "notable": v.spec.notable if v else None,
            "advice": v.advice_zh if v else [],
            "image": per_metric_images.get(key),
        })
    # Diagnostics last: they are context for the readings above, not verdicts.
    rows.sort(key=lambda r: (r["diagnostic"], r["plane"] != "sagittal"))

    result = {
        "ok": a.ok,
        "blocked": a.blocked,
        "error": a.error_zh,
        "view": a.view,
        "overlay": overlay,
        "rows": rows,
        "blocks": [f for f in a.findings if f.severity == G.SEVERITY_BLOCK],
        "warns": [f for f in a.findings if f.severity == G.SEVERITY_WARN],
        "noise": a.noise,
    }
    return render_template("index.html", result=result, disclaimer=DISCLAIMER_ZH)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", 5000)),
            debug=bool(os.environ.get("DEBUG")))
