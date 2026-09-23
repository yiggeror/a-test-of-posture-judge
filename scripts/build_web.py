#!/usr/bin/env python3
"""Build the web app, and prove the browser gives the same answers as Python.

Output (web/dist/):
    index.html        standalone page for any static host or a mini-program webview;
                      loads vision_wasm_internal.wasm and pose_heavy.N.bin
    artifact.html     the same page as a body fragment, for hosts that add their
                      own <html>/<head> skeleton and serve only web file types:
                      loads the model as base64 text parts, pose_heavy.N.b64.txt

The point of this script is the verification, not the packaging. web/posture.js
is a hand port of posture/, and a hand port drifts silently: a sign error in
one angle convention produces plausible numbers with the wrong meaning. So
every build runs both implementations over the same landmarks for every
committed test photograph and compares everything: view, blocking and warning
guards, withheld metrics, every reading and uncertainty, every band, and the
plain-language report a person reads. Any difference fails the build.

The Python side runs with the stability guard (the browser runs it too, on a
second detection) but without the independent person detector, which the
browser does not ship. That is the one deliberate difference.

The model is pose_landmarker_heavy, the one every threshold here was measured
with. scripts/compare_models.py measured the smaller `full` model as a
different instrument (readings differ by more than their own uncertainty on
about half the photos), so it is not a drop-in to save download size.
"""
from __future__ import annotations

import argparse
import base64
import glob
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture import guards as G  # noqa: E402
from posture import landmarks as L  # noqa: E402
from posture import view as V  # noqa: E402
from posture.assess import assess  # noqa: E402
from posture.metrics import NoiseModel  # noqa: E402
from posture.report import COPY, consumer_report  # noqa: E402
from posture.thresholds import DIAGNOSTIC_ONLY, load_thresholds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
DIST = os.path.join(WEB, "dist")

# Must equal the Python mediapipe version: landmarks were checked identical
# (median 3e-5 of body height) between the two at this version.
VISION_VERSION = "1.0.1"
WASM_URL = (f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{VISION_VERSION}"
            "/wasm/vision_wasm_internal.wasm")
MODEL = os.path.join(ROOT, "models", "pose_landmarker_heavy.task")
PART_BYTES = 14 * 1024 * 1024   # raw parts, kept under common 15 MB per-file limits
B64_PART_RAW = 3 * 3_495_253    # raw bytes per base64 part: a multiple of 3, so each
                                # part decodes on its own; ~14 MB of text each

# Home-page examples. All Pexels License. The first is also the hero figure.
SAMPLES = [
    ("testdata/pexels/side/side_pexels_30033728.jpg", "侧面照"),
    ("testdata/pexels/edge/edge_weight_one_leg_26125921.jpg", "正面照"),
    ("testdata/pexels/edge/edge_head_back_32084086.jpg", "侧面照"),
]
VERIFY_DIRS = ["pexels/side", "pexels/front", "pexels/edge", "pexels/negative",
               "side", "front", "negative"]


def constants(assets: dict) -> dict:
    n = NoiseModel.load()
    guard_keys = ("OUT_OF_FRAME_MARGIN", "TORSO_LEG_RATIO_MIN", "TORSO_LEG_RATIO_MAX",
                  "MAX_PLAUSIBLE_ANGLE", "MAX_KNEE_FLEXION", "MAX_TRUNK_LEAN",
                  "MAX_STANCE_WIDTH_RATIO", "MAX_WEIGHT_SHIFT_FRAC",
                  "MIN_WRIST_BELOW_HIP_FRAC", "MIN_KNEE_EXTENSION_FRONT_DEG",
                  "MIN_SUBJECT_HEIGHT_PX", "BYSTANDER_SIZE_RATIO",
                  "UNSTABLE_LANDMARK_FRAC", "MAX_WRIST_AHEAD_OF_HIP_FRAC")
    copy = {k: v for k, v in COPY.items() if not k.startswith("_")}
    return {
        "noise": {"overall": n.frac_of_body_scale,
                  "per_landmark": {k: v["rms_frac"] for k, v in n.detail.items()}},
        "thresholds": {k: {"slight": s.slight, "notable": s.notable,
                           "direction": s.direction}
                       for k, s in load_thresholds().items()},
        "diagnostic_only": sorted(DIAGNOSTIC_ONLY),
        "view": {"SIDE_MAX_SPREAD": V.SIDE_MAX_SPREAD,
                 "FRONT_MIN_SPREAD": V.FRONT_MIN_SPREAD},
        "guards": {k: getattr(G, k) for k in guard_keys},
        "copy": copy,
        "assets": assets,
    }


def detect_pair(rgb):
    """(pose, pose on the stability-perturbed image), as assess() sees them."""
    pose = L.detect(rgb)
    if pose is None:
        return None, None
    return pose, L.detect(G.stability_perturbation(rgb))


def js_input(pose, pose2, sc: float = 1.0) -> dict:
    return {
        "w": round(pose.width * sc), "h": round(pose.height * sc),
        "oscale": round(L.body_scale(pose), 2),
        "lm": [[round(l.x * sc, 3), round(l.y * sc, 3), round(l.visibility, 4)]
               for l in pose.landmarks],
        "lm2": (None if pose2 is None else
                [[round(l.x * sc, 3), round(l.y * sc, 3), round(l.visibility, 4)]
                 for l in pose2.landmarks]),
        "others": [[round(v * sc, 2) for v in b] for b in pose.other_pose_bboxes],
    }


def summarise_report(rep: dict) -> dict:
    """The parts of a report that carry meaning, in a comparable form."""
    return {
        "status": rep["status"],
        "retake": sorted(x["key"] for x in rep["retake"]),
        "issues": [(x["key"], x["level"], x.get("side"), x.get("direction"))
                   for x in rep["issues"]],
        "borderline": [(x["key"], x.get("side"), x.get("direction"))
                       for x in rep["borderline"]],
        "good": [x["key"] for x in rep["good"]],
        "not_measured": [(x["key"], x["reason"]) for x in rep["not_measured"]],
        "photo_tips": sorted(rep["photo_tips"]),
        "summaries": [x["summary"] for x in rep["issues"] + rep["borderline"]],
    }


def python_side(path: str):
    """Python's answer with the stability guard on and the object detector off."""
    real = G.check_is_person
    G.check_is_person = lambda *a, **k: []
    try:
        a = assess(L.load_rgb(path), deep_guards=True)
    finally:
        G.check_is_person = real
    return a


def verify(consts: dict) -> int:
    paths = sorted(p for d in VERIFY_DIRS
                   for p in glob.glob(os.path.join(ROOT, "testdata", d, "*.jpg")))
    cases, pys = {}, {}
    for p in paths:
        rgb = L.load_rgb(p)
        pose, pose2 = detect_pair(rgb)
        if pose is None:
            continue
        cid = os.path.relpath(p, ROOT)
        cases[cid] = js_input(pose, pose2)
        pys[cid] = python_side(p)

    with tempfile.TemporaryDirectory() as d:
        harness = os.path.join(d, "h.js")
        with open(harness, "w", encoding="utf-8") as fh:
            fh.write("const C=" + json.dumps(consts, ensure_ascii=False) + ";\n")
            fh.write(open(os.path.join(WEB, "posture.js"), encoding="utf-8").read())
            fh.write("\nconst CASES=" + json.dumps(cases) + ";\n")
            fh.write("""
const out={};
for(const id in CASES){const r=assess(CASES[id]),rep=report(r);
 const uniq=a=>[...new Set(a)].sort();
 out[id]={view:r.V.view,blocked:r.blocked,
  blocks:uniq(r.findings.filter(f=>f.sev==="block").map(f=>f.key)),
  warns:uniq(r.findings.filter(f=>f.sev==="warn").map(f=>f.key)),
  unavail:Object.keys(r.unavail).sort(),unstable:Object.keys(r.unstable).sort(),
  mets:Object.fromEntries(Object.entries(r.mets).map(([k,m])=>[k,[m.val,m.unc]])),
  verdicts:Object.fromEntries(Object.entries(r.verdicts).map(([k,v])=>[k,[v.band,v.resolved,v.floored]])),
  rep};}
console.log(JSON.stringify(out));
""")
        try:
            raw = subprocess.run(["node", harness], capture_output=True, text=True,
                                 check=True, timeout=300).stdout
        except FileNotFoundError:
            print("node not found; cannot verify the JS port", file=sys.stderr)
            return -1
        except subprocess.CalledProcessError as exc:
            print("node failed:\n" + exc.stderr[:3000], file=sys.stderr)
            return -1
    js = json.loads(raw)

    bad = 0
    status = {"ok": 0, "retake": 0}
    for cid, a in pys.items():
        j, d = js[cid], []
        if a.view.view != j["view"]:
            d.append(f"view {a.view.view}!={j['view']}")
        if a.blocked != j["blocked"]:
            d.append(f"blocked {a.blocked}!={j['blocked']}")
        for sev, key in (("block", "blocks"), ("warn", "warns")):
            pk = sorted({f.key for f in a.findings if f.severity == sev})
            if pk != j[key]:
                d.append(f"{key} {pk}!={j[key]}")
        if sorted(a.unavailable) != j["unavail"]:
            d.append(f"unavail {sorted(a.unavailable)}!={j['unavail']}")
        if sorted(a.unstable) != j["unstable"]:
            d.append(f"unstable {sorted(a.unstable)}!={j['unstable']}")
        for k, m in a.measurements.items():
            jm = j["mets"].get(k)
            if jm is None:
                d.append(f"missing {k}")
            elif abs(m.value - jm[0]) > 0.02 or abs(m.uncertainty - jm[1]) > 0.02:
                d.append(f"{k} {m.value:.3f}±{m.uncertainty:.3f} vs {jm}")
        for k, v in a.verdicts.items():
            jb = j["verdicts"].get(k)
            if jb is None or (v.band, v.resolved, v.below_noise_floor) != tuple(jb):
                d.append(f"verdict {k}")
        pr = summarise_report(consumer_report(a))
        jr = summarise_report(j["rep"])
        jr = json.loads(json.dumps(jr))   # tuples -> lists on both sides
        pr = json.loads(json.dumps(pr))
        if pr != jr:
            diff = [k for k in pr if pr[k] != jr[k]]
            d.append(f"report differs in {diff}")
        status[pr["status"]] = status.get(pr["status"], 0) + 1
        if d:
            bad += len(d)
            print(f"  {cid[:48]:<50} {'; '.join(d)[:160]}")
    print(f"  {len(pys)} photos compared, {bad} differences "
          f"({status['ok']} give a result, {status['retake']} ask for a retake)")
    return bad


def build_samples() -> list[dict]:
    from PIL import Image
    out = []
    for rel, label in SAMPLES:
        path = os.path.join(ROOT, rel)
        rgb = L.load_rgb(path)
        pose, pose2 = detect_pair(rgb)
        if pose is None:
            raise SystemExit(f"sample has no pose: {rel}")
        im = Image.fromarray(rgb)
        sc = min(1.0, 1000 / im.height)
        big = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
        small = im.resize((round(im.width * 320 / im.height), 320), Image.LANCZOS)
        enc = lambda x, q: base64.b64encode(   # noqa: E731
            (lambda b: (x.save(b, "JPEG", quality=q, optimize=True, progressive=True), b.getvalue())[1])(io.BytesIO())).decode()
        s = js_input(pose, pose2, sc)
        s.update({"id": os.path.basename(rel).rsplit(".", 1)[0], "label": label,
                  "jpg": enc(big, 80), "thumb": enc(small, 72)})
        out.append(s)
    return out


def build_assets() -> tuple[dict, dict]:
    """Write the runtime and the model next to the page, in both encodings."""
    os.makedirs(DIST, exist_ok=True)
    wasm = os.path.join(DIST, "vision_wasm_internal.wasm")
    if not os.path.exists(wasm):
        print(f"fetching {WASM_URL}")
        with urllib.request.urlopen(WASM_URL, timeout=120) as r, open(wasm, "wb") as fh:
            fh.write(r.read())
    for old in glob.glob(os.path.join(DIST, "pose_heavy.*")):
        os.remove(old)
    data = open(MODEL, "rb").read()
    wasm_bytes = os.path.getsize(wasm)

    raw = []
    for i in range(0, len(data), PART_BYTES):
        name = f"pose_heavy.{len(raw) + 1}.bin"
        with open(os.path.join(DIST, name), "wb") as fh:
            fh.write(data[i:i + PART_BYTES])
        raw.append(name)

    b64, b64_bytes = [], 0
    for i in range(0, len(data), B64_PART_RAW):
        name = f"pose_heavy.{len(b64) + 1}.b64.txt"
        enc = base64.b64encode(data[i:i + B64_PART_RAW])
        with open(os.path.join(DIST, name), "wb") as fh:
            fh.write(enc)
        b64.append(name)
        b64_bytes += len(enc)

    binary = {"wasm": os.path.basename(wasm), "model": raw, "encoding": "binary",
              "bytes": wasm_bytes + len(data)}
    text = {"wasm": os.path.basename(wasm), "model": b64, "encoding": "base64",
            "bytes": wasm_bytes + b64_bytes}
    return binary, text


def page(consts: dict, samples: list[dict]) -> str:
    src = os.path.join(WEB, "src")
    read = lambda p: open(p, encoding="utf-8").read()   # noqa: E731
    return "\n".join([
        "<title>挺拔体态自测</title>",
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap">',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&text=%E6%8C%BA%E6%8B%94&display=swap">',
        "<style>\n" + read(os.path.join(src, "app.css")) + "</style>",
        read(os.path.join(src, "app.html")),
        f'<script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{VISION_VERSION}/vision_bundle.js" crossorigin="anonymous"></script>',
        "<script>\nconst C=" + json.dumps(consts, ensure_ascii=False, separators=(",", ":"))
        + ";\nconst SAMPLES=" + json.dumps(samples, separators=(",", ":")) + ";\n</script>",
        "<script>\n" + read(os.path.join(WEB, "posture.js")) + "\n</script>",
        "<script>\n" + read(os.path.join(WEB, "engine.js")) + "\n</script>",
        "<script>\n" + read(os.path.join(src, "moves.js")) + "\n</script>",
        "<script>\n" + read(os.path.join(src, "app.js")) + "\n</script>",
    ])


STANDALONE_HEAD = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>html,body{margin:0;height:100%}img{max-width:100%}[hidden]{display:none!important}
:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}</style>
</head>
<body>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    py_mp = importlib.metadata.version("mediapipe")
    if py_mp != VISION_VERSION:
        print(f"mediapipe {py_mp} in Python but {VISION_VERSION} in the browser; "
              "landmark parity was only established for equal versions", file=sys.stderr)
        return 1

    assets, text_assets = build_assets()
    consts = constants(assets)
    if not args.skip_verify:
        print("verifying web/posture.js against posture/ on every test photo:")
        bad = verify(consts)
        if bad != 0:
            print(f"\nFAIL: {bad} differences between web/posture.js and posture/",
                  file=sys.stderr)
            return 1
    samples = build_samples()
    with open(os.path.join(DIST, "artifact.html"), "w", encoding="utf-8") as fh:
        fh.write(page({**consts, "assets": text_assets}, samples))
    with open(os.path.join(DIST, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(STANDALONE_HEAD + page(consts, samples) + "\n</body>\n</html>\n")
    kb = os.path.getsize(os.path.join(DIST, "index.html")) // 1024
    print(f"\nwritten web/dist/index.html ({kb} KB) + artifact.html; "
          f"assets {assets['bytes'] / 1e6:.1f} MB binary / "
          f"{text_assets['bytes'] / 1e6:.1f} MB with base64 model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
