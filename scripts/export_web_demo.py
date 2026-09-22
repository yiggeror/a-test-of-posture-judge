#!/usr/bin/env python3
"""Build the browser demo, and verify the JS port matches Python exactly.

The point of this script is the verification, not the packaging. web/posture.js
is a hand port of the posture/ package, and a hand port drifts silently: a sign
error in one angle convention produces plausible numbers with the wrong
meaning, which is precisely the failure this project exists to remove.

So every build re-runs both implementations over the same landmarks and
compares view, blocking, warnings, withheld metrics, every reading, every
uncertainty and every verdict band. A non-zero diff fails the build.

Comparison runs with deep_guards=False: the browser has no independent person
detector and no landmark-stability check, both of which need a second model
pass. That difference is deliberate and is stated on the demo page itself.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture import guards as G  # noqa: E402
from posture import landmarks as L  # noqa: E402
from posture.assess import ADVICE_ZH, DISCLAIMER_ZH, assess  # noqa: E402
from posture.metrics import NoiseModel  # noqa: E402
from posture.thresholds import DIAGNOSTIC_ONLY, load_thresholds  # noqa: E402
from posture import view as V  # noqa: E402

SAMPLES = [
    ("testdata/pexels/side/side_pexels_30033728.jpg", "侧面·合格", "工作室侧面照，全身在画面内"),
    ("testdata/pexels/front/front_pexels_26728096.jpg", "正面·合格", "工作室正面照，双臂自然垂放"),
    ("testdata/pexels/side/side_feetcropped_13802944.jpg", "侧面·脚出画", "小腿以下被裁切"),
    ("testdata/pexels/edge/edge_rotated53_16135927.jpg", "斜侧·应拒绝", "躯干偏航约 53°"),
    ("testdata/pexels/edge/edge_body_lean_32192287.jpg", "前倾·应拒绝", "非中立站姿"),
    ("testdata/pexels/negative/neg_statue_pexels_17135448.jpg", "雕塑·已知无解",
     "几何手段无法与真人区分"),
]
MAX_H = 760
JPEG_Q = 74


def constants() -> dict:
    n = NoiseModel.load()
    return {
        "noise": {"overall": n.frac_of_body_scale, "provenance": n.provenance,
                  "per_landmark": {k: v["rms_frac"] for k, v in n.detail.items()}},
        "thresholds": {k: {"label": s.label_zh, "slight": s.slight,
                           "notable": s.notable, "direction": s.direction,
                           "provenance": s.provenance, "n": s.n}
                       for k, s in load_thresholds().items()},
        "diagnostic_only": sorted(DIAGNOSTIC_ONLY),
        "advice": ADVICE_ZH,
        "disclaimer": DISCLAIMER_ZH,
        "view": {"SIDE_MAX_SPREAD": V.SIDE_MAX_SPREAD,
                 "FRONT_MIN_SPREAD": V.FRONT_MIN_SPREAD},
        "guards": {k: getattr(G, k) for k in (
            "OUT_OF_FRAME_MARGIN", "TORSO_LEG_RATIO_MIN", "TORSO_LEG_RATIO_MAX",
            "MAX_PLAUSIBLE_ANGLE", "MAX_KNEE_FLEXION", "MAX_TRUNK_LEAN",
            "MAX_STANCE_WIDTH_RATIO", "MAX_WEIGHT_SHIFT_FRAC",
            "MIN_WRIST_BELOW_HIP_FRAC", "MIN_KNEE_EXTENSION_FRONT_DEG",
            "MIN_SUBJECT_HEIGHT_PX", "BYSTANDER_SIZE_RATIO")},
    }


def build_samples(root: str) -> list[dict]:
    from PIL import Image
    out = []
    for rel, title, note in SAMPLES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"[skip] missing {rel}", file=sys.stderr)
            continue
        rgb = L.load_rgb(path)
        pose = L.detect(rgb)
        if pose is None:
            print(f"[skip] no pose in {rel}", file=sys.stderr)
            continue
        # The original-resolution body scale is kept because one guard
        # (subject_too_small) is in absolute pixels: every other quantity is
        # normalised, so downscaling for payload size would flip only that one.
        oscale = L.body_scale(pose)
        im = Image.open(path).convert("RGB")
        w0, h0 = im.size
        sc = min(1.0, MAX_H / h0)
        im2 = im.resize((round(w0 * sc), round(h0 * sc)), Image.LANCZOS)
        buf = io.BytesIO()
        im2.save(buf, "JPEG", quality=JPEG_Q, optimize=True)
        out.append({
            "id": os.path.basename(path).rsplit(".", 1)[0],
            "title": title, "note": note,
            "w": im2.size[0], "h": im2.size[1],
            "sc": round(sc, 5), "oscale": round(oscale, 1),
            "jpg": base64.b64encode(buf.getvalue()).decode(),
            "lm": [[round(l.x * sc, 2), round(l.y * sc, 2), round(l.visibility, 3)]
                   for l in pose.landmarks],
            "nPoses": pose.n_poses_detected,
            "others": [[round(v, 1) for v in
                        (b[0] * sc, b[1] * sc, b[2] * sc, b[3] * sc)]
                       for b in pose.other_pose_bboxes],
        })
    return out


def verify(root: str, samples: list[dict], consts: dict) -> int:
    """Run both implementations and diff. Returns the number of mismatches."""
    js_logic = os.path.join(root, "web", "posture.js")
    with tempfile.TemporaryDirectory() as d:
        harness = os.path.join(d, "h.js")
        with open(harness, "w") as fh:
            fh.write("const SAMPLES=" + json.dumps(samples) + ";\n")
            fh.write("const C=" + json.dumps(consts, ensure_ascii=False) + ";\n")
            fh.write(open(js_logic).read())
            fh.write("""
const out={};
for(const S of SAMPLES){const r=assess(S);
 out[S.id]={view:r.V.view,blocked:r.blocked,
  blocks:r.findings.filter(f=>f.sev==="block").map(f=>f.key).sort(),
  warns:r.findings.filter(f=>f.sev==="warn").map(f=>f.key).sort(),
  unavail:Object.keys(r.unavail).sort(),
  mets:Object.fromEntries(Object.entries(r.mets).map(([k,m])=>[k,[+m.val.toFixed(2),+m.unc.toFixed(2)]])),
  verdicts:Object.fromEntries(Object.entries(r.verdicts).map(([k,v])=>[k,[v.band,v.resolved,v.floored]]))};}
console.log(JSON.stringify(out));
""")
        try:
            raw = subprocess.run(["node", harness], capture_output=True,
                                 text=True, check=True, timeout=120).stdout
        except FileNotFoundError:
            print("node not found; cannot verify the JS port", file=sys.stderr)
            return -1
        except subprocess.CalledProcessError as exc:
            print("node failed:\n" + exc.stderr[:2000], file=sys.stderr)
            return -1
    js = json.loads(raw)

    bad = 0
    for (rel, _, _), s in zip(SAMPLES, samples):
        a = assess(L.load_rgb(os.path.join(root, rel)), deep_guards=False)
        j = js[s["id"]]
        d = []
        if a.view.view != j["view"]:
            d.append(f"view {a.view.view}!={j['view']}")
        if a.blocked != j["blocked"]:
            d.append(f"blocked {a.blocked}!={j['blocked']}")
        pb = sorted(f.key for f in a.findings if f.severity == "block")
        if pb != j["blocks"]:
            d.append(f"blocks {pb}!={j['blocks']}")
        pw = sorted(f.key for f in a.findings if f.severity == "warn")
        if pw != j["warns"]:
            d.append(f"warns {pw}!={j['warns']}")
        if sorted(a.unavailable) != j["unavail"]:
            d.append(f"unavail {sorted(a.unavailable)}!={j['unavail']}")
        for k, m in a.measurements.items():
            jm = j["mets"].get(k)
            if jm is None:
                d.append(f"missing {k}")
            elif abs(m.value - jm[0]) > 0.06 or abs(m.uncertainty - jm[1]) > 0.06:
                d.append(f"{k} {m.value:.2f}±{m.uncertainty:.2f} vs {jm}")
        for k, v in a.verdicts.items():
            jb = j["verdicts"].get(k)
            if not jb:
                d.append(f"verdict missing {k}")
            elif (v.band, v.resolved, v.below_noise_floor) != tuple(jb):
                d.append(f"{k} verdict mismatch")
        bad += len(d)
        print(f"  {s['id'][:32]:<34} {'OK' if not d else '; '.join(d)[:90]}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="web/demo.html")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    consts = constants()
    samples = build_samples(root)
    if not samples:
        print("no samples built", file=sys.stderr)
        return 1

    if not args.skip_verify:
        print("verifying JS port against Python:")
        bad = verify(root, samples, consts)
        if bad > 0:
            print(f"\nFAIL: {bad} mismatches between web/posture.js and posture/",
                  file=sys.stderr)
            return 1
        if bad == 0:
            print("  0 mismatches\n")

    w = os.path.join(root, "web")
    parts = [open(os.path.join(w, f)).read() for f in
             ("demo-styles.html", "demo-gauge.css", "demo-body.html")]
    parts.append("<script>\nconst SAMPLES=" + json.dumps(samples, separators=(",", ":"))
                 + ";\nconst C=" + json.dumps(consts, ensure_ascii=False,
                                              separators=(",", ":")) + ";\n</script>")
    parts.append("<script>\n" + open(os.path.join(w, "posture.js")).read()
                 + open(os.path.join(w, "demo-render.js")).read())

    dest = os.path.join(root, args.out)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w") as fh:
        fh.write("\n".join(parts))
    print(f"written {dest} ({os.path.getsize(dest)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
