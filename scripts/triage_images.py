#!/usr/bin/env python3
"""Grade candidate photos BEFORE they are accepted as test data.

Point it at a folder of candidates. For each image it says ACCEPT or REJECT
with a concrete reason, tags the usable ones by view, and prints a tally
against the collection targets. Anything it rejects is not worth keeping.

    python scripts/triage_images.py --dir inbox --manifest inbox/manifest.json
    python scripts/triage_images.py --dir inbox --move-accepted testdata

Exit code is 0 if every target is met, 1 otherwise - so a collecting agent can
loop until it passes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture import engine                                   # noqa: E402
from posture.landmarks import (LEFT_ANKLE, LEFT_EAR, LEFT_HIP,   # noqa: E402
                               LEFT_SHOULDER, RIGHT_ANKLE, RIGHT_EAR,
                               RIGHT_HIP, RIGHT_SHOULDER)
from posture.metrics import (check_plausibility, compute_for_view,  # noqa: E402
                             detect_view)

# Collection targets. See reports/IMAGE_BRIEF.md for the reasoning.
TARGETS = {"side": 20, "front": 15, "negative": 10}

MIN_SHORT_SIDE = 480          # below this the landmarks get mushy
MIN_BODY_FRACTION = 0.55      # head-to-ankle must span this much of the frame


def grade(path: str) -> dict:
    """Return a verdict dict for one image. Never guesses - only reports."""
    out = {"file": os.path.basename(path), "accept": False,
           "category": None, "reasons": []}
    raw = open(path, "rb").read()
    img = engine.read_image(raw)
    if img is None:
        out["reasons"].append("无法解码（不是有效图片？）")
        return out
    h, w = img.shape[:2]
    out["size"] = f"{w}x{h}"
    if min(h, w) < MIN_SHORT_SIDE:
        out["reasons"].append(f"分辨率过低：短边 {min(h, w)}px < {MIN_SHORT_SIDE}px")
        return out

    det = engine.detect(img)
    if det is None:
        # A no-pose image is a perfectly good NEGATIVE sample.
        out.update(accept=True, category="negative")
        out["reasons"].append("未检测到人体 —— 作为负样本收下")
        return out

    pts, world = det.pts, det.world
    plaus = check_plausibility(pts, (w, h), det.n_poses)
    out["torso_tilt_deg"] = round(plaus.torso_tilt_deg or 0.0, 1)
    out["head_torso_ratio"] = round(plaus.head_torso_ratio or 0.0, 3)
    if not plaus.ok:
        out.update(accept=True, category="negative")
        out["reasons"].append("合理性检查拒绝 —— 作为负样本收下："
                              + "；".join(plaus.reasons))
        return out

    view, cue = detect_view(pts, world)
    out["view"] = view
    out["view_cue"] = round(cue, 2)
    out["view_cue_kind"] = "yaw_deg" if world else "shoulder_torso_ratio"

    vis = {"ear": max(pts[LEFT_EAR].visibility, pts[RIGHT_EAR].visibility),
           "shoulder": max(pts[LEFT_SHOULDER].visibility,
                           pts[RIGHT_SHOULDER].visibility),
           "hip": max(pts[LEFT_HIP].visibility, pts[RIGHT_HIP].visibility),
           "ankle": max(pts[LEFT_ANKLE].visibility, pts[RIGHT_ANKLE].visibility)}
    out["visibility"] = {k: round(v, 3) for k, v in vis.items()}

    # Hard requirement: the whole body must be in frame. Visibility alone does
    # not establish this - MediaPipe extrapolates landmarks past the border and
    # still scores them plausibly - so check the coordinates too.
    best_ankle = max((pts[LEFT_ANKLE], pts[RIGHT_ANKLE]),
                     key=lambda p: p.visibility)
    if not (0 <= best_ankle.x <= w and 0 <= best_ankle.y <= h):
        out["reasons"].append(
            f"踝部关键点在画面外（y={best_ankle.y:.0f}，图高 {h}）—— "
            "模型外推值，脚底没有完整入镜")
        return out
    if vis["ankle"] < 0.5:
        out["reasons"].append(f"踝部不可见（vis={vis['ankle']:.2f}）—— 不是全身照")
        return out
    if vis["hip"] < 0.5:
        out["reasons"].append(f"髋部不可见（vis={vis['hip']:.2f}）")
        return out
    # The ear drives both sagittal metrics; a hat/hair/mask kills them.
    if vis["ear"] < 0.7:
        out["reasons"].append(f"耳部可见度不足（vis={vis['ear']:.2f}）—— "
                              "两项矢状面指标依赖耳点")
        return out

    top = min(pts[LEFT_EAR].y, pts[RIGHT_EAR].y)
    bottom = max(pts[LEFT_ANKLE].y, pts[RIGHT_ANKLE].y)
    frac = (bottom - top) / h
    out["body_fraction"] = round(frac, 3)
    if frac < MIN_BODY_FRACTION:
        out["reasons"].append(f"人体只占画面 {frac:.0%}（需 ≥{MIN_BODY_FRACTION:.0%}）"
                              " —— 人太小，关键点精度差")
        return out

    if view == "oblique":
        cue_txt = (f"躯干偏航 {cue:.0f}°" if world else f"肩宽/躯干={cue:.2f}")
        out["reasons"].append(f"斜侧视角（{cue_txt}）—— "
                              "正面与侧面指标都会被透视污染，请重拍正侧面")
        return out

    computed = [m.key for m in compute_for_view(pts, view, (w, h))
                if m.value is not None]
    out["metrics_computed"] = computed
    if not computed:
        out["reasons"].append("该视角下无任何指标可算")
        return out

    out.update(accept=True, category=view)
    out["reasons"].append(f"合格 {view} 样本；可算指标：{', '.join(computed)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="inbox", help="candidate image folder")
    ap.add_argument("--manifest", help="write the full verdict JSON here")
    ap.add_argument("--move-accepted", metavar="DEST",
                    help="copy accepted images into DEST/<category>/")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    if not files:
        print(f"{a.dir}/ 里没有图片。")
        return 1

    rows, tally = [], {k: 0 for k in TARGETS}
    for f in files:
        r = grade(os.path.join(a.dir, f))
        rows.append(r)
        mark = "ACCEPT" if r["accept"] else "REJECT"
        cat = f"[{r['category']}]" if r["category"] else ""
        print(f"{mark:<7}{r['file']:<40}{cat:<11}{'; '.join(r['reasons'])}")
        if r["accept"]:
            tally[r["category"]] = tally.get(r["category"], 0) + 1
            if a.move_accepted:
                dest = os.path.join(a.move_accepted, r["category"])
                os.makedirs(dest, exist_ok=True)
                shutil.copy2(os.path.join(a.dir, f), os.path.join(dest, f))

    print("\n" + "=" * 72)
    ok = True
    for cat, want in TARGETS.items():
        got = tally.get(cat, 0)
        hit = got >= want
        ok &= hit
        print(f"  {cat:<10} {got:>3} / {want:<3} {'OK' if hit else 'SHORT'}")
    print("=" * 72)
    print(f"共 {len(files)} 张，合格 {sum(tally.values())} 张。")
    if not ok:
        print("目标未达成 —— 请按 reports/IMAGE_BRIEF.md 继续补图。")

    if a.manifest:
        with open(a.manifest, "w", encoding="utf-8") as fh:
            json.dump({"targets": TARGETS, "tally": tally, "images": rows},
                      fh, ensure_ascii=False, indent=2)
        print(f"清单已写入 {a.manifest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
