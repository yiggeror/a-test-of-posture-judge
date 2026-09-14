#!/usr/bin/env python3
"""Phase 2 validation harness.

Runs the real pipeline over every image in testdata/ and prints exactly what
came out, including failures. Nothing here is smoothed over: images where no
pose is found, or where a metric cannot be computed, are printed as such.

Usage: python scripts/validate.py [--variant full] [--save-overlays]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture import engine, render                       # noqa: E402
from posture.landmarks import NAMES                      # noqa: E402
from posture.metrics import compute_all                  # noqa: E402

KEY_LM = ["left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_hip", "right_hip", "left_ankle", "right_ankle"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=list(engine.VARIANTS))
    ap.add_argument("--dir", default="testdata")
    ap.add_argument("--save-overlays", action="store_true")
    a = ap.parse_args()

    # Recurse, so the testdata/{side,front,negative}/ layout that
    # scripts/triage_images.py --move-accepted produces is picked up too.
    files = sorted(
        os.path.join(root, f)
        for root, _, names in os.walk(a.dir)
        for f in names
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    if not files:
        print(f"no images in {a.dir}/ - run scripts/fetch_testdata.py")
        return 1

    print(f"model variant: {a.variant}   images: {len(files)}\n")
    detected = failed = 0

    for f in files:
        name = os.path.relpath(f, a.dir)
        print("=" * 78)
        print(name)
        print("=" * 78)
        img = cv2.imread(f)
        if img is None:
            print("  DECODE FAILED\n"); failed += 1; continue
        img = engine.read_image(open(f, "rb").read())
        h, w = img.shape[:2]
        pts = engine.detect(img, a.variant)
        if pts is None:
            print(f"  {w}x{h}  NO POSE DETECTED  <-- pipeline failure\n")
            failed += 1
            continue
        detected += 1
        view, ratio, metrics, plaus = compute_all(pts)
        print(f"  image {w}x{h}   view={view} (shoulder/torso={ratio:.3f})")
        print(f"  plausibility: {'PASS' if plaus.ok else 'REJECTED'}  "
              f"torso_tilt={plaus.torso_tilt_deg:.1f}deg  "
              f"head/torso={plaus.head_torso_ratio:.2f}")
        for r in plaus.reasons:
            print(f"      ! {r}")
        print("  key landmark visibility:")
        for k in KEY_LM:
            p = pts[NAMES.index(k)]
            mark = "   " if p.visibility >= 0.5 else " ! "
            print(f"   {mark}{k:<16} ({p.x:7.1f},{p.y:7.1f})  vis={p.visibility:.3f}")
        print("  metrics:")
        for m in metrics:
            if m.value is None:
                print(f"    - {m.label_en:<28} UNAVAILABLE  ({m.detail})")
            else:
                unit = "deg" if m.unit == "deg" else "xtorso"
                print(f"    - {m.label_en:<28} {m.value:+8.2f} {unit:<7} "
                      f"[{m.band}]")
                if m.detail:
                    print(f"        {m.detail}")
        if a.save_overlays:
            os.makedirs("reports/overlays", exist_ok=True)
            out = os.path.join("reports/overlays",
                               os.path.splitext(name)[0].replace(os.sep, "_")
                               + "_overlay.png")
            cv2.imwrite(out, render.draw(img, pts, view))
            print(f"  overlay -> {out}")
        print()

    print("=" * 78)
    print(f"pose detected: {detected}/{len(files)}   no pose: {failed}/{len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
