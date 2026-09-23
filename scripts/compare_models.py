#!/usr/bin/env python3
"""Can a smaller pose model stand in for the one the thresholds were measured on?

Every threshold and uncertainty in this repo was measured with
pose_landmarker_heavy (30.7 MB). A browser page or a mini-program would rather
ship pose_landmarker_full (9.4 MB). This script answers the question with the
data instead of assuming it: run both models over the same photographs,
compute every metric from each, and report how far the readings disagree,
next to the uncertainty the heavy model's readings already carry.

If the disagreement is small against that uncertainty, the swap is cheap. If it
is comparable, the smaller model is a different instrument and would need its
own calibration (repeatability.py + reference_distribution.py re-run on it).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture import landmarks as L  # noqa: E402
from posture.metrics import NoiseModel, compute_all  # noqa: E402
from posture.view import estimate_view  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(dirs: list[str], a: str, b: str) -> dict:
    noise = NoiseModel.load()
    paths = sorted(p for d in dirs for p in glob.glob(os.path.join(d, "*.jpg")))
    per_metric: dict[str, list[tuple[float, float]]] = {}
    view_flips = 0
    compared = 0
    for p in paths:
        rgb = L.load_rgb(p)
        pa = L.detect(rgb, model_path=a)
        pb = L.detect(rgb, model_path=b)
        if pa is None or pb is None:
            continue
        va, vb = estimate_view(pa), estimate_view(pb)
        compared += 1
        if va.view != vb.view:
            view_flips += 1
            continue
        ma, mb = compute_all(pa, va, noise), compute_all(pb, vb, noise)
        for k in ma:
            if k in mb:
                per_metric.setdefault(k, []).append(
                    (abs(ma[k].value - mb[k].value), ma[k].uncertainty))

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")

    out = {"n_images": compared, "view_disagreements": view_flips, "metrics": {}}
    for k, rows in sorted(per_metric.items()):
        diffs = [d for d, _ in rows]
        unc = [u for _, u in rows]
        out["metrics"][k] = {
            "n": len(rows),
            "median_abs_diff_deg": round(pct(diffs, 0.5), 2),
            "p90_abs_diff_deg": round(pct(diffs, 0.9), 2),
            "median_uncertainty_deg": round(pct(unc, 0.5), 2),
            # share of readings where the two models differ by more than the
            # reading's own 1-sigma uncertainty
            "frac_diff_gt_1sigma": round(sum(d > u for d, u in rows) / len(rows), 3),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=[
        os.path.join(ROOT, "testdata", d) for d in
        ("side", "front", "pexels/side", "pexels/front", "pexels/edge")])
    ap.add_argument("--reference", default=os.path.join(ROOT, "models", "pose_landmarker_heavy.task"))
    ap.add_argument("--candidate", default=os.path.join(ROOT, "models", "pose_landmarker_full.task"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "model_swap.json"))
    args = ap.parse_args(argv)
    res = run(args.dirs, args.reference, args.candidate)
    res["reference"] = os.path.basename(args.reference)
    res["candidate"] = os.path.basename(args.candidate)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
