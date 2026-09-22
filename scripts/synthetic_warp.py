#!/usr/bin/env python3
"""Measure the tool's response to a KNOWN change in the subject's posture.

Why this is different from the rotation test
-------------------------------------------
`repeatability.py` rotates the image. That changes where the camera thinks
"down" is, and every angle-from-vertical shifts by exactly the rotation. It is
a strong precision test, but it cannot detect the failure that matters most
for a posture tool:

    if a person's head really is 10 degrees further forward,
    does the reading move by 10 degrees, or by 6?

Rotation cannot answer that, because it moves the reference axis rather than
the body. A systematic under-response to real postural deviation would pass
the rotation test perfectly and still make every verdict too lenient.

This script answers it by deforming the photograph itself. A horizontal shear
is applied whose magnitude ramps from zero at the hips to `d` pixels at the
ears, which is geometrically what forward head carriage looks like: the pelvis
stays put, the neck and head translate forward, the shoulders come partway.

The ground truth is exact and needs no labelling. Every landmark's true new
position is `(x + d * ramp(y), y)`, so the true new value of every metric can
be computed directly from the baseline landmarks, and compared against what
the tool measures after re-detecting the warped image.

    measured delta / true delta  ->  should be 1.000

A slope below 1 means the tool under-reports real postural deviation. Above 1
means it exaggerates it.

What this does NOT establish
---------------------------
This is accuracy with respect to a DELTA, not an absolute. It shows whether a
known change of N degrees reads as N degrees; it says nothing about whether
the baseline photograph's absolute reading was right, which would still need a
reference measurement on a real body.

It also only deforms real photographs, so it inherits their framing. The warp
is a plausible-looking approximation of forward head carriage, not a
biomechanical simulation: real forward head posture also changes the cervical
curve and the position of the chin relative to the skull, which a shear does
not reproduce.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture import landmarks as L  # noqa: E402
from posture.metrics import NoiseModel, compute_all  # noqa: E402
from posture.view import estimate_view  # noqa: E402

# Shear magnitudes at the ears, as a fraction of body scale. +-0.10 of a
# ~1.4 m body is about +-14 cm of head translation, which spans the range
# from unremarkable to pronounced forward head carriage.
SHIFT_FRACTIONS = (-0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08, 0.10)

# Metrics whose true value this warp changes in a computable way. Everything
# in compute_all() is recomputed from warped landmark positions, so this list
# exists only to order the report.
REPORT_ORDER = ("head_over_hip", "forward_head", "shoulder_protraction",
                "trunk_sway", "knee_deviation", "shoulder_tilt", "pelvis_tilt",
                "head_tilt", "lateral_head_shift", "head_vs_shoulder_tilt")


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ramp_for(y: np.ndarray, y_hip: float, y_ear: float) -> np.ndarray:
    """Shear weight at each row: 0 at the hips, 1 at the ears, smooth between.

    Above the ears the weight stays 1 so the whole head translates rigidly;
    below the hips it stays 0 so the legs and the floor do not move.
    """
    span = y_hip - y_ear
    if abs(span) < 1e-6:
        return np.zeros_like(y, dtype=float)
    return _smoothstep((y_hip - y) / span)


def warp(rgb: np.ndarray, shift_px: float, y_hip: float, y_ear: float
         ) -> np.ndarray:
    """Horizontal shear with a hip-to-ear ramp, bilinear resampled.

    Implemented as an inverse map (for each output pixel, sample the source)
    so the result has no holes. numpy only -- scipy is not a dependency.
    """
    h, w = rgb.shape[:2]
    ys = np.arange(h, dtype=float)
    offs = shift_px * ramp_for(ys, y_hip, y_ear)          # per-row offset

    xs = np.arange(w, dtype=float)
    src_x = xs[None, :] - offs[:, None]                    # (h, w)
    src_x = np.clip(src_x, 0, w - 1)

    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    frac = (src_x - x0)[..., None]

    rows = np.arange(h)[:, None]
    left = rgb[rows, x0].astype(np.float32)
    right = rgb[rows, x1].astype(np.float32)
    return np.clip(left * (1 - frac) + right * frac, 0, 255).astype(np.uint8)


def warped_pose(pose: L.PoseResult, shift_px: float, y_hip: float,
                y_ear: float) -> L.PoseResult:
    """The pose the warped image SHOULD have. This is the ground truth."""
    lms = []
    for lm in pose.landmarks:
        r = float(ramp_for(np.array([lm.y]), y_hip, y_ear)[0])
        lms.append(L.Landmark(x=lm.x + shift_px * r, y=lm.y, z=lm.z,
                              visibility=lm.visibility, presence=lm.presence))
    return L.PoseResult(landmarks=lms, width=pose.width, height=pose.height,
                        n_poses_detected=1)


def _fit(xs: list[float], ys: list[float]) -> dict:
    """Slope through the origin, plus robust slope and residual spread."""
    if not xs:
        return {}
    sxx = sum(x * x for x in xs)
    ls = sum(x * y for x, y in zip(xs, ys)) / sxx if sxx > 0 else float("nan")
    ratios = sorted(y / x for x, y in zip(xs, ys) if abs(x) > 1e-9)
    robust = (ratios[len(ratios) // 2] if ratios else float("nan"))
    resid = [y - x for x, y in zip(xs, ys)]          # against slope 1.0
    return {
        "n": len(xs),
        "slope_least_squares": round(ls, 4),
        "slope_robust_median": round(robust, 4),
        "rms_residual_deg": round(
            math.sqrt(sum(r * r for r in resid) / len(resid)), 3),
        "max_abs_residual_deg": round(max(abs(r) for r in resid), 3),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs="+", required=True)
    ap.add_argument("--out", default="reports/synthetic_warp.json")
    ap.add_argument("--save-examples", default=None,
                    help="directory to write a few warped images for eyeballing")
    args = ap.parse_args(argv)

    files = sorted(f for d in args.dir for ext in ("jpg", "jpeg", "png")
                   for f in glob.glob(os.path.join(d, f"*.{ext}")))
    if not files:
        print("no images", file=sys.stderr)
        return 1

    noise = NoiseModel.load()
    # metric -> list of (true_delta, measured_delta)
    pts: dict[str, list[tuple[float, float]]] = {}
    n_used = 0
    skipped = {"no_pose": 0, "not_side": 0, "warp_lost_pose": 0,
               "view_changed": 0}

    for path in files:
        rgb = L.load_rgb(path)
        base = L.detect(rgb)
        if base is None:
            skipped["no_pose"] += 1
            continue
        bview = estimate_view(base)
        if bview.view != "side":
            skipped["not_side"] += 1
            continue
        scale = L.body_scale(base)
        if scale < 200:
            skipped["not_side"] += 1
            continue

        from posture.view import near_side_indices
        side = near_side_indices(base, bview.facing)
        y_hip = base.xy(side["hip"])[1]
        y_ear = base.xy(side["ear"])[1]
        bmetrics = compute_all(base, bview, noise)
        n_used += 1
        print(f"  {os.path.basename(path)} (body {scale:.0f}px)", file=sys.stderr)

        for frac in SHIFT_FRACTIONS:
            shift = frac * scale
            wimg = warp(rgb, shift, y_hip, y_ear)

            # ground truth: recompute metrics from the exact warped landmarks
            tpose = warped_pose(base, shift, y_hip, y_ear)
            tmetrics = compute_all(tpose, bview, noise)

            got = L.detect(wimg)
            if got is None:
                skipped["warp_lost_pose"] += 1
                continue
            gview = estimate_view(got)
            if gview.view != bview.view or gview.facing != bview.facing:
                skipped["view_changed"] += 1
                continue
            mmetrics = compute_all(got, gview, noise)

            for k in mmetrics:
                if k not in bmetrics or k not in tmetrics:
                    continue
                true_d = tmetrics[k].value - bmetrics[k].value
                meas_d = mmetrics[k].value - bmetrics[k].value
                if abs(true_d) < 0.25:
                    continue   # this metric barely moves; ratio is meaningless
                pts.setdefault(k, []).append((true_d, meas_d))

            if args.save_examples and frac in (0.10, -0.06):
                os.makedirs(args.save_examples, exist_ok=True)
                from PIL import Image
                Image.fromarray(wimg).save(os.path.join(
                    args.save_examples,
                    f"{os.path.basename(path).rsplit('.',1)[0]}_shift{frac:+.2f}.jpg"))

    report = {"n_images": n_used, "skipped": skipped,
              "shift_fractions": list(SHIFT_FRACTIONS), "metrics": {}}
    for k, vals in pts.items():
        report["metrics"][k] = _fit([t for t, _ in vals], [m for _, m in vals])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n{'='*74}")
    print(f"RESPONSE TO A KNOWN POSTURE CHANGE -- {n_used} side-view images")
    print(f"{'='*74}")
    print("\nThe subject is sheared forward by a known amount, so the true "
          "change in\nevery reading is computable. Slope 1.000 means the tool "
          "reports a real\npostural change at its true size. Below 1 means it "
          "under-reports.\n")
    print(f"  {'metric':<24} {'n':>4} {'LS slope':>9} {'robust':>8} "
          f"{'RMS resid':>10} {'max':>8}")
    for k in REPORT_ORDER:
        e = report["metrics"].get(k)
        if not e:
            continue
        print(f"  {k:<24} {e['n']:>4} {e['slope_least_squares']:>9.3f} "
              f"{e['slope_robust_median']:>8.3f} {e['rms_residual_deg']:>10.2f} "
              f"{e['max_abs_residual_deg']:>8.2f}")
    if skipped:
        print(f"\nskipped: {skipped}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
