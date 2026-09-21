#!/usr/bin/env python3
"""Measure how much the readings move when the posture does not.

Why this can be done without any labelled data
----------------------------------------------
The honest objection to "calibrating" this tool is that nobody has told us the
true forward-head angle of anyone in any photograph. That blocks accuracy
measurement. It does NOT block precision measurement, because some image
transformations have a mathematically known effect on the true answer:

  ROTATION by theta   Every angle-from-vertical and angle-from-horizontal
                      changes by exactly theta. The posture is untouched; only
                      the camera's idea of "down" moved.

                      The expected slope is NOT one number. PIL rotates
                      counter-clockwise for a positive angle, which decreases
                      an angle measured from vertical and increases one
                      measured from horizontal -- so frontal metrics expect
                      +1.0 and sagittal metrics -1.0. And the sagittal sign
                      convention is anchored to which way the subject faces
                      (angle_from_vertical multiplies the horizontal offset by
                      `anterior`), so a left-facing subject's sagittal reading
                      is the negative of a right-facing one's and expects
                      +1.0 instead. The harness normalises by facing before
                      fitting.

  SCALE, JPEG, BRIGHTNESS, CONTRAST
                      The true answer does not change at all. Any movement in
                      the reading is pure measurement error.

  HORIZONTAL FLIP     Frontal readings negate exactly. Sagittal readings are
                      preserved, because the subject's facing direction flips
                      with them and the sign convention is anchored to facing.

Three ways this measurement has already been got wrong
------------------------------------------------------
Each of these produced a confident number that described nothing. They are
recorded because every one of them looked like a finding about the tool.

1. POOLING A CHANGED SUBJECT. The first version reported a rotation slope of
   -0.25 against -1.00 expected, which reads as near-total failure. It was the
   detector locking onto a DIFFERENT PERSON after the transform -- on a 193px
   subject with bystanders, the post-transform "subject" was someone else,
   landmarks 1.98 body-lengths away. Fixed by mapping landmarks back through
   the known transform and rejecting the comparison when the subject moved.

2. ONE EXPECTED SLOPE FOR BOTH PLANES. Frontal metrics expect +1.0, not -1.0.
   Before that was fixed, correct frontal behaviour (+0.98) looked like a
   catastrophe.

3. POOLING BOTH FACING DIRECTIONS. Sagittal readings for a left-facing subject
   are the negative of a right-facing one's, so on a mixed set the two
   populations CANCEL. This dragged the least-squares sagittal slope to -0.05
   and inflated the apparent residuals, which was written up as a heavy
   error tail before the cause was found. Fixed by normalising each sagittal
   delta by the baseline facing direction.

Those are genuinely different problems and mixing any of them into the
precision statistics produces a number that describes neither. So this script
measures:

  RE-ACQUISITION RATE  how often the detector finds the same subject again
                       after a transform that should not have disturbed it

  READING PRECISION    how far the reading moves GIVEN that the same subject
                       was re-acquired

A tool can be bad at either one. Reporting only the second would flatter it.

What this does NOT measure
--------------------------
Real test-retest repeatability also includes the subject re-standing, the
photographer re-framing, and clothing shifting. Those are strictly larger than
what is measured here, so every number this produces is a LOWER BOUND on
real-world variability. It is a noise floor, not a field-accuracy estimate.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture import landmarks as L  # noqa: E402
from posture.geometry import distance, percentile  # noqa: E402
from posture.metrics import NoiseModel, compute_all  # noqa: E402
from posture.view import estimate_view  # noqa: E402

ROTATIONS = (-12.0, -9.0, -6.0, -3.0, 3.0, 6.0, 9.0, 12.0)
SCALES = (0.7, 0.85, 1.25, 1.5)
JPEG_QUALITIES = (50, 70, 85)
ENHANCERS = (("brightness", 0.8), ("brightness", 1.25),
             ("contrast", 0.8), ("contrast", 1.25))

# PIL rotates counter-clockwise for a positive angle. That DECREASES an angle
# measured from vertical (sagittal metrics) and INCREASES an angle measured
# from horizontal (frontal metrics) -- the two planes respond with opposite
# sign, and an earlier version of this script got frontal results that looked
# like a catastrophic failure purely because it expected one slope for both.
ROTATION_EXPECTED: dict[str, float] = {
    # sagittal: angle from vertical
    "forward_head": -1.0, "shoulder_protraction": -1.0,
    "trunk_sway": -1.0, "head_over_hip": -1.0,
    # frontal: angle from horizontal
    "shoulder_tilt": +1.0, "pelvis_tilt": +1.0, "head_tilt": +1.0,
    # differences of two same-plane angles, and joint interior angles, are
    # rotation-invariant by construction
    "knee_deviation": 0.0, "head_vs_shoulder_tilt": 0.0,
}
# lateral_head_shift is an offset divided by a span, not an angle measured
# from either axis, so it has no simple closed-form response to rotation and
# is deliberately given no expected slope rather than a made-up one.
NO_ROTATION_EXPECTATION = {"lateral_head_shift"}

# A residual larger than this, after the known rotation effect is removed, is
# a gross failure rather than noise: it exceeds the entire range that
# separates "within reference" from "notable tendency" for most metrics.
# provenance: geometric-estimate -- set to the smallest "notable" threshold
GROSS_ERROR_DEG = 5.0

# A transformed detection whose landmarks, mapped back into the original
# frame, sit further than this from the baseline landmarks is a different
# subject, not a moved reading.
# provenance: geometric-estimate -- 0.15 of body height is far larger than any
# plausible localisation error and far smaller than the gap to another person
SUBJECT_IDENTITY_MAX_DRIFT = 0.15

# Subjects smaller than this are where re-acquisition collapses. Images below
# it are excluded and counted, rather than silently degrading every statistic.
# provenance: measured -- re-acquisition failures in this repo's reference set
# are concentrated almost entirely below this size
MIN_BODY_SCALE_PX = 250.0


def _measure(rgb, noise, model_path):
    pose = L.detect(rgb, model_path=model_path)
    if pose is None:
        return None
    view = estimate_view(pose)
    return pose, view, compute_all(pose, view, noise)


def _jpeg(rgb, quality):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.array(im.convert("RGB"))


def _scale(rgb, s):
    h, w = rgb.shape[:2]
    return np.array(Image.fromarray(rgb).resize(
        (max(8, int(w * s)), max(8, int(h * s))), Image.LANCZOS))


def _enhance(rgb, kind, factor):
    cls = {"brightness": ImageEnhance.Brightness,
           "contrast": ImageEnhance.Contrast}[kind]
    return np.array(cls(Image.fromarray(rgb)).enhance(factor))


def _rotate(rgb, deg):
    return np.array(Image.fromarray(rgb).rotate(
        deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)))


def _backmap_identity():
    """Transforms that leave pixel coordinates alone (JPEG, brightness, ...)."""
    return lambda p: p


def _backmap_scale(s):
    return lambda p: (p[0] / s, p[1] / s)


def _backmap_rotate(deg, shape_before, shape_after):
    """Undo PIL's rotate(deg, expand=True) for a point in the rotated frame."""
    h1, w1 = shape_before[:2]
    h2, w2 = shape_after[:2]
    c1 = (w1 / 2.0, h1 / 2.0)
    c2 = (w2 / 2.0, h2 / 2.0)
    rad = math.radians(deg)
    cos, sin = math.cos(rad), math.sin(rad)

    def back(p):
        dx, dy = p[0] - c2[0], p[1] - c2[1]
        return (dx * cos - dy * sin + c1[0], dx * sin + dy * cos + c1[1])
    return back


def _same_subject(base_pose, pose, backmap, body_scale):
    """Median landmark displacement after undoing the known transform."""
    ds = []
    for i in range(L.N_LANDMARKS):
        ds.append(distance(backmap(pose.xy(i)), base_pose.xy(i)) / body_scale)
    ds.sort()
    return ds[len(ds) // 2]


def _median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _fit_slope(xs, ys, expected=None):
    """Fit the rotation response two ways, because they disagree informatively.

    The least-squares slope is dragged a long way by a minority of gross
    failures; the median-of-ratios slope is not. On this repo's reference set
    the two differ substantially for the sagittal metrics (LS ~-0.63, robust
    ~-0.9), and that gap IS the finding: the typical reading tracks a known
    rotation well, while a heavy tail does not. Reporting only one of them
    would describe a tool that does not exist.

    Both are forced through the origin: a zero rotation must give a zero delta
    by construction, and fitting an intercept would let a systematic bias hide
    inside it.
    """
    if not xs:
        return {}
    sxx = sum(x * x for x in xs)
    ls = sum(x * y for x, y in zip(xs, ys)) / sxx if sxx > 0 else float("nan")
    robust = _median([y / x for x, y in zip(xs, ys) if x != 0])

    ref = ls if expected is None else expected
    resid = [y - ref * x for x, y in zip(xs, ys)]
    abs_resid = [abs(r) for r in resid]
    return {
        "n": len(xs),
        "slope_least_squares": round(ls, 4),
        "slope_robust_median": round(robust, 4),
        "rms_residual_deg": round(math.sqrt(sum(r * r for r in resid) / len(resid)), 3),
        "median_abs_residual_deg": round(_median(abs_resid), 3),
        "p90_abs_residual_deg": round(percentile(abs_resid, 90), 3),
        "gross_error_rate": round(
            sum(1 for r in abs_resid if r > GROSS_ERROR_DEG) / len(abs_resid), 4),
    }


def _rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, nargs="+",
                    help="one or more image directories")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="reports/repeatability.json")
    ap.add_argument("--noise-out", default="reports/landmark_noise.json")
    ap.add_argument("--write-noise", action="store_true")
    ap.add_argument("--min-body-scale", type=float, default=MIN_BODY_SCALE_PX)
    args = ap.parse_args(argv)

    files = sorted(f for d in args.dir for ext in ("jpg", "jpeg", "png")
                   for f in glob.glob(os.path.join(d, f"*.{ext}")))
    if not files:
        print(f"no images in {args.dir}", file=sys.stderr)
        return 1
    dirs_label = ", ".join(args.dir)

    noise = NoiseModel(frac_of_body_scale=0.0, provenance="unused-here")

    # family -> metric -> deltas
    inv_by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rot_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    flip_deltas: dict[str, list[float]] = defaultdict(list)
    landmark_drift: dict[str, list[float]] = defaultdict(list)

    counts = {"images_seen": 0, "images_used": 0, "too_small": 0,
              "no_baseline_pose": 0, "comparisons": 0,
              "reacquired": 0, "subject_changed": 0, "view_changed": 0,
              "no_detection": 0}

    used = 0
    for path in files:
        if used >= args.limit:
            break
        counts["images_seen"] += 1
        try:
            base_rgb = L.load_rgb(path)
        except Exception:
            continue
        base = _measure(base_rgb, noise, args.model)
        if base is None:
            counts["no_baseline_pose"] += 1
            continue
        b_pose, b_view, b_metrics = base
        b_scale = L.body_scale(b_pose)
        if b_scale < args.min_body_scale:
            counts["too_small"] += 1
            continue
        used += 1
        counts["images_used"] = used

        # Each job carries the back-map that undoes its own coordinate change,
        # so the subject-identity check compares like with like. Rotation's
        # back-map depends on the expanded canvas size, so it is built below
        # once the transformed image exists.
        jobs = []
        for q in JPEG_QUALITIES:
            jobs.append(("jpeg", f"q{q}", _jpeg(base_rgb, q),
                         _backmap_identity(), None))
        for s in SCALES:
            jobs.append(("scale", f"x{s}", _scale(base_rgb, s),
                         _backmap_scale(s), None))
        for kind, f in ENHANCERS:
            jobs.append((kind, f"{f}", _enhance(base_rgb, kind, f),
                         _backmap_identity(), None))
        for th in ROTATIONS:
            jobs.append(("rotate", f"{th:+.0f}", _rotate(base_rgb, th), None, th))

        for family, tag, rgb, backmap, theta in jobs:
            counts["comparisons"] += 1
            got = _measure(rgb, noise, args.model)
            if got is None:
                counts["no_detection"] += 1
                continue
            pose, view, metrics = got

            bm = (_backmap_rotate(theta, base_rgb.shape, rgb.shape)
                  if theta is not None else backmap)

            drift = _same_subject(b_pose, pose, bm, b_scale)
            if drift > SUBJECT_IDENTITY_MAX_DRIFT:
                counts["subject_changed"] += 1
                continue

            # The view gate and the facing direction must also be unchanged:
            # a reading taken under a different view classification, or with
            # the sagittal sign convention anchored the other way, is not
            # comparable to the baseline.
            if view.view != b_view.view or view.facing != b_view.facing:
                counts["view_changed"] += 1
                continue
            counts["reacquired"] += 1

            if theta is None:
                for k, m in metrics.items():
                    if k in b_metrics:
                        inv_by_family[family][k].append(m.value - b_metrics[k].value)
                for i in range(L.N_LANDMARKS):
                    d = distance(bm(pose.xy(i)), b_pose.xy(i)) / b_scale
                    landmark_drift[L.LANDMARK_NAMES[i]].append(d)
            else:
                # The sagittal sign convention is anchored to which way the
                # subject faces: angle_from_vertical multiplies the horizontal
                # offset by `anterior`, so a subject facing -x reads as the
                # NEGATIVE of one facing +x. A rotation therefore moves the
                # reading by -theta for a right-facing subject and by +theta
                # for a left-facing one.
                #
                # Pooling both without normalising makes the two populations
                # cancel. That is what dragged the least-squares slope to
                # -0.05 on the mixed-facing COCO set, which was previously
                # (wrongly) written up as a heavy error tail.
                for k, m in metrics.items():
                    if k not in b_metrics:
                        continue
                    delta = m.value - b_metrics[k].value
                    if ROTATION_EXPECTED.get(k) == -1.0:
                        delta *= b_view.facing
                    rot_points[k].append((theta, delta))

        # mirror
        got = _measure(np.ascontiguousarray(base_rgb[:, ::-1, :]), noise, args.model)
        if got is not None:
            _, _, metrics = got
            for k, m in metrics.items():
                if k not in b_metrics:
                    continue
                expected = (-b_metrics[k].value if b_metrics[k].plane == "frontal"
                            else b_metrics[k].value)
                flip_deltas[k].append(m.value - expected)

        print(f"  [{used}/{min(args.limit, len(files))}] "
              f"{os.path.basename(path)}", file=sys.stderr)

    # --- aggregate ----------------------------------------------------------
    all_metrics = sorted(set(rot_points) | set(flip_deltas)
                         | {k for fam in inv_by_family.values() for k in fam})
    report = {"dir": dirs_label, "counts": counts, "metrics": {},
              "expected_rotation_slope": ROTATION_EXPECTED,
              "gross_error_threshold_deg": GROSS_ERROR_DEG}

    for k in all_metrics:
        entry = {"invariance_by_family": {}}
        pooled = []
        for family, per_metric in inv_by_family.items():
            ds = per_metric.get(k, [])
            if ds:
                entry["invariance_by_family"][family] = {
                    "n": len(ds), "rms_deg": round(_rms(ds), 3),
                    "max_abs_deg": round(max(abs(d) for d in ds), 3)}
                pooled += ds
        if pooled:
            entry["invariance_pooled"] = {
                "n": len(pooled), "rms_deg": round(_rms(pooled), 3),
                "p95_abs_deg": round(percentile([abs(d) for d in pooled], 95), 3),
                "max_abs_deg": round(max(abs(d) for d in pooled), 3)}
        if rot_points.get(k):
            xs = [t for t, _ in rot_points[k]]
            ys = [d for _, d in rot_points[k]]
            expected = (None if k in NO_ROTATION_EXPECTATION
                        else ROTATION_EXPECTED.get(k))
            entry["rotation"] = _fit_slope(xs, ys, expected)
            entry["rotation"]["expected_slope"] = expected
        if flip_deltas.get(k):
            fd = flip_deltas[k]
            entry["flip"] = {"n": len(fd), "rms_deg": round(_rms(fd), 3),
                             "max_abs_deg": round(max(abs(d) for d in fd), 3)}
        report["metrics"][k] = entry

    per_landmark, all_drift = {}, []
    for name, ds in landmark_drift.items():
        if ds:
            per_landmark[name] = {"n": len(ds), "rms_frac": round(_rms(ds), 5),
                                  "p95_frac": round(percentile(ds, 95), 5)}
    critical = [L.LANDMARK_NAMES[i] for i in
                (L.LEFT_EAR, L.RIGHT_EAR, L.LEFT_SHOULDER, L.RIGHT_SHOULDER,
                 L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_ANKLE, L.RIGHT_ANKLE)]
    for nm in critical:
        all_drift += landmark_drift.get(nm, [])
    overall = _rms(all_drift)
    report["landmark_noise"] = {"overall_rms_frac_of_body_scale": round(overall, 5),
                                "critical_landmarks": critical,
                                "per_landmark": per_landmark}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    if args.write_noise and all_drift:
        with open(args.noise_out, "w") as fh:
            json.dump({"frac_of_body_scale": round(overall, 5),
                       "provenance": "measured",
                       "source": (f"scripts/repeatability.py, {used} images from "
                                  f"{dirs_label}, {counts['reacquired']} re-acquired "
                                  "comparisons under JPEG/scale/brightness/contrast"),
                       "per_landmark": per_landmark}, fh, indent=2)

    # --- print ---------------------------------------------------------------
    c = counts
    tot = max(1, c["reacquired"] + c["subject_changed"] + c["view_changed"]
                + c["no_detection"])
    print(f"\n{'='*78}\nREPEATABILITY -- {used} images from {dirs_label}\n{'='*78}")
    print(f"\nSubject re-acquisition (transform should not disturb the subject):")
    print(f"  same subject found again : {c['reacquired']}/{tot} "
          f"({100.0*c['reacquired']/tot:.1f}%)")
    print(f"  DIFFERENT subject locked : {c['subject_changed']} "
          f"({100.0*c['subject_changed']/tot:.1f}%)")
    print(f"  no detection at all      : {c['no_detection']} "
          f"({100.0*c['no_detection']/tot:.1f}%)")
    print(f"  images skipped (too small, < {args.min_body_scale:.0f}px) : {c['too_small']}")

    print("\nInvariance -- reading movement under transforms that do NOT change")
    print("the posture. Ideal = 0. Given the same subject was re-acquired.\n")
    fams = sorted(inv_by_family)
    print(f"  {'metric':<24}" + "".join(f"{f:>11}" for f in fams) + f"{'pooled':>11}")
    for k in all_metrics:
        e = report["metrics"][k]
        if not e.get("invariance_pooled"):
            continue
        row = f"  {k:<24}"
        for f in fams:
            d = e["invariance_by_family"].get(f)
            row += f"{d['rms_deg']:>11.2f}" if d else f"{'-':>11}"
        row += f"{e['invariance_pooled']['rms_deg']:>11.2f}"
        print(row)

    print("\nRotation -- the image is rotated by a known angle, so the true "
          "change in\neach reading is exactly that angle. Sagittal metrics "
          "(angle from vertical)\nmust track -1.000, frontal metrics (angle "
          "from horizontal) +1.000, and\nrotation-invariant metrics 0.000.\n"
          "\n'LS slope' is least-squares, 'robust' is the median of "
          "per-sample ratios.\nWhere they disagree, the typical reading "
          "tracks well but a tail does not --\nand 'gross' is how often the "
          f"residual exceeds {GROSS_ERROR_DEG:.0f} deg, which is the size of "
          "a whole\nverdict band.\n")
    print(f"  {'metric':<22} {'n':>5} {'expect':>7} {'LS':>7} {'robust':>8} "
          f"{'medResid':>9} {'p90':>7} {'gross':>7}")
    for k in all_metrics:
        r = report["metrics"][k].get("rotation")
        if not r:
            continue
        exp = r.get("expected_slope")
        exp_s = f"{exp:+.1f}" if exp is not None else "n/a"
        print(f"  {k:<22} {r['n']:>5} {exp_s:>7} "
              f"{r['slope_least_squares']:>7.2f} {r['slope_robust_median']:>8.2f} "
              f"{r['median_abs_residual_deg']:>9.2f} "
              f"{r['p90_abs_residual_deg']:>7.2f} "
              f"{100*r['gross_error_rate']:>6.0f}%")

    print("\nMirror -- frontal readings must negate, sagittal readings must be "
          "preserved.\n")
    print(f"  {'metric':<24} {'n':>5} {'RMS deg':>10} {'max':>9}")
    for k in all_metrics:
        f = report["metrics"][k].get("flip")
        if f:
            print(f"  {k:<24} {f['n']:>5} {f['rms_deg']:>10.2f} {f['max_abs_deg']:>9.2f}")

    print(f"\nLandmark localisation noise (RMS, fraction of shoulder-to-ankle):")
    print(f"  overall over critical landmarks: {overall:.5f}")
    for nm, st in sorted(per_landmark.items(), key=lambda kv: -kv[1]["rms_frac"])[:6]:
        print(f"    {nm:<20} {st['rms_frac']:.5f}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
