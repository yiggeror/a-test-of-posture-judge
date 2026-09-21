#!/usr/bin/env python3
"""Build the norm-reference distribution that thresholds are derived from.

What this is
------------
For each metric, the distribution of readings over a set of photographs that
passed every guard. Percentiles of that distribution replace the hand-picked
thresholds, converting

    "18 degrees is a notable tendency"                      (a guess)

into

    "18 degrees is above the 95th percentile of n=NN photographs"
                                                            (a measurement)

What this is NOT
----------------
Clinical calibration. Nobody has labelled any image here as showing, or not
showing, any postural condition. The reference population is photographs from
a general-purpose object-detection dataset, not a screened cohort, so an
unusual reading means unusual-for-this-set and nothing more. That limitation
is written into the generated file itself, so it travels with the numbers.

Threshold-versus-noise check
----------------------------
A threshold smaller than the measurement error is not a threshold, it is a
coin flip with a number attached. This script cross-references
reports/repeatability.json and marks any metric whose proposed cut point sits
below its own p90 measurement error. Those are reported as unusable rather
than shipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture import assess  # noqa: E402
from posture.geometry import mean, percentile, stdev  # noqa: E402
from posture.landmarks import load_rgb  # noqa: E402
from posture.thresholds import DIAGNOSTIC_ONLY  # noqa: E402

# Below this, percentiles are noise. Matches the guard in thresholds.py.
# provenance: guess -- a rule of thumb, not a power analysis
MIN_N_FOR_PERCENTILES = 20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", default="reports/reference_distribution.json")
    ap.add_argument("--repeatability", default="reports/repeatability.json")
    ap.add_argument("--source-label", default=None,
                    help="how to describe the reference population in the "
                         "generated file; must be accurate")
    args = ap.parse_args(argv)

    values: dict[str, list[float]] = defaultdict(list)
    n_images = n_accepted = 0

    for d in args.dirs:
        files = sorted(f for ext in ("jpg", "jpeg", "png")
                       for f in glob.glob(os.path.join(d, f"*.{ext}")))
        for path in files:
            n_images += 1
            try:
                rgb = load_rgb(path)
            except Exception:
                continue
            a = assess(rgb)
            if not a.ok:
                continue
            n_accepted += 1
            for k, m in a.measurements.items():
                values[k].append(m.value)
            print(f"  accepted {os.path.basename(path)}", file=sys.stderr)

    noise_p90: dict[str, float] = {}
    if os.path.exists(args.repeatability):
        try:
            with open(args.repeatability) as fh:
                rep = json.load(fh)
            for k, e in (rep.get("metrics") or {}).items():
                r = e.get("rotation") or {}
                if "p90_abs_residual_deg" in r:
                    noise_p90[k] = r["p90_abs_residual_deg"]
        except (OSError, ValueError):
            pass

    metrics = {}
    for k, vs in sorted(values.items()):
        abs_vs = [abs(v) for v in vs]
        entry = {
            "n": len(vs),
            "mean": round(mean(vs), 3),
            "sd": round(stdev(vs), 3) if len(vs) > 1 else None,
            "p05": round(percentile(vs, 5), 3),
            "p50": round(percentile(vs, 50), 3),
            "p95": round(percentile(vs, 95), 3),
            "abs_p80": round(percentile(abs_vs, 80), 3),
            "abs_p95": round(percentile(abs_vs, 95), 3),
            "diagnostic_only": k in DIAGNOSTIC_ONLY,
            "usable_for_thresholds": len(vs) >= MIN_N_FOR_PERCENTILES,
        }
        if k in noise_p90:
            entry["measurement_p90_error_deg"] = noise_p90[k]
            # A cut point below the measurement error cannot separate anything.
            entry["threshold_exceeds_noise"] = entry["abs_p80"] > noise_p90[k]
            if not entry["threshold_exceeds_noise"]:
                entry["usable_for_thresholds"] = False
                entry["unusable_reason"] = (
                    f"proposed 80th-percentile cut ({entry['abs_p80']} deg) is "
                    f"below this metric's own p90 measurement error "
                    f"({noise_p90[k]} deg), so it cannot separate a real "
                    "difference from measurement noise")
        metrics[k] = entry

    source = args.source_label or (
        f"{n_accepted} photographs accepted out of {n_images} examined, from "
        f"{', '.join(os.path.basename(d.rstrip('/')) for d in args.dirs)}")

    out = {
        "source": source,
        "n_images_examined": n_images,
        "n_images_accepted": n_accepted,
        "min_n_for_percentiles": MIN_N_FOR_PERCENTILES,
        "caveat": (
            "Norm-reference only. No image in this set carries a clinical "
            "label, and the population is unscreened general photography "
            "rather than a cohort. A reading above the 95th percentile is "
            "unusual for this set; that is not the same as unhealthy."),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{'='*76}")
    print(f"REFERENCE DISTRIBUTION  ({n_accepted} accepted / {n_images} examined)")
    print(f"{'='*76}")
    print(f"  {'metric':<24} {'n':>4} {'p50':>8} {'|p80|':>8} {'|p95|':>8} "
          f"{'noise p90':>10}  usable")
    for k, e in metrics.items():
        noise = e.get("measurement_p90_error_deg")
        noise_s = f"{noise:>10.2f}" if noise is not None else f"{'-':>10}"
        flag = "yes" if e["usable_for_thresholds"] else "NO"
        print(f"  {k:<24} {e['n']:>4} {e['p50']:>8.2f} {e['abs_p80']:>8.2f} "
              f"{e['abs_p95']:>8.2f} {noise_s}  {flag}")
    for k, e in metrics.items():
        if e.get("unusable_reason"):
            print(f"\n  {k}: {e['unusable_reason']}")
        elif not e["usable_for_thresholds"] and e["n"] < MIN_N_FOR_PERCENTILES:
            print(f"\n  {k}: only n={e['n']}, below the n>={MIN_N_FOR_PERCENTILES} "
                  "minimum; hand-picked thresholds stay in use")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
