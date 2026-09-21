#!/usr/bin/env python3
"""Run the full pipeline over a directory and report what happened.

Reports acceptance rate and rejection reasons as prominently as the readings
themselves. A tool that accepts 5% of real photographs is a different product
from one that accepts 80%, and that number is invisible if you only ever look
at the images that worked.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture import assess  # noqa: E402
from posture.geometry import mean, percentile, stdev  # noqa: E402
from posture.landmarks import load_rgb  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--overlay-dir", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--no-deep-guards", action="store_true",
                    help="skip the guards that re-run a model (faster)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(f for ext in ("jpg", "jpeg", "png", "webp")
                   for f in glob.glob(os.path.join(args.dir, f"*.{ext}")))
    if not files:
        print(f"no images in {args.dir}", file=sys.stderr)
        return 1

    overlay_dir = args.overlay_dir or os.path.join(args.dir, "_overlays")
    if args.save_overlays:
        os.makedirs(overlay_dir, exist_ok=True)

    records, views, block_reasons, warn_reasons = [], Counter(), Counter(), Counter()
    values: dict[str, list[float]] = {}
    n_accept = 0

    for path in files:
        try:
            rgb = load_rgb(path)
        except Exception as exc:
            block_reasons["unreadable_file"] += 1
            records.append({"file": os.path.basename(path), "error": str(exc)})
            continue

        a = assess(rgb, deep_guards=not args.no_deep_guards)
        name = os.path.basename(path)

        if a.view is None:
            block_reasons["no_pose_detected"] += 1
            records.append({"file": name, "accepted": False,
                            "blocked_by": ["no_pose_detected"]})
            if not args.quiet:
                print(f"{name:<28} BLOCK  no pose detected")
            continue

        views[a.view.view] += 1
        blocks = [f.key for f in a.findings if f.severity == "block"]
        warns = [f.key for f in a.findings if f.severity == "warn"]
        for b in blocks:
            block_reasons[b] += 1
        for w in warns:
            warn_reasons[w] += 1

        if a.ok:
            n_accept += 1
            for k, m in a.measurements.items():
                values.setdefault(k, []).append(m.value)

        records.append({
            "file": name,
            "accepted": a.ok,
            "view": a.view.view,
            "yaw_deg": round(a.view.yaw_deg, 1),
            "shoulder_spread": round(a.view.shoulder_spread, 4),
            "facing": a.view.facing,
            "facing_confidence": round(a.view.facing_confidence, 3),
            "blocked_by": blocks,
            "warnings": warns,
            "measurements": {k: {"value": round(m.value, 2),
                                 "uncertainty": round(m.uncertainty, 2),
                                 "span_px": round(m.span_px, 1)}
                             for k, m in a.measurements.items()},
            "verdicts": {k: {"band": v.band, "resolved": v.resolved,
                             "provenance": v.spec.provenance}
                         for k, v in a.verdicts.items()},
        })

        if not args.quiet:
            status = "OK   " if a.ok else "BLOCK"
            detail = ",".join(blocks) if blocks else ""
            print(f"{name:<28} {status}  {a.view.view:<8} {detail}")

        if args.save_overlays:
            from posture import landmarks as L
            from posture.overlay import draw
            pose = L.detect(rgb)
            if pose is not None:
                unstable = set()
                for f in a.findings:
                    if f.key == "unstable_landmarks":
                        unstable |= set(f.detail.get("drift_frac_of_body_scale", {}))
                draw(rgb, pose, unstable=unstable).save(
                    os.path.join(overlay_dir, name))

    n = len(files)
    print(f"\n{'='*64}")
    print(f"images                : {n}")
    print(f"accepted              : {n_accept}  ({100.0*n_accept/n:.1f}%)")
    print(f"blocked               : {n - n_accept}  ({100.0*(n-n_accept)/n:.1f}%)")
    print(f"\nviews detected        : {dict(views)}")
    print("\nblock reasons:")
    for k, v in block_reasons.most_common():
        print(f"  {k:<26} {v}")
    if warn_reasons:
        print("\nwarnings:")
        for k, v in warn_reasons.most_common():
            print(f"  {k:<26} {v}")

    if values:
        print("\nmetric distributions over ACCEPTED images:")
        print(f"  {'metric':<24} {'n':>4} {'mean':>8} {'sd':>7} {'p5':>7} "
              f"{'p50':>7} {'p95':>7}")
        for k in sorted(values):
            vs = values[k]
            print(f"  {k:<24} {len(vs):>4} {mean(vs):>8.1f} {stdev(vs):>7.1f} "
                  f"{percentile(vs,5):>7.1f} {percentile(vs,50):>7.1f} "
                  f"{percentile(vs,95):>7.1f}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"dir": args.dir, "n_images": n, "n_accepted": n_accept,
                       "views": dict(views), "block_reasons": dict(block_reasons),
                       "warn_reasons": dict(warn_reasons), "records": records},
                      fh, indent=2)
        print(f"\njson written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
