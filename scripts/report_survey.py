#!/usr/bin/env python3
"""What would a person actually be told, photo by photo?

Runs the full pipeline plus posture/report.py over the committed test sets and
prints, per set, how many photos get a result versus a retake request, and for
each photo that gets a result: the issues, the borderline items, what looked
fine, and what was withheld and why.

This is the view that exposed pelvis_tilt: it came back "cannot measure" on
every front-view photo, which is why report.py no longer lists it. The last
line re-derives that figure so the claim in report.py stays checkable.
"""
from __future__ import annotations

import collections
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posture.assess import assess_file  # noqa: E402
from posture.report import consumer_report  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = ["pexels/side", "pexels/front", "pexels/edge", "side", "front",
        "pexels/negative", "negative"]


def main() -> int:
    by_set = collections.defaultdict(collections.Counter)
    pelvis = collections.Counter()
    for s in SETS:
        for p in sorted(glob.glob(os.path.join(ROOT, "testdata", s, "*.jpg"))):
            a = assess_file(p)
            r = consumer_report(a)
            by_set[s][r["status"]] += 1
            if a.view is not None and a.view.view == "front":
                v = a.verdicts.get("pelvis_tilt")
                if v is not None:
                    pelvis["floored" if v.below_noise_floor else "resolvable"] += 1
            if r["status"] != "ok":
                continue
            fmt = lambda xs: ", ".join(xs) or "-"   # noqa: E731
            print(f"{s:<16} {os.path.basename(p)[:36]:<37}"
                  f" issues: {fmt(i['name'] + '·' + i['level_label'] for i in r['issues'])}"
                  f" | borderline: {fmt(i['name'] for i in r['borderline'])}"
                  f" | good: {fmt(g['name'] for g in r['good'])}"
                  f" | withheld: {fmt(n['name'] + '(' + n['reason'] + ')' for n in r['not_measured'])}")
    print()
    for s in SETS:
        c = by_set[s]
        print(f"{s:<16} result {c['ok']:>3}   retake {c['retake']:>3}")
    print(f"\npelvis_tilt on front views: {pelvis['floored']} below noise floor, "
          f"{pelvis['resolvable']} resolvable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
