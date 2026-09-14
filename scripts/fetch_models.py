#!/usr/bin/env python3
"""Download MediaPipe Pose Landmarker weights and verify their integrity.

All three variants are Apache-2.0, published by Google at
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

BASE = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_{v}/float16/1/pose_landmarker_{v}.task")

# Recorded from the downloads made while building this demo (2026-09-14).
# They pin exactly what was validated here; if an upstream republish changes
# them the script says so loudly rather than silently using different weights.
EXPECTED = {
    "lite":  ("59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
              5777746),
    "full":  ("5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
              9398198),
    "heavy": ("64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
              30664242),
}

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "models")


def main(variants: list[str]) -> int:
    os.makedirs(OUT, exist_ok=True)
    bad = 0
    for v in variants:
        dest = os.path.join(OUT, f"pose_landmarker_{v}.task")
        if not os.path.exists(dest):
            url = BASE.format(v=v)
            print(f"downloading {v} <- {url}")
            urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        exp_sha, exp_size = EXPECTED[v]
        ok = (digest == exp_sha and size == exp_size)
        print(f"{'OK  ' if ok else 'FAIL'} {v:<6} {size:>9} bytes  sha256={digest[:16]}…")
        if not ok:
            print(f"     expected {exp_size} bytes sha256={exp_sha[:16]}…")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["lite", "full", "heavy"]))
