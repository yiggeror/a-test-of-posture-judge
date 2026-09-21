#!/usr/bin/env python3
"""Download images listed in a CSV manifest, recording provenance for each.

Prefers the Flickr-hosted original over the COCO mirror: COCO caps images at
640px on the long side, while the Flickr `_b` variant is 1024px and usually
still available. Resolution matters here because the previous phase found that
candidate quality judged at low resolution is systematically overestimated --
whether an ear is actually visible is not decidable from a thumbnail.

Provenance is written to `sources.csv` next to the images and is never
invented: every row records the exact URL the bytes came from, the COCO image
id, and the license id. Rows whose download failed are recorded as failures
rather than silently dropped.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.request

# COCO license id -> human-readable name. Ids 4/5/7/8 permit redistribution
# and derivative works; the rest do not and must not be committed to the repo.
LICENSE_NAMES = {
    1: "CC BY-NC-SA 2.0", 2: "CC BY-NC 2.0", 3: "CC BY-NC-ND 2.0",
    4: "CC BY 2.0", 5: "CC BY-SA 2.0", 6: "CC BY-ND 2.0",
    7: "No known copyright restrictions", 8: "US Government Work",
}
REDISTRIBUTABLE = {4, 5, 7, 8}

UA = "posture-judge/0.2 (research; contact via repository)"


def flickr_large(url: str) -> str | None:
    """Rewrite a Flickr `_z` (640px) URL to the `_b` (1024px) variant."""
    m = re.match(r"^(https?://.*?_[0-9a-f]+)_[a-z]\.jpg$", url)
    return f"{m.group(1)}_b.jpg" if m else None


def fetch(url: str, dest: str, timeout: float = 30.0) -> tuple[bool, str, int]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return False, f"http {r.status}", 0
            data = r.read()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", 0
    if len(data) < 2048:
        return False, f"suspiciously small ({len(data)} bytes)", 0
    with open(dest, "wb") as fh:
        fh.write(data)
    return True, "", len(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redistributable-only", action="store_true",
                    help="keep only license ids that allow redistribution "
                         "and derivatives (4,5,7,8). Required for anything "
                         "committed to this repository.")
    ap.add_argument("--delay", type=float, default=0.2,
                    help="seconds between requests, to stay polite")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.manifest) as fh:
        rows = list(csv.DictReader(fh))

    if args.redistributable_only:
        before = len(rows)
        rows = [r for r in rows if int(r.get("license_id") or 0) in REDISTRIBUTABLE]
        print(f"[license] kept {len(rows)}/{before} rows", file=sys.stderr)
    if args.limit:
        rows = rows[:args.limit]

    records, n_ok, n_fail = [], 0, 0
    for i, r in enumerate(rows, 1):
        image_id = r["image_id"]
        name = f"coco_{image_id}.jpg"
        dest = os.path.join(args.out_dir, name)

        candidates = []
        if r.get("flickr_url"):
            big = flickr_large(r["flickr_url"])
            if big:
                candidates.append(big)
            candidates.append(r["flickr_url"])
        if r.get("coco_url"):
            candidates.append(r["coco_url"])

        ok, err, size, used = False, "no url", 0, ""
        if os.path.exists(dest) and os.path.getsize(dest) > 2048:
            ok, size, used = True, os.path.getsize(dest), "(cached)"
        else:
            for url in candidates:
                ok, err, size = fetch(url, dest)
                if ok:
                    used = url
                    break
                time.sleep(args.delay)

        lic = int(r.get("license_id") or 0)
        records.append({
            "file_name": name if ok else "",
            "coco_image_id": image_id,
            "split": r.get("split", ""),
            "source_url": used,
            "license_id": lic,
            "license_name": LICENSE_NAMES.get(lic, "unknown"),
            "redistributable": "yes" if lic in REDISTRIBUTABLE else "no",
            "view_guess": r.get("view_guess", ""),
            "bytes": size,
            "status": "ok" if ok else f"failed: {err}",
        })
        n_ok += ok
        n_fail += (not ok)
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ok={n_ok} fail={n_fail}", file=sys.stderr)
        if not ok:
            time.sleep(args.delay)

    src = os.path.join(args.out_dir, "sources.csv")
    with open(src, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    print(f"\ndownloaded {n_ok}, failed {n_fail}", file=sys.stderr)
    print(f"provenance written to {src}", file=sys.stderr)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
