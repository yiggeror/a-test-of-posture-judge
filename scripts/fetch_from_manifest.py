#!/usr/bin/env python3
"""Download candidate images listed in a collector's sources.csv.

The fallback path when the collecting agent can hand back URLs and attribution
but not the image bytes themselves. Run it anywhere with open network access,
then triage the result:

    python scripts/fetch_from_manifest.py --csv sources.csv --out inbox
    python scripts/triage_images.py --dir inbox

Expected CSV header: filename,source_url,author,license
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.request

# Licences we accept without a human decision. Anything else is downloaded but
# flagged, so the call stays with a person rather than with this script.
CLEAR_LICENCES = ("cc0", "public domain", "pd", "cc-by", "cc by",
                  "pexels", "unsplash", "pixabay")

UA = "posture-demo-testdata-fetcher/1.0"


def looks_open(lic: str) -> bool:
    low = (lic or "").strip().lower()
    return any(k in low for k in CLEAR_LICENCES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="inbox")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    with open(a.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{a.csv} 里没有数据行。")
        return 1

    missing = {"filename", "source_url", "license"} - set(rows[0])
    if missing:
        print(f"CSV 缺少必需列：{', '.join(sorted(missing))}")
        return 1

    ok = skipped = failed = 0
    flagged: list[str] = []
    for r in rows:
        name, url, lic = r["filename"], r["source_url"], r.get("license", "")
        dest = os.path.join(a.out, name)
        if os.path.exists(dest):
            print(f"SKIP   {name} （已存在）")
            skipped += 1
            continue
        if not looks_open(lic):
            flagged.append(f"{name} -> {lic!r}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=a.timeout) as resp:
                data = resp.read()
            with open(dest, "wb") as out:
                out.write(data)
            print(f"OK     {name:<40}{len(data):>9} B  [{lic}]")
            ok += 1
        except Exception as e:
            print(f"FAIL   {name:<40}{e}")
            failed += 1

    print(f"\n下载 {ok}，跳过 {skipped}，失败 {failed}。")
    if flagged:
        print("\n以下条目的授权字段不在已知开放许可清单内，请人工确认后再使用：")
        for f in flagged:
            print("  " + f)
    print(f"\n下一步： python scripts/triage_images.py --dir {a.out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
