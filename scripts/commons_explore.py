#!/usr/bin/env python3
"""Explore Wikimedia Commons for posture-assessment candidate photographs.

Why Commons is worth a second look
----------------------------------
The previous phase reported that `Category:Human posture` and
`Category:Full-length portrait photographs` "do not exist" and scored 0 hits
over ~350 images. The first half of that is a near miss: the category is
`Category:Human postures` -- PLURAL -- and it does exist, as does
`Category:Anthropometry`. Guessing category names is unreliable; the API can
be asked instead.

Commons also has the cleanest licensing of any source tried: everything on it
is freely licensed or public domain, so hits can be committed to the repo
rather than used for statistics only. That is not true of COCO's
NonCommercial-licensed majority, nor of research-only datasets like
DeepFashion.

Rate limits
-----------
Commons returns HTTP 429 quickly for unthrottled clients. This script makes
sequential requests with a delay and a descriptive User-Agent, per
https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits. It is slow on
purpose.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = ("posture-judge-research/0.2 "
      "(https://github.com/yiggeror/a-test-of-posture-judge; research use)")

# Seconds between requests. Commons 429s aggressively; this is deliberately
# conservative rather than tuned to the edge.
DELAY = 3.0


def api(params: dict, *, retries: int = 4) -> dict:
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    backoff = DELAY
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                body = r.read()
            return json.loads(body)
        except Exception as exc:
            msg = str(exc)
            if attempt == retries - 1:
                print(f"    [give up] {msg[:90]}", file=sys.stderr)
                return {}
            time.sleep(backoff)
            backoff *= 2
    return {}


def subcategories(title: str, limit: int = 50) -> list[str]:
    d = api({"action": "query", "list": "categorymembers",
             "cmtitle": title, "cmtype": "subcat", "cmlimit": limit})
    return [c["title"] for c in d.get("query", {}).get("categorymembers", [])]


def files_in(title: str, limit: int = 100) -> list[str]:
    d = api({"action": "query", "list": "categorymembers",
             "cmtitle": title, "cmtype": "file", "cmlimit": limit})
    return [c["title"] for c in d.get("query", {}).get("categorymembers", [])]


def search_files(query: str, limit: int = 50) -> list[str]:
    d = api({"action": "query", "list": "search", "srsearch": query,
             "srnamespace": 6, "srlimit": limit})
    return [c["title"] for c in d.get("query", {}).get("search", [])]


def image_info(titles: list[str]) -> list[dict]:
    """URL, size and license for up to 50 files per call."""
    out = []
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = api({"action": "query", "titles": "|".join(chunk),
                 "prop": "imageinfo",
                 "iiprop": "url|size|extmetadata",
                 "iiextmetadatafilter": "LicenseShortName|Artist|License"})
        for page in (d.get("query", {}).get("pages") or {}).values():
            for ii in page.get("imageinfo", []) or []:
                meta = ii.get("extmetadata") or {}
                out.append({
                    "title": page.get("title", ""),
                    "url": ii.get("url", ""),
                    "width": ii.get("width", 0),
                    "height": ii.get("height", 0),
                    "license": (meta.get("LicenseShortName", {}) or {}).get("value", "?"),
                    "artist": (meta.get("Artist", {}) or {}).get("value", "")[:80],
                })
        time.sleep(DELAY)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["explore", "collect"], default="explore")
    ap.add_argument("--out", default=None, help="CSV manifest (collect mode)")
    ap.add_argument("--min-height", type=int, default=800,
                    help="skip images shorter than this")
    args = ap.parse_args(argv)

    # Categories confirmed to exist by querying, not by guessing.
    seeds = [
        "Category:Human postures",
        "Category:Anthropometry",
    ]
    # Free-text searches aimed at the specific framing this tool needs.
    searches = [
        'intitle:"side view" standing',
        'intitle:"profile" standing full length',
        "anthropometry standing posture side",
        "posture assessment lateral view",
        'intitle:"standing" "full length" portrait',
    ]

    if args.mode == "explore":
        for s in seeds:
            print(f"\n=== {s} -- subcategories ===")
            for t in subcategories(s):
                print("   ", t)
            time.sleep(DELAY)
            print(f"=== {s} -- direct files (first 20) ===")
            for t in files_in(s, limit=20):
                print("   ", t)
            time.sleep(DELAY)

        for q in searches:
            print(f"\n=== search: {q} ===")
            hits = search_files(q, limit=15)
            for t in hits:
                print("   ", t)
            time.sleep(DELAY)
        return 0

    # collect mode: gather candidates with URLs, sizes and licenses
    titles: list[str] = []
    for s in seeds:
        for sub in subcategories(s):
            titles += files_in(sub, limit=60)
            time.sleep(DELAY)
        titles += files_in(s, limit=100)
        time.sleep(DELAY)
    for q in searches:
        titles += search_files(q, limit=50)
        time.sleep(DELAY)

    titles = sorted(set(titles))
    print(f"\n{len(titles)} unique files found; fetching metadata",
          file=sys.stderr)
    infos = image_info(titles)
    infos = [i for i in infos if i["height"] >= args.min_height]
    infos.sort(key=lambda i: -i["height"])

    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "image_id", "split", "file_name", "coco_url", "flickr_url",
                "license_id", "license_name", "width", "height", "view_guess",
                "commons_title"])
            w.writeheader()
            for i in infos:
                name = i["title"].replace("File:", "").replace(" ", "_")
                w.writerow({
                    "image_id": name.rsplit(".", 1)[0][:60],
                    "split": "commons",
                    "file_name": name,
                    "coco_url": i["url"],
                    "flickr_url": "",
                    "license_id": "",
                    "license_name": i["license"],
                    "width": i["width"],
                    "height": i["height"],
                    "view_guess": "",
                    "commons_title": i["title"],
                })
        print(f"manifest written to {args.out}", file=sys.stderr)

    print(f"\n{len(infos)} files at >= {args.min_height}px tall:")
    for i in infos[:40]:
        print(f"  {i['height']:>5}px  {i['license']:<18} {i['title'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
