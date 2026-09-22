#!/usr/bin/env python3
"""Mine COCO person-keypoint annotations for posture-assessment candidate photos.

Why this exists
---------------
The previous phase searched stock galleries visually and measured a ~0.1% hit
rate for usable side-view standing photos (~1400 Pexels images inspected, plus
~350 Wikimedia images with zero hits). That is a content-scarcity problem, not a
network problem, so re-running the same search with better connectivity would
reproduce the same rate.

COCO changes the economics because it ships *annotations*, not just images.
Every person instance carries 17 keypoints with per-point visibility flags,
including both ears and both ankles. The exact properties that had to be judged
by eye before -- "is the ear visible", "are the feet in frame", "is this a side
view", "are the arms hanging free" -- are all decidable from the annotation
vector. So instead of viewing N images to find 0.001*N candidates, we filter
262k annotated person instances programmatically and only download the hits.

This does NOT claim the resulting images are good posture-assessment photos.
It claims they are worth a MediaPipe triage pass, which `triage_images.py`
then runs. Every threshold below is tunable and tagged with its provenance.

Licensing
---------
COCO images are Flickr-sourced with per-image license ids. We keep only the
ids that permit redistribution *and* derivative works (overlays are
derivatives), so the candidate set can be committed to the repo:

    4 = CC BY 2.0        5 = CC BY-SA 2.0
    7 = No known copyright restrictions (Flickr Commons)
    8 = United States Government Work

Deliberately excluded: 1/2/3 (NonCommercial), 6 (NoDerivs).
The COCO annotation files themselves are CC BY 4.0.

Usage
-----
    python scripts/coco_mine.py --ann-dir <dir> --view side --out candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any

# COCO keypoint order (fixed by the dataset format).
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KP_NAMES)}

# Visibility flags as defined by COCO: 0 unlabeled, 1 labeled-but-occluded,
# 2 labeled-and-visible.
V_ABSENT, V_OCCLUDED, V_VISIBLE = 0, 1, 2

REDISTRIBUTABLE_LICENSES = (4, 5, 7, 8)

# --- Thresholds -------------------------------------------------------------
# Provenance tags follow the project convention:
#   guess              - picked by hand, no supporting evidence
#   geometric-estimate - derived from body proportions / projection geometry
#   literature-adjacent- taken from published anthropometry, adapted
#   measured           - computed from data in this repo

# View is judged from shoulder spread normalised by TORSO length
# (shoulder-to-hip), not by shoulder-to-ankle height. Torso length is
# available whether or not the feet are in frame, and most usable side-view
# photographs are cropped somewhere below the knee -- normalising by a span
# that needs the ankles discards them before they are ever looked at.
#
# Biacromial width is ~0.40 m against a ~0.50 m torso, so a fully front-facing
# adult projects ~0.80 and a true lateral view ~0. These cut points are the
# old shoulder-to-ankle ones (0.14 / 0.22) rescaled by the 1.38/0.50 ratio
# between the two spans, so the geometry is unchanged.
# provenance: geometric-estimate (Pheasant anthropometry, ratio derived here)
SIDE_MAX_SHOULDER_SPREAD = 0.38
FRONT_MIN_SHOULDER_SPREAD = 0.60

# Standing test when the ankles are NOT in frame: in a lateral view a standing
# subject's hip-to-knee segment is near vertical, while a seated one's is near
# horizontal. The two are ~90 deg apart, so this cut sits in a wide gap.
# provenance: geometric-estimate (seated vs standing thigh orientation)
MAX_THIGH_ANGLE_FROM_VERTICAL = 40.0

# A standing adult's hip-to-ankle vertical span is ~50% of their eye-to-ankle
# span. Sitting or crouching collapses this ratio well below 0.40.
# provenance: geometric-estimate (segment proportions)
STAND_MIN_LEG_FRACTION = 0.42
STAND_MAX_LEG_FRACTION = 0.60

# The knee should lie close to the hip-ankle line when the leg is extended.
# Expressed as a fraction of leg length. A bent knee (sitting, walking,
# lunging) deviates far more than this.
# provenance: guess -- chosen to admit normal stance variation, never validated
KNEE_MAX_OFFSET_FRACTION = 0.13

# COCO images are capped at 640px on the long side. Requiring the person to
# occupy at least this many pixels of height keeps enough resolution for the
# ear/shoulder landmarks to mean anything. Pitfall from the previous phase:
# candidate quality judged at low resolution is systematically overestimated.
# provenance: guess -- a resolution floor, not a validated quality threshold
MIN_PERSON_HEIGHT_PX = 380

# A second person this large relative to the subject risks tripping the
# multi-person guard downstream, or occluding the subject.
# provenance: guess
MAX_BYSTANDER_AREA_RATIO = 0.15

# Total keypoint count is NOT used as a gate. It penalises exactly the view
# being searched for: in a lateral photograph half the body is self-occluded,
# so COCO's num_keypoints is structurally lower for side views than for front
# views of equal quality. Gating on it rejected 500 of 1856 otherwise-usable
# side candidates. What matters is whether the SPECIFIC points each metric
# needs are present, which extract_features and passes() check directly.
# Retained only as a floor against near-empty annotations.
# provenance: geometric-estimate (self-occlusion in a lateral view)
MIN_KEYPOINTS = 7


def _pt(kps: list[int], idx: int) -> tuple[float, float, int]:
    """Return (x, y, visibility) for one keypoint index."""
    return float(kps[idx * 3]), float(kps[idx * 3 + 1]), int(kps[idx * 3 + 2])


def _mid(a: tuple[float, float, int], b: tuple[float, float, int]) -> tuple[float, float] | None:
    """Midpoint of two keypoints, or None if either is unlabeled."""
    if a[2] == V_ABSENT or b[2] == V_ABSENT:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def extract_features(ann: dict[str, Any]) -> dict[str, Any] | None:
    """Compute view/pose features from one COCO person annotation.

    Returns None when the annotation is too incomplete to describe a standing
    full-body subject. Image coordinates: +y points down.
    """
    kps = ann.get("keypoints") or []
    if len(kps) != len(KP_NAMES) * 3:
        return None

    ls, rs = _pt(kps, KP["left_shoulder"]), _pt(kps, KP["right_shoulder"])
    lh, rh = _pt(kps, KP["left_hip"]), _pt(kps, KP["right_hip"])
    la, ra = _pt(kps, KP["left_ankle"]), _pt(kps, KP["right_ankle"])
    lk, rk = _pt(kps, KP["left_knee"]), _pt(kps, KP["right_knee"])
    lear, rear = _pt(kps, KP["left_ear"]), _pt(kps, KP["right_ear"])
    nose = _pt(kps, KP["nose"])

    mid_sh = _mid(ls, rs)
    mid_hip = _mid(lh, rh)
    if mid_sh is None or mid_hip is None:
        return None

    # Feet are OPTIONAL. Two separate traps live here, and both cost most of
    # the candidate pool:
    #
    #  1. Requiring both ankles V_VISIBLE structurally excludes lateral views,
    #     because in a true side view the far ankle is occluded by the near
    #     leg and annotators mark it V_OCCLUDED.
    #  2. Requiring the feet AT ALL discards photographs that fully support
    #     the head and shoulder metrics. `head_over_hip` uses the ear and the
    #     hip; it does not care where the feet are. Demanding them cut the
    #     side-view pool from ~4900 to 47.
    #
    # So the ankles are recorded when usable and the downstream metrics that
    # need them are withheld per metric, exactly as posture/guards.py does.
    has_ankles = (la[2] != V_ABSENT and ra[2] != V_ABSENT
                  and (la[2] == V_VISIBLE or ra[2] == V_VISIBLE))
    mid_ankle = (((la[0] + ra[0]) / 2.0, (la[1] + ra[1]) / 2.0)
                 if has_ankles else None)

    # Head reference: topmost labelled head point.
    head_candidates = [p for p in (nose, lear, rear,
                                   _pt(kps, KP["left_eye"]), _pt(kps, KP["right_eye"]))
                       if p[2] != V_ABSENT]
    if not head_candidates:
        return None
    head_y = min(p[1] for p in head_candidates)

    torso_h = mid_hip[1] - mid_sh[1]
    if torso_h <= 1.0:
        return None  # not upright, or degenerate
    head_hip_h = mid_hip[1] - head_y
    shoulder_ankle_h = (mid_ankle[1] - mid_sh[1]) if mid_ankle else None
    head_ankle_h = (mid_ankle[1] - head_y) if mid_ankle else None

    # View proxy: horizontal separation of the shoulders, normalised by TORSO
    # length so it works whether or not the feet are in frame.
    shoulder_spread = abs(ls[0] - rs[0]) / torso_h if (
        ls[2] != V_ABSENT and rs[2] != V_ABSENT) else None
    hip_spread = abs(lh[0] - rh[0]) / torso_h if (
        lh[2] != V_ABSENT and rh[2] != V_ABSENT) else None

    # Standing proxy. With the ankles, use the leg's share of body height.
    leg_fraction = ((mid_ankle[1] - mid_hip[1]) / head_ankle_h
                    if mid_ankle and head_ankle_h and head_ankle_h > 1 else None)

    # Without them, use thigh orientation: a standing thigh is near vertical,
    # a seated one near horizontal.
    thigh_angle = None
    thighs = []
    for hip, knee in ((lh, lk), (rh, rk)):
        if V_ABSENT in (hip[2], knee[2]):
            continue
        dx, dy = knee[0] - hip[0], knee[1] - hip[1]
        if math.hypot(dx, dy) < 1.0:
            continue
        thighs.append(abs(math.degrees(math.atan2(dx, dy))))
    if thighs:
        thigh_angle = min(thighs)

    # Leg extension: distance from each knee to its hip-ankle line, as a
    # fraction of leg length. Large values mean a bent knee.
    knee_offsets = []
    for hip, knee, ankle in ((lh, lk, la), (rh, rk, ra)):
        if V_ABSENT in (hip[2], knee[2], ankle[2]):
            continue
        leg_len = math.hypot(ankle[0] - hip[0], ankle[1] - hip[1])
        if leg_len < 1.0:
            continue
        # Perpendicular distance from knee to the hip->ankle line.
        cross = abs((ankle[0] - hip[0]) * (hip[1] - knee[1])
                    - (hip[0] - knee[0]) * (ankle[1] - hip[1]))
        knee_offsets.append(cross / leg_len / leg_len)
    knee_offset = max(knee_offsets) if knee_offsets else None

    # Ear visibility: a lateral view shows one ear, a frontal view shows two.
    n_ears_visible = sum(1 for e in (lear, rear) if e[2] == V_VISIBLE)
    n_ears_labelled = sum(1 for e in (lear, rear) if e[2] != V_ABSENT)

    # Arms-at-side proxy: wrists below hip level. The previous phase found the
    # dominant failure mode was arms crossed / on hips / in pockets, all of
    # which raise the wrists to or above hip height.
    #
    # ANY rather than ALL: in a lateral view the far arm is behind the torso
    # and is frequently mislabelled or placed by inference, so requiring both
    # wrists to be low rejects side views for a reason that is about
    # annotation rather than about the pose.
    wrists_below_hips = None
    wrist_pts = [p for p in (_pt(kps, KP["left_wrist"]), _pt(kps, KP["right_wrist"]))
                 if p[2] != V_ABSENT]
    if wrist_pts:
        wrists_below_hips = any(w[1] > mid_hip[1] for w in wrist_pts)

    return {
        "shoulder_spread": shoulder_spread,
        "hip_spread": hip_spread,
        "leg_fraction": leg_fraction,
        "thigh_angle": thigh_angle,
        "has_ankles": has_ankles,
        "torso_px": torso_h,
        "knee_offset": knee_offset,
        "n_ears_visible": n_ears_visible,
        "n_ears_labelled": n_ears_labelled,
        "wrists_below_hips": wrists_below_hips,
        "person_height_px": head_ankle_h if head_ankle_h else head_hip_h * 1.95,
        "num_keypoints": ann.get("num_keypoints", 0),
    }


def classify_view(f: dict[str, Any]) -> str:
    """Label the camera view from the shoulder-spread proxy."""
    s = f["shoulder_spread"]
    if s is None:
        # Far shoulder entirely unlabelled is itself evidence of a lateral view,
        # but weak evidence, so it is reported separately rather than merged in.
        return "unknown"
    if s <= SIDE_MAX_SHOULDER_SPREAD:
        return "side"
    if s >= FRONT_MIN_SHOULDER_SPREAD:
        return "front"
    return "oblique"


def passes(f: dict[str, Any], view: str, want: str, min_height: float,
           require_feet: bool = False) -> tuple[bool, str]:
    """Apply the acceptance gate. Returns (accepted, reason_if_rejected)."""
    if f["num_keypoints"] < MIN_KEYPOINTS:
        return False, "too_few_keypoints"
    # Size is judged on the TORSO, the span the head and shoulder metrics are
    # actually measured across. Estimating whole-body height from the torso
    # and then gating on that just adds a conversion error for photographs
    # whose legs are out of frame.
    if f["torso_px"] < min_height / 2.83:
        return False, "person_too_small"

    if require_feet and not f["has_ankles"]:
        return False, "no_feet_in_frame"

    # Standing test, adapted to what is actually visible. With the ankles,
    # use the legs' share of body height. Without them, use thigh
    # orientation -- a standing thigh is near vertical, a seated one near
    # horizontal, and the two are ~90 deg apart.
    if f["leg_fraction"] is not None:
        if not (STAND_MIN_LEG_FRACTION <= f["leg_fraction"]
                <= STAND_MAX_LEG_FRACTION):
            return False, "not_standing"
    elif f["thigh_angle"] is not None:
        if f["thigh_angle"] > MAX_THIGH_ANGLE_FROM_VERTICAL:
            return False, "not_standing_thigh"
    else:
        return False, "cannot_verify_standing"

    if f["knee_offset"] is not None and f["knee_offset"] > KNEE_MAX_OFFSET_FRACTION:
        return False, "knee_bent"
    if f["wrists_below_hips"] is False:
        return False, "arms_raised"
    if want != "any" and view != want:
        return False, f"view_is_{view}"
    if want == "side" and f["n_ears_visible"] < 1:
        return False, "no_visible_ear"
    if want == "front" and f["n_ears_visible"] < 2:
        return False, "ears_not_both_visible"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann-dir", required=True,
                    help="directory holding person_keypoints_*.json")
    ap.add_argument("--splits", default="train2017,val2017")
    ap.add_argument("--view", default="side", choices=["side", "front", "oblique", "any"])
    ap.add_argument("--licenses", default=",".join(str(i) for i in REDISTRIBUTABLE_LICENSES),
                    help="comma-separated COCO license ids to keep")
    ap.add_argument("--min-height-px", type=float, default=MIN_PERSON_HEIGHT_PX)
    ap.add_argument("--max-bystander", type=float, default=MAX_BYSTANDER_AREA_RATIO,
                    help="reject if another person's area exceeds this fraction "
                         "of the subject's")
    ap.add_argument("--require-feet", action="store_true",
                    help="only keep photos with the ankles in frame. Off by "
                         "default: head and shoulder metrics do not need the "
                         "feet, and demanding them cut the side-view pool "
                         "from ~4900 to 47.")
    ap.add_argument("--all-licenses", action="store_true",
                    help="keep every license. Use for computing reference "
                         "statistics only -- images under COCO license ids "
                         "1/2/3/6 must NOT be redistributed in this repo.")
    ap.add_argument("--out", required=True, help="output CSV manifest")
    ap.add_argument("--stats-out", default=None, help="optional rejection-reason tally JSON")
    args = ap.parse_args(argv)

    keep_licenses = (None if args.all_licenses
                     else {int(x) for x in args.licenses.split(",") if x.strip()})
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = defaultdict(int)
    totals = {"images": 0, "person_anns": 0, "license_ok_images": 0}

    for split in args.splits.split(","):
        split = split.strip()
        path = os.path.join(args.ann_dir, f"person_keypoints_{split}.json")
        if not os.path.exists(path):
            print(f"[skip] missing {path}", file=sys.stderr)
            continue
        print(f"[load] {path}", file=sys.stderr)
        with open(path) as fh:
            data = json.load(fh)

        images = {im["id"]: im for im in data["images"]}
        by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("iscrowd"):
                continue
            by_image[ann["image_id"]].append(ann)

        totals["images"] += len(images)
        totals["person_anns"] += sum(len(v) for v in by_image.values())

        for image_id, anns in by_image.items():
            im = images[image_id]
            if keep_licenses is not None and im.get("license") not in keep_licenses:
                reasons["license_excluded"] += 1
                continue
            totals["license_ok_images"] += 1

            anns_sorted = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)
            primary = anns_sorted[0]
            others = anns_sorted[1:]

            # Reject crowds: any other person comparable in size to the subject.
            p_area = primary.get("area", 0) or 1
            if any((o.get("area", 0) / p_area) > args.max_bystander for o in others):
                reasons["multiple_people"] += 1
                continue

            f = extract_features(primary)
            if f is None:
                reasons["incomplete_annotation"] += 1
                continue

            view = classify_view(f)
            ok, why = passes(f, view, args.view, args.min_height_px,
                             require_feet=args.require_feet)
            if not ok:
                reasons[why] += 1
                continue

            rows.append({
                "image_id": image_id,
                "split": split,
                "file_name": im["file_name"],
                "coco_url": im.get("coco_url", ""),
                "flickr_url": im.get("flickr_url", ""),
                "license_id": im.get("license"),
                "width": im.get("width"),
                "height": im.get("height"),
                "view_guess": view,
                "shoulder_spread": round(f["shoulder_spread"], 4)
                if f["shoulder_spread"] is not None else "",
                "hip_spread": round(f["hip_spread"], 4) if f["hip_spread"] is not None else "",
                "leg_fraction": round(f["leg_fraction"], 4)
                if f["leg_fraction"] is not None else "",
                "thigh_angle": round(f["thigh_angle"], 1)
                if f["thigh_angle"] is not None else "",
                "has_ankles": int(bool(f["has_ankles"])),
                "torso_px": round(f["torso_px"], 1),
                "knee_offset": round(f["knee_offset"], 4)
                if f["knee_offset"] is not None else "",
                "n_ears_visible": f["n_ears_visible"],
                "person_height_px": round(f["person_height_px"], 1),
                "num_keypoints": f["num_keypoints"],
            })

    # Rank by subject size: resolution was the binding constraint on usefulness
    # in the previous phase, so the largest subjects get triaged first.
    rows.sort(key=lambda r: r["person_height_px"], reverse=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n=== coco_mine: view={args.view} ===", file=sys.stderr)
    print(f"images scanned          : {totals['images']}", file=sys.stderr)
    print(f"person annotations      : {totals['person_anns']}", file=sys.stderr)
    print(f"images w/ allowed license: {totals['license_ok_images']}", file=sys.stderr)
    print(f"CANDIDATES              : {len(rows)}", file=sys.stderr)
    print("rejections:", file=sys.stderr)
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26} {v}", file=sys.stderr)

    if args.stats_out:
        with open(args.stats_out, "w") as fh:
            json.dump({"totals": totals, "reasons": dict(reasons),
                       "candidates": len(rows), "view": args.view}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
