"""What the person who uploaded the photo actually reads.

`assess()` returns an instrument's output: readings, uncertainties, bands,
guard findings. That is the right thing to test and to audit, and the wrong
thing to show someone who just wants to know whether their posture has a
problem and what to do about it. This module turns an Assessment into that
answer, and nothing more:

    issues        concrete problems, each with a plain summary, a likely
                  cause, and three exercises
    good          what looked fine
    not_measured  what could not be judged from this photo, and why, in words
    retake        when the photo cannot be judged at all: what to change
    photo_tips    small things that would make the next photo more accurate

The mapping from bands to what is said is the one place where the tool's
caution has to survive translation into plain language, so it is spelled out:

    reading's 1-sigma interval sits inside...      the person is told
    ------------------------------------------     ------------------------
    the reference band                             good
    reference and slight (straddles the cut)       borderline ("临界")
    the slight band, or slight and notable         slight ("轻度")
    the notable band                               notable ("明显")
    ...uncertainty wider than the slight band      not measured

That is: the person is told the LEAST severe thing the measurement cannot rule
out. A reading whose error bar reaches from "slight" to "notable" is reported
as slight, never rounded up. A metric the instrument cannot resolve at all is
reported as not measured, never as fine.

The copy lives in consumer_zh.json so the API, the web page and a mini-program
say exactly the same words; web/posture.js implements this same mapping and
scripts/export_web_demo.py checks the two agree.
"""
from __future__ import annotations

import json
import os

from . import landmarks as L
from .assess import Assessment
from .thresholds import BAND_NOTABLE, BAND_REFERENCE, BAND_SLIGHT

_COPY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "consumer_zh.json")
with open(_COPY_PATH, encoding="utf-8") as _fh:
    COPY = json.load(_fh)

LEVEL_NOTABLE, LEVEL_SLIGHT, LEVEL_BORDERLINE = "notable", "slight", "borderline"
LEVEL_GOOD, LEVEL_NOT_MEASURED = "good", "not_measured"

# Order issues are listed in: most noticeable first, then by body region.
#
# pelvis_tilt is judged by assess() but deliberately left out of this list.
# On every front-view photograph in testdata/ (40 of 40, per
# scripts/report_survey.py) its uncertainty was wider than its own slight
# band, so it came back below the noise floor every time: the two hip-joint
# landmarks are too close together for their noise.
# Listing it would print the same "could not measure" line on every result,
# which tells the reader nothing and makes the tool look broken. It returns
# here if a better hip estimate ever resolves it.
ISSUE_ORDER = ("head_over_hip", "shoulder_protraction", "trunk_sway",
               "shoulder_tilt", "lateral_head_shift")


def level_for(v) -> str:
    """Band + interval -> what the person is told (see the module table).

    The least severe band inside the 1-sigma interval is computed directly
    rather than from MetricVerdict.alternative_band, because that field keeps
    only the MOST severe neighbour: a reading whose interval spans all three
    bands would otherwise be reported as slight instead of borderline.
    """
    if v.below_noise_floor:
        return LEVEL_NOT_MEASURED
    m, spec = v.measurement, v.spec
    u = m.uncertainty if m.uncertainty == m.uncertainty else 0.0   # NaN -> 0
    if spec.direction == "positive_only":
        nearest = m.value - u
    else:
        nearest = 0.0 if abs(m.value) <= u else abs(m.value) - u
    lowest = spec.classify(nearest)
    if lowest == BAND_NOTABLE:
        return LEVEL_NOTABLE
    if lowest == BAND_SLIGHT:
        return LEVEL_SLIGHT
    # The interval reaches the reference band: fine only if it stays there.
    return LEVEL_GOOD if v.resolved else LEVEL_BORDERLINE


def _image_side(pose: L.PoseResult, a: int, b: int, *, higher: bool) -> str:
    """'left'/'right' in IMAGE terms for the higher of two landmarks.

    Deliberately not the person's anatomical side: a mirrored selfie swaps
    those, and the person can always check the photo they are looking at.
    """
    pa, pb = pose.xy(a), pose.xy(b)
    top = pa if (pa[1] < pb[1]) == higher else pb
    other = pb if top is pa else pa
    return "left" if top[0] < other[0] else "right"


def _direction(key: str, a: Assessment) -> dict:
    """Which way the deviation goes, for the metrics where that matters."""
    pose, m = a.pose, a.measurements[key]
    if key == "trunk_sway":
        return {"direction": "forward" if m.value > 0 else "back"}
    if pose is None:
        return {}
    if key == "shoulder_tilt":
        return {"side": _image_side(pose, L.LEFT_SHOULDER, L.RIGHT_SHOULDER, higher=True)}
    if key == "pelvis_tilt":
        return {"side": _image_side(pose, L.LEFT_HIP, L.RIGHT_HIP, higher=True)}
    if key == "lateral_head_shift":
        # positive value = nose to the image-right of the shoulder midline
        return {"side": "right" if m.value > 0 else "left"}
    return {}


def _copy_for(key: str, extra: dict) -> dict:
    c = COPY["issues"][key]
    if key == "trunk_sway":
        c = {**c, **c[extra.get("direction", "forward")]}
    side = COPY["sides"].get(extra.get("side", ""), "")
    return {"name": c["name"], "summary": c["summary"].replace("{side}", side),
            "why": c["why"], "tips": c["tips"]}


def _parts(names: list[str]) -> str:
    out: list[str] = []
    for n in names:
        for frag, part in COPY["parts"].items():
            if frag in n and part not in out:
                out.append(part)
    return "、".join(out) or "部分身体"


def consumer_report(a: Assessment) -> dict:
    """Assessment -> the plain answer. Pure function of its input."""
    if a.view is None:
        r = COPY["retake"]["no_pose"]
        return {"status": "retake", "view": None,
                "retake": [{"key": "no_pose", **r}],
                "issues": [], "borderline": [], "good": [], "not_measured": [],
                "photo_tips": [], "other_view": None,
                "disclaimer": COPY["disclaimer"]}

    view = a.view.view
    other = {"side": "front", "front": "side"}.get(view)
    base = {"view": view, "disclaimer": COPY["disclaimer"],
            "other_view": ({"view": other, **COPY["other_view"][other]}
                           if other else None)}

    if a.blocked:
        seen, retake = set(), []
        for f in a.findings:
            if f.severity != "block" or f.key in seen:
                continue
            seen.add(f.key)
            c = COPY["retake"].get(f.key)
            if c:
                retake.append({"key": f.key, **c})
        if not retake:   # a block with no consumer copy must still say something
            retake.append({"key": "unknown", "title": "这张照片没法分析",
                           "tip": "请换一张从头到脚、站直拍的全身照。"})
        return {**base, "status": "retake", "retake": retake, "issues": [],
                "borderline": [], "good": [], "not_measured": [],
                "photo_tips": []}

    issues, borderline, good, not_measured = [], [], [], []
    for key in ISSUE_ORDER:
        if key not in COPY["issues"]:
            continue
        good_name = COPY["issues"][key]["good"]
        if key in a.unavailable:
            not_measured.append({"key": key, "name": good_name,
                                 "reason": COPY["not_measured"]["out_of_frame"]
                                 .replace("{parts}", _parts(a.unavailable[key]))})
            continue
        if key in a.unstable:
            not_measured.append({"key": key, "name": good_name,
                                 "reason": COPY["not_measured"]["unstable"]})
            continue
        v = a.verdicts.get(key)
        if v is None:
            continue
        lvl = level_for(v)
        if lvl == LEVEL_NOT_MEASURED:
            not_measured.append({"key": key, "name": good_name,
                                 "reason": COPY["not_measured"]["floored"]})
            continue
        if lvl == LEVEL_GOOD:
            good.append({"key": key, "name": good_name})
            continue
        extra = _direction(key, a)
        item = {"key": key, "level": lvl,
                "level_label": COPY["levels"][lvl],
                "landmarks": list(a.measurements[key].landmark_indices),
                **extra, **_copy_for(key, extra)}
        (borderline if lvl == LEVEL_BORDERLINE else issues).append(item)

    if not (issues or borderline or good):
        # Every metric was withheld one way or another. An empty "no problems
        # found" page would be the worst possible reading of that, so it is a
        # retake instead.
        c = COPY["retake"]["unstable_detection"]
        return {**base, "status": "retake",
                "retake": [{"key": "nothing_judged", **c}], "issues": [],
                "borderline": [], "good": [], "not_measured": not_measured,
                "photo_tips": []}

    issues.sort(key=lambda i: 0 if i["level"] == LEVEL_NOTABLE else 1)
    tips = [COPY["photo_tips"][f.key] for f in a.findings
            if f.severity == "warn" and f.key in COPY["photo_tips"]]
    return {**base, "status": "ok", "retake": [], "issues": issues,
            "borderline": borderline, "good": good,
            "not_measured": not_measured, "photo_tips": tips}
