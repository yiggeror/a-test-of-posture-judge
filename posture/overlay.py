"""Skeleton and measurement overlay drawing.

Uses PIL only, so the overlay works in a headless container without any GL
context. Landmarks that a guard flagged as unstable are drawn hollow, so the
picture shows which points the reading actually rests on.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import landmarks as L
from .geometry import midpoint

SKELETON_COLOR = (64, 196, 255)
POINT_COLOR = (255, 255, 255)
UNSTABLE_COLOR = (255, 96, 96)
PLUMB_COLOR = (255, 214, 64)
MEASURE_COLOR = (124, 252, 160)


def draw(image_rgb: np.ndarray, pose: L.PoseResult, *,
         unstable: set[str] | None = None,
         highlight: list[tuple[int, int]] | None = None,
         plumb_from: int | None = None) -> Image.Image:
    """Return a copy of the image with the skeleton drawn on it.

    `highlight` draws the specific segments a metric was computed from.
    `plumb_from` drops a vertical reference line from one landmark, which is
    what makes a sagittal angle readable at a glance.
    """
    img = Image.fromarray(np.ascontiguousarray(image_rgb)).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    unstable = unstable or set()

    scale = L.body_scale(pose)
    lw = max(2, int(scale * 0.008))
    r = max(3, int(scale * 0.012))

    for a, b in L.POSE_EDGES:
        d.line([pose.xy(a), pose.xy(b)], fill=SKELETON_COLOR + (200,), width=lw)

    if plumb_from is not None:
        x, y = pose.xy(plumb_from)
        d.line([(x, max(0, y - scale * 1.1)), (x, min(img.height, y + scale * 0.2))],
               fill=PLUMB_COLOR + (180,), width=max(1, lw // 2))

    for a, b in (highlight or []):
        d.line([pose.xy(a), pose.xy(b)], fill=MEASURE_COLOR + (255,),
               width=lw + 2)

    for idx, lm in enumerate(pose.landmarks):
        name = L.LANDMARK_NAMES[idx]
        x, y = lm.xy
        if name in unstable:
            # Hollow marker: the model inferred this point rather than seeing it.
            d.ellipse([x - r, y - r, x + r, y + r], outline=UNSTABLE_COLOR,
                      width=max(2, lw))
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=POINT_COLOR + (230,))
    return img


def draw_for_metric(image_rgb: np.ndarray, pose: L.PoseResult, metric,
                    unstable: set[str] | None = None) -> Image.Image:
    """Overlay tailored to one measurement: its segment plus a plumb line."""
    idxs = metric.landmark_indices
    segs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)] if len(idxs) >= 2 else []
    plumb = idxs[0] if metric.plane == "sagittal" and idxs else None
    return draw(image_rgb, pose, unstable=unstable, highlight=segs, plumb_from=plumb)
