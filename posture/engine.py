"""MediaPipe Pose Landmarker wrapper.

Model: MediaPipe Pose Landmarker (BlazePose GHUM).
Licence: Apache License 2.0 - both the mediapipe package and the published
.task bundles. See LICENSES.md for why this satisfies the Apache-2.0/MIT-only
constraint, and why YOLO-pose (AGPL-3.0) and OpenPose (non-commercial academic
licence) are excluded.
"""
from __future__ import annotations

import os
import threading

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .landmarks import Pt

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
VARIANTS = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}

# Two distinct locks, never nested in the other's direction.
# `_create_lock` guards building/caching a landmarker; `_infer_lock` serialises
# detection calls. A single shared lock deadlocks, because detect() would hold
# it while get_landmarker() tried to take it again (threading.Lock is not
# reentrant).
_create_lock = threading.Lock()
_infer_lock = threading.Lock()
_cache: dict[str, vision.PoseLandmarker] = {}


class ModelMissing(RuntimeError):
    pass


def model_path(variant: str = "full") -> str:
    p = os.path.join(MODEL_DIR, VARIANTS[variant])
    if not os.path.exists(p):
        raise ModelMissing(
            f"Model weights not found: {p}\nRun: python scripts/fetch_models.py")
    return p


def get_landmarker(variant: str = "full") -> vision.PoseLandmarker:
    """Create (once) and reuse a landmarker.

    PoseLandmarker is not documented as thread-safe, so detection is
    serialised under `_infer_lock` in `detect`. For a demo server that is fine;
    a real deployment would use a worker pool instead.
    """
    with _create_lock:
        if variant not in _cache:
            opts = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_path(variant)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                output_segmentation_masks=False,
            )
            _cache[variant] = vision.PoseLandmarker.create_from_options(opts)
        return _cache[variant]


def detect(bgr: np.ndarray, variant: str = "full") -> list[Pt] | None:
    """Run pose detection, returning 33 landmarks in PIXEL coordinates.

    Returns None when the model finds no pose. Never fabricates a result.
    """
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker = get_landmarker(variant)   # takes _create_lock, then releases it
    with _infer_lock:
        result = landmarker.detect(image)
    if not result.pose_landmarks:
        return None
    # Scale normalised (x/width, y/height) into pixels here, once, so that no
    # downstream geometry ever sees aspect-ratio-distorted coordinates.
    return [
        Pt(lm.x * w, lm.y * h,
           float(getattr(lm, "visibility", 1.0) or 0.0),
           float(getattr(lm, "presence", 1.0) or 0.0))
        for lm in result.pose_landmarks[0]
    ]


def read_image(data: bytes, max_side: int = 1400) -> np.ndarray | None:
    """Decode bytes to BGR, downscaling very large images.

    Downscaling is uniform, so it changes no angle and no normalised ratio.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img
