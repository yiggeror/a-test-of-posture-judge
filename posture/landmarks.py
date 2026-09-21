"""MediaPipe Pose Landmarker wrapper.

Uses the Tasks API (mediapipe >= 0.10). The older `mp.solutions.pose` API was
removed in mediapipe 1.0.x, so most tutorials found online no longer apply.

The wrapper deliberately exposes `visibility` and `presence` but the rest of
the codebase must not treat them as evidence that a landmark is correctly
placed. Three confirmed counter-examples from the previous phase:

  * a photo of a bare palm got a full 33-point skeleton
  * a sock/calf close-up got a full 33-point skeleton
  * a subject whose long hair covered the ear had the `ear` landmark placed on
    the hair with visibility 1.000, and the app reported a 25.39 deg forward
    head angle from it

`visibility` answers "would this point be occluded if a person were here", not
"is there a person here" and not "is this point in the right place".
"""
from __future__ import annotations

import atexit
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

# Landmark indices, fixed by the BlazePose 33-point topology.
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

N_LANDMARKS = 33

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Edges used for the overlay drawing.
POSE_EDGES: tuple[tuple[int, int], ...] = (
    (LEFT_EAR, LEFT_EYE), (LEFT_EYE, NOSE), (NOSE, RIGHT_EYE), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE), (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE), (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
)

DEFAULT_MODEL = os.environ.get(
    "POSTURE_MODEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "pose_landmarker_heavy.task"),
)


@dataclass(frozen=True)
class Landmark:
    x: float           # pixels
    y: float           # pixels
    z: float           # relative depth, roughly in units of hip-width; unreliable
    visibility: float  # see module docstring -- NOT a correctness signal
    presence: float

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class PoseResult:
    landmarks: list[Landmark]
    width: int
    height: int
    n_poses_detected: int
    # Bounding boxes (x0, y0, x1, y1) in pixels of every OTHER detected pose.
    # The multi-person guard needs their sizes, not just the count: a bystander
    # in the background is harmless, a second person of comparable size is not.
    other_pose_bboxes: list[tuple[float, float, float, float]] = field(
        default_factory=list)

    def __getitem__(self, idx: int) -> Landmark:
        return self.landmarks[idx]

    def xy(self, idx: int) -> tuple[float, float]:
        return self.landmarks[idx].xy

    def visibility(self, idx: int) -> float:
        return self.landmarks[idx].visibility


class LandmarkerUnavailable(RuntimeError):
    """Raised when the model file is missing or mediapipe cannot be loaded."""


# Every landmarker built, so `_close_landmarkers` can shut them down before
# interpreter teardown. lru_cache exposes no way to enumerate its values.
_LIVE_LANDMARKERS: list = []


@lru_cache(maxsize=4)
def _get_landmarker(model_path: str, num_poses: int, min_conf: float):
    """Build and cache a PoseLandmarker.

    Cached because constructing one costs ~1s and the validation scripts call
    detect() thousands of times. IMAGE running mode is stateless, so reuse
    across images is safe and keeps results deterministic.
    """
    if not os.path.exists(model_path):
        raise LandmarkerUnavailable(
            f"model not found: {model_path}\n"
            "Run scripts/setup.sh, or set POSTURE_MODEL to a .task file."
        )
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover - environment failure
        raise LandmarkerUnavailable(f"mediapipe import failed: {exc}") from exc

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=min_conf,
        min_pose_presence_confidence=min_conf,
        output_segmentation_masks=False,
    )
    lm = vision.PoseLandmarker.create_from_options(options)
    _LIVE_LANDMARKERS.append(lm)
    return lm


def detect(image: "np.ndarray | str", *, model_path: str | None = None,
           num_poses: int = 4, min_confidence: float = 0.5) -> PoseResult | None:
    """Run pose detection on an RGB array or an image path.

    `num_poses` is >1 on purpose: the multi-person guard needs to know that a
    second person was found, which is impossible if the detector is told to
    return at most one.

    Returns None when no pose is detected. The FIRST returned pose is used as
    the subject; MediaPipe orders poses by detection score, not by size, so
    `guards` re-checks that the chosen subject is the dominant one.
    """
    import mediapipe as mp

    model_path = model_path or DEFAULT_MODEL
    landmarker = _get_landmarker(model_path, num_poses, min_confidence)

    if isinstance(image, str):
        mp_image = mp.Image.create_from_file(image)
        arr = mp_image.numpy_view()
        h, w = arr.shape[0], arr.shape[1]
    else:
        arr = np.ascontiguousarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 RGB array, got shape {arr.shape}")
        h, w = arr.shape[0], arr.shape[1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)

    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None

    subject = result.pose_landmarks[0]
    lms = [
        Landmark(
            x=lm.x * w,
            y=lm.y * h,
            z=lm.z,
            visibility=float(getattr(lm, "visibility", 0.0) or 0.0),
            presence=float(getattr(lm, "presence", 0.0) or 0.0),
        )
        for lm in subject
    ]
    others = []
    for other in result.pose_landmarks[1:]:
        xs = [lm.x * w for lm in other]
        ys = [lm.y * h for lm in other]
        others.append((min(xs), min(ys), max(xs), max(ys)))

    return PoseResult(landmarks=lms, width=w, height=h,
                      n_poses_detected=len(result.pose_landmarks),
                      other_pose_bboxes=others)


@atexit.register
def _close_landmarkers() -> None:
    """Close cached landmarkers before interpreter teardown.

    mediapipe's PoseLandmarker.__del__ calls into module globals that Python
    has already torn down by then, raising a confusing
    "TypeError: 'NoneType' object is not callable" after every script and test
    run. Closing them here means __del__ finds nothing left to do.
    """
    for lm in _LIVE_LANDMARKERS:
        try:
            lm.close()
        except Exception:
            pass
    _LIVE_LANDMARKERS.clear()
    _get_landmarker.cache_clear()


def load_rgb(path: str) -> np.ndarray:
    """Read an image file into an HxWx3 uint8 RGB array."""
    from PIL import Image

    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def body_scale(pose: PoseResult) -> float:
    """A length to normalise pixel distances by.

    Shoulder-midpoint to ankle-midpoint, which is robust to the subject's
    distance from the camera and does not depend on the head landmarks (which
    are the least reliable ones). Falls back to shoulder-to-hip if the ankles
    are missing, scaled by the population ratio between the two spans.
    """
    from .geometry import distance, midpoint

    sh = midpoint(pose.xy(LEFT_SHOULDER), pose.xy(RIGHT_SHOULDER))
    hip = midpoint(pose.xy(LEFT_HIP), pose.xy(RIGHT_HIP))
    ank = midpoint(pose.xy(LEFT_ANKLE), pose.xy(RIGHT_ANKLE))

    d = distance(sh, ank)
    if d > 1.0:
        return d
    # Shoulder-to-ankle is ~2.6x shoulder-to-hip in a median adult.
    # provenance: geometric-estimate (segment proportions)
    return distance(sh, hip) * 2.6
