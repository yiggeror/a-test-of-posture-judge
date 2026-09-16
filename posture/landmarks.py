"""MediaPipe Pose Landmarker landmark indices and helpers.

MediaPipe Pose (BlazePose GHUM) emits 33 landmarks. Index order is fixed by the
model and documented at
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""
from __future__ import annotations

from dataclasses import dataclass

NAMES: list[str] = [
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

IDX: dict[str, int] = {n: i for i, n in enumerate(NAMES)}

NOSE = IDX["nose"]
LEFT_EAR, RIGHT_EAR = IDX["left_ear"], IDX["right_ear"]
LEFT_SHOULDER, RIGHT_SHOULDER = IDX["left_shoulder"], IDX["right_shoulder"]
LEFT_HIP, RIGHT_HIP = IDX["left_hip"], IDX["right_hip"]
LEFT_KNEE, RIGHT_KNEE = IDX["left_knee"], IDX["right_knee"]
LEFT_ANKLE, RIGHT_ANKLE = IDX["left_ankle"], IDX["right_ankle"]

# Bone pairs used only for drawing the overlay skeleton.
SKELETON: list[tuple[int, int]] = [
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, IDX["left_elbow"]), (IDX["left_elbow"], IDX["left_wrist"]),
    (RIGHT_SHOULDER, IDX["right_elbow"]), (IDX["right_elbow"], IDX["right_wrist"]),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, IDX["left_heel"]), (IDX["left_heel"], IDX["left_foot_index"]),
    (RIGHT_ANKLE, IDX["right_heel"]), (IDX["right_heel"], IDX["right_foot_index"]),
    (NOSE, LEFT_EAR), (NOSE, RIGHT_EAR),
]


@dataclass(frozen=True)
class Pt:
    """A landmark in PIXEL coordinates.

    Pixel space matters: MediaPipe returns x normalised by image WIDTH and y by
    image HEIGHT. Computing an angle straight from those normalised values
    silently applies the image aspect ratio to every angle, so every geometric
    quantity in this project is derived from pixels, never from raw normalised
    coordinates.

    y grows DOWNWARD (standard image convention).
    """

    x: float
    y: float
    visibility: float
    presence: float = 1.0

    @property
    def ok(self) -> bool:
        return self.visibility >= 0.5


@dataclass(frozen=True)
class Pt3:
    """A world landmark: metric 3D, in metres, origin at the hip midpoint.

    MediaPipe emits these alongside the image landmarks. The z axis is the
    least reliable output of a single-image model, so it is used here only as a
    coarse GATE (which view is this?), never as a measurement.
    """

    x: float
    y: float
    z: float
    visibility: float
