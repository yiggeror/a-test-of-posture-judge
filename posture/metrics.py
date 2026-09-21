"""Posture measurements and their uncertainties.

Design note -- what changed from the original four-metric set
------------------------------------------------------------
The original design carried a "pelvic tilt" metric. It is not implemented here,
because it is not measurable from these landmarks. Anterior/posterior pelvic
tilt is defined clinically by the ASIS-PSIS line; BlazePose emits a single
approximate hip-JOINT centre per side and no pelvic landmarks at all. Any
number reported under that name would have been a re-scaled hip-position
reading wearing a clinical label. It is replaced by `trunk_sway`, which is
what the hip landmark can actually support, and the gap is stated in the
README rather than papered over.

Metric independence
-------------------
`forward_head` (ear vs shoulder) and `shoulder_protraction` (shoulder vs hip)
share the shoulder landmark, and share it with opposite sign: a shoulder
placed too far anterior *decreases* forward_head and *increases*
shoulder_protraction. They are therefore negatively coupled through their own
measurement error, not just through real anatomy. `head_over_hip` is reported
alongside them precisely because it skips the shoulder entirely, so when the
two disagree it identifies which landmark is responsible.

Uncertainty
-----------
Each angle is a two-point measurement, so a first-order propagation is exact
enough and far more honest than a fixed number. For two endpoints each with
isotropic positional noise sigma_px, separated by L pixels:

    sigma_angle = sqrt(2) * sigma_px / L   (radians)

The important consequence is structural: the ear-shoulder segment is ~25 cm
while the ankle-hip segment is ~90 cm, so identical landmark noise makes
`forward_head` roughly 3.6x noisier than `trunk_sway`. That ratio is a
property of the geometry and no amount of model improvement removes it.

sigma_px is not guessed -- it is measured by scripts/repeatability.py and
stored in reports/landmark_noise.json. Until that file exists the fallback
below is used and is tagged as a guess.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from . import landmarks as L
from .geometry import angle_from_horizontal, angle_from_vertical, distance, interior_angle, midpoint
from .view import ViewEstimate, near_side_indices

# Landmark positional noise as a fraction of body scale (shoulder-to-ankle).
# provenance: guess -- placeholder used only when reports/landmark_noise.json
# is absent. Run scripts/repeatability.py to replace it with a measured value.
FALLBACK_LANDMARK_NOISE_FRAC = 0.004

_NOISE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports", "landmark_noise.json")


@dataclass(frozen=True)
class Measurement:
    key: str
    label: str
    value: float          # degrees
    uncertainty: float    # +/- degrees, 1 sigma, from landmark noise propagation
    plane: str            # "sagittal" or "frontal"
    span_px: float        # length of the measured segment
    landmark_indices: tuple[int, ...] = ()
    note: str = ""

    def format(self) -> str:
        return f"{self.value:+.1f} +/- {self.uncertainty:.1f} deg"


@dataclass
class NoiseModel:
    """Landmark localisation noise, expressed scale-free."""
    frac_of_body_scale: float
    provenance: str
    source: str = ""
    detail: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | None = None) -> "NoiseModel":
        path = path or _NOISE_PATH
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    d = json.load(fh)
                return cls(frac_of_body_scale=float(d["frac_of_body_scale"]),
                           provenance=d.get("provenance", "measured"),
                           source=d.get("source", path),
                           detail=d.get("per_landmark", {}))
            except (KeyError, ValueError, OSError):
                pass
        return cls(frac_of_body_scale=FALLBACK_LANDMARK_NOISE_FRAC,
                   provenance="guess",
                   source="posture/metrics.py fallback constant")

    def sigma_px(self, body_scale: float) -> float:
        return self.frac_of_body_scale * body_scale


def _angle_sigma(sigma_px: float, span_px: float) -> float:
    """First-order angular uncertainty for a two-point angle, in degrees."""
    if span_px <= 1e-6:
        return float("nan")
    return math.degrees(math.sqrt(2.0) * sigma_px / span_px)


def compute_sagittal(pose: L.PoseResult, view: ViewEstimate,
                     noise: NoiseModel | None = None) -> dict[str, Measurement]:
    """Sagittal-plane metrics. Only meaningful for a lateral view.

    Positive always means "displaced toward the subject's front".
    """
    noise = noise or NoiseModel.load()
    scale = L.body_scale(pose)
    sigma = noise.sigma_px(scale)
    side = near_side_indices(pose, view.facing)
    ant = view.facing

    ear = pose.xy(side["ear"])
    sh = pose.xy(side["shoulder"])
    hip = pose.xy(side["hip"])
    knee = pose.xy(side["knee"])
    ankle = pose.xy(side["ankle"])

    out: dict[str, Measurement] = {}

    span = distance(sh, ear)
    out["forward_head"] = Measurement(
        key="forward_head",
        label="Forward head (ear relative to shoulder)",
        value=angle_from_vertical(sh, ear, anterior=ant),
        uncertainty=_angle_sigma(sigma, span),
        plane="sagittal", span_px=span,
        landmark_indices=(side["shoulder"], side["ear"]),
        note="Proxy for forward head posture. NOT the clinical craniovertebral "
             "angle: CVA is measured from the C7 spinous process, which these "
             "landmarks do not include. The acromion sits anterior to C7, so "
             "this reading is systematically smaller than (90 - CVA).",
    )

    span = distance(hip, sh)
    out["shoulder_protraction"] = Measurement(
        key="shoulder_protraction",
        label="Shoulder protraction (shoulder relative to hip)",
        value=angle_from_vertical(hip, sh, anterior=ant),
        uncertainty=_angle_sigma(sigma, span),
        plane="sagittal", span_px=span,
        landmark_indices=(side["hip"], side["shoulder"]),
        note="Shares the shoulder landmark with forward_head, with opposite "
             "sign. Read the two together, and compare against head_over_hip.",
    )

    span = distance(ankle, hip)
    out["trunk_sway"] = Measurement(
        key="trunk_sway",
        label="Body lean (hip relative to ankle)",
        value=angle_from_vertical(ankle, hip, anterior=ant),
        uncertainty=_angle_sigma(sigma, span),
        plane="sagittal", span_px=span,
        landmark_indices=(side["ankle"], side["hip"]),
        note="Whole-body anterior/posterior lean. The longest segment measured "
             "here, so the most precise of the sagittal readings.",
    )

    span = distance(hip, ear)
    out["head_over_hip"] = Measurement(
        key="head_over_hip",
        label="Head over hip (shoulder-independent)",
        value=angle_from_vertical(hip, ear, anterior=ant),
        uncertainty=_angle_sigma(sigma, span),
        plane="sagittal", span_px=span,
        landmark_indices=(side["hip"], side["ear"]),
        note="Diagnostic. Does not use the shoulder landmark, so it separates "
             "genuine head position from shoulder mislocalisation.",
    )

    # Knee: signed deviation from a straight leg. The interior angle alone is
    # unsigned, so flexion and hyperextension are distinguished by which side
    # of the hip-ankle line the knee falls on.
    interior = interior_angle(hip, knee, ankle)
    leg_len = distance(hip, ankle)
    if leg_len > 1e-6 and not math.isnan(interior):
        cross = ((ankle[0] - hip[0]) * (knee[1] - hip[1])
                 - (ankle[1] - hip[1]) * (knee[0] - hip[0]))
        # cross > 0 means the knee is on the +x side in image coords; combine
        # with facing so positive = anterior = flexed.
        anterior_sign = 1.0 if (cross * ant) < 0 else -1.0
        knee_dev = (180.0 - interior) * anterior_sign
        # A knee deviation is an angle at a vertex between two segments of
        # roughly half-leg length each.
        span = leg_len / 2.0
        out["knee_deviation"] = Measurement(
            key="knee_deviation",
            label="Knee deviation (+ flexed / - hyperextended)",
            value=knee_dev,
            uncertainty=_angle_sigma(sigma, span) * math.sqrt(2.0),
            plane="sagittal", span_px=span,
            landmark_indices=(side["hip"], side["knee"], side["ankle"]),
            note="Also used as a guard: a flexed knee means the subject was "
                 "not standing neutrally, which invalidates the other readings.",
        )
    return out


def compute_frontal(pose: L.PoseResult,
                    noise: NoiseModel | None = None) -> dict[str, Measurement]:
    """Frontal-plane metrics. Only meaningful for a front (or rear) view.

    Positive means the subject's LEFT side is higher in the image. Note the
    subject's left appears on the viewer's right in a front-facing photo;
    the labels below are anatomical.
    """
    noise = noise or NoiseModel.load()
    scale = L.body_scale(pose)
    sigma = noise.sigma_px(scale)

    lsh, rsh = pose.xy(L.LEFT_SHOULDER), pose.xy(L.RIGHT_SHOULDER)
    lhip, rhip = pose.xy(L.LEFT_HIP), pose.xy(L.RIGHT_HIP)
    leye, reye = pose.xy(L.LEFT_EYE), pose.xy(L.RIGHT_EYE)

    out: dict[str, Measurement] = {}

    span = distance(rsh, lsh)
    out["shoulder_tilt"] = Measurement(
        key="shoulder_tilt",
        label="Shoulder levelness",
        value=angle_from_horizontal(rsh, lsh),
        uncertainty=_angle_sigma(sigma, span),
        plane="frontal", span_px=span,
        landmark_indices=(L.RIGHT_SHOULDER, L.LEFT_SHOULDER),
        note="Positive means the subject's left shoulder sits higher. Camera "
             "roll adds directly to this reading and cannot be separated from "
             "a real shoulder difference without a level reference in frame.",
    )

    span = distance(rhip, lhip)
    out["pelvis_tilt"] = Measurement(
        key="pelvis_tilt",
        label="Hip levelness",
        value=angle_from_horizontal(rhip, lhip),
        uncertainty=_angle_sigma(sigma, span),
        plane="frontal", span_px=span,
        landmark_indices=(L.RIGHT_HIP, L.LEFT_HIP),
        note="Lateral pelvic obliquity only. This is NOT anterior/posterior "
             "pelvic tilt, which these landmarks cannot measure at all.",
    )

    span = distance(reye, leye)
    out["head_tilt"] = Measurement(
        key="head_tilt",
        label="Head tilt",
        value=angle_from_horizontal(reye, leye),
        uncertainty=_angle_sigma(sigma, span),
        plane="frontal", span_px=span,
        landmark_indices=(L.RIGHT_EYE, L.LEFT_EYE),
        note="Measured across the eyes, a short span, so this is the noisiest "
             "frontal reading. Camera roll affects it identically to "
             "shoulder_tilt, so their DIFFERENCE is roll-invariant.",
    )

    # Lateral head shift: horizontal offset of the nose from the shoulder
    # midpoint, as an angle subtended at the hips. Roll-sensitive but useful.
    sh_mid = midpoint(lsh, rsh)
    hip_mid = midpoint(lhip, rhip)
    span = distance(hip_mid, sh_mid)
    if span > 1e-6:
        nose = pose.xy(L.NOSE)
        out["lateral_head_shift"] = Measurement(
            key="lateral_head_shift",
            label="Head lateral shift",
            value=math.degrees(math.atan2(nose[0] - sh_mid[0], span)),
            uncertainty=_angle_sigma(sigma, span),
            plane="frontal", span_px=span,
            landmark_indices=(L.NOSE, L.LEFT_SHOULDER, L.RIGHT_SHOULDER),
            note="Diagnostic. Positive means the head sits to the viewer's "
                 "right of the shoulder midline.",
        )

    # Roll-invariant combination: if the camera is rolled by r degrees, both
    # shoulder_tilt and head_tilt gain r, so the difference cancels it.
    if "shoulder_tilt" in out and "head_tilt" in out:
        st, ht = out["shoulder_tilt"], out["head_tilt"]
        out["head_vs_shoulder_tilt"] = Measurement(
            key="head_vs_shoulder_tilt",
            label="Head tilt relative to shoulders (roll-invariant)",
            value=ht.value - st.value,
            uncertainty=math.hypot(ht.uncertainty, st.uncertainty),
            plane="frontal", span_px=min(ht.span_px, st.span_px),
            landmark_indices=ht.landmark_indices + st.landmark_indices,
            note="Camera roll cancels in this difference, so unlike the two "
                 "readings it is built from, this one is not confounded by how "
                 "level the photographer held the camera.",
        )
    return out


def compute_all(pose: L.PoseResult, view: ViewEstimate,
                noise: NoiseModel | None = None) -> dict[str, Measurement]:
    """Compute whichever metrics the detected view can support."""
    noise = noise or NoiseModel.load()
    out: dict[str, Measurement] = {}
    if view.view in ("side", "oblique"):
        out.update(compute_sagittal(pose, view, noise))
    if view.view in ("front", "oblique"):
        out.update(compute_frontal(pose, noise))
    return out
