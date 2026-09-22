"""Turn measurements into reference readings, with their limits attached.

Wording rule for anything a user sees: this tool reports 参考 (reference) and
倾向 (tendency). It never 诊断 (diagnoses). It is not a medical device.

One behaviour worth calling out: a verdict is only reported as settled when
the measurement's uncertainty interval sits entirely inside one band. When the
reading is within its own error bar of a threshold, the band is reported as
undetermined rather than rounded to whichever side it happens to land on. With
the measured landmark noise this happens often, especially for `forward_head`,
whose short ear-shoulder span makes it the least precise reading. That is the
intended behaviour: a tool that resolves a coin-flip into a confident verdict
is exactly the failure this project is trying to remove.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import guards as G
from . import landmarks as L
from .metrics import Measurement, NoiseModel, compute_all
from .thresholds import (BAND_LABELS_ZH, BAND_NOTABLE, BAND_REFERENCE,
                         BAND_SLIGHT, DIAGNOSTIC_ONLY, ThresholdSpec,
                         load_thresholds)
from .view import ViewEstimate, estimate_view

# Improvement suggestions. These are general movement-and-ergonomics advice of
# the kind that is safe for anyone, deliberately not tailored to a "condition"
# the tool cannot establish. Each is phrased as something to try, not a
# prescription.
ADVICE_ZH: dict[str, list[str]] = {
    "head_over_hip": [
        "把屏幕垫高到视线水平，减少长时间低头。",
        "每坐 30-45 分钟起身活动一次，做几次收下巴（下巴水平后收）的动作。",
        "留意枕头高度：过高会让颈部整夜保持前屈。",
    ],
    "shoulder_protraction": [
        "加入一些肩胛后缩的练习，例如靠墙天使、俯身划船。",
        "放松胸前肌群，例如门框处的胸部伸展。",
        "调整键盘位置，避免长时间双臂前伸。",
    ],
    "trunk_sway": [
        "站立时把重心放在双脚中段，避免长期靠向前脚掌或后跟。",
        "如果长期单侧负重（背单肩包、抱孩子），尝试两侧交替。",
    ],
    "shoulder_tilt": [
        "注意日常是否总用同一侧肩膀背包或提重物。",
        "两侧交替进行单侧负重的活动。",
        "先确认拍照时相机是水平的——相机倾斜会 1:1 叠加到这一项上。",
    ],
    "pelvis_tilt": [
        "留意站立时是否习惯把重心压在一条腿上。",
        "久坐时避免总是翘同一条腿。",
    ],
    "lateral_head_shift": [
        "检查屏幕是否摆在正前方而非偏向一侧。",
        "注意用耳朵夹手机接电话这类习惯。",
        "如果拍照时身体略有转向，这一项会偏大，建议正对镜头重拍一张对照。",
    ],
}

DISCLAIMER_ZH = (
    "本页面给出的是照片测量得到的参考读数与倾向描述，不是医学诊断，"
    "也不能替代专业评估。读数受拍摄角度、相机倾斜、衣物遮挡影响较大，"
    "请结合下方的不确定度与提示一并阅读。"
)


@dataclass
class MetricVerdict:
    measurement: Measurement
    band: str
    band_label_zh: str
    spec: ThresholdSpec
    resolved: bool           # uncertainty interval sits inside a single band
    alternative_band: str | None  # the other band it could be, when unresolved
    below_noise_floor: bool = False
    advice_zh: list[str] = field(default_factory=list)

    @property
    def display_band_zh(self) -> str:
        if self.below_noise_floor:
            return ("测量精度不足以判定"
                    f"（不确定度 ±{self.measurement.uncertainty:.1f}° "
                    f"已超过「{BAND_LABELS_ZH[BAND_SLIGHT]}」整档的宽度 "
                    f"{self.spec.notable - self.spec.slight:.1f}°）")
        if self.resolved:
            return self.band_label_zh
        other = BAND_LABELS_ZH.get(self.alternative_band or "", "")
        return f"{self.band_label_zh} / {other}（读数落在阈值的不确定度范围内，无法区分）"


def _below_noise_floor(m: Measurement, spec: ThresholdSpec) -> bool:
    """True when this metric cannot resolve its own middle band.

    The bands are reference / slight / notable. The narrowest of them is
    `slight`, of width (notable - slight). If the measurement uncertainty is
    at least that wide, no reading can ever be placed in `slight` with
    confidence -- the scale is finer than the instrument.

    This is reported explicitly rather than as an undetermined two-way band,
    because "it could be either of these" invites the reader to pick one,
    whereas "this metric cannot be resolved at this precision" does not.
    With the measured landmark noise this fires for `forward_head`, whose
    +/-11 deg swamps its 8 deg middle band.
    """
    if m.uncertainty != m.uncertainty:  # NaN
        return False
    width = spec.notable - spec.slight
    return width > 0 and m.uncertainty >= width


@dataclass
class Assessment:
    ok: bool
    view: ViewEstimate | None
    measurements: dict[str, Measurement]
    verdicts: dict[str, MetricVerdict]
    findings: list[G.GuardFinding]
    blocked: bool
    noise: NoiseModel
    # metric key -> landmark names that are outside the image, so the metric
    # was measured from an extrapolated point and is withheld
    unavailable: dict[str, list[str]] = field(default_factory=dict)
    error_zh: str = ""

    @property
    def disclaimer_zh(self) -> str:
        return DISCLAIMER_ZH


def _resolve_band(m: Measurement, spec: ThresholdSpec) -> tuple[str, bool, str | None]:
    """Classify, and report whether the error bar crosses a threshold.

    Uses a 1-sigma interval. Widening to 2-sigma would leave almost every
    forward_head reading undetermined, which is arguably the more honest
    choice; 1-sigma is used so the tool still says something, and the interval
    itself is always displayed next to the verdict.
    """
    band = spec.classify(m.value)
    if m.uncertainty != m.uncertainty:  # NaN
        return band, False, None

    lo = spec.classify(m.value - m.uncertainty)
    hi = spec.classify(m.value + m.uncertainty)
    if lo == hi == band:
        return band, True, None
    # Report the most severe band it could be, as the alternative.
    order = {BAND_REFERENCE: 0, BAND_SLIGHT: 1, BAND_NOTABLE: 2}
    alt = max({lo, hi} - {band}, key=lambda b: order[b], default=None)
    return band, False, alt


def assess(image_rgb: np.ndarray, *, deep_guards: bool = True,
           model_path: str | None = None) -> Assessment:
    """Full pipeline: detect -> view -> measure -> guard -> judge."""
    noise = NoiseModel.load()

    try:
        pose = L.detect(image_rgb, model_path=model_path)
    except L.LandmarkerUnavailable as exc:
        return Assessment(ok=False, view=None, measurements={}, verdicts={},
                          findings=[], blocked=True, noise=noise,
                          error_zh=f"姿态模型不可用：{exc}")

    if pose is None:
        return Assessment(
            ok=False, view=None, measurements={}, verdicts={}, findings=[],
            blocked=True, noise=noise,
            error_zh="没有在照片中检测到人体。请上传一张完整的全身照。")

    view = estimate_view(pose)
    measurements = compute_all(pose, view, noise)
    findings = G.run_all(pose, view, measurements, image_rgb, deep=deep_guards)
    is_blocked = G.blocked(findings)

    # A cropped photo is not a whole-photo failure. Withhold only the metrics
    # that actually rest on a landmark outside the image, and report the rest.
    unavailable = G.unavailable_metrics(pose, measurements)

    specs = load_thresholds()
    judged = [k for k in measurements
              if k not in DIAGNOSTIC_ONLY and k in specs]
    if judged and all(k in unavailable for k in judged):
        # Nothing survives; now it IS a whole-photo failure.
        is_blocked = True
        findings.append(G.GuardFinding(
            key="all_metrics_unavailable", severity=G.SEVERITY_BLOCK,
            message_zh="这张照片里可用的关键点不足以支持任何一项判定，"
                       "通常是人物被裁切得太多。请上传更完整的照片。",
            detail={"unavailable": unavailable}))

    verdicts: dict[str, MetricVerdict] = {}
    if not is_blocked:
        for key, m in measurements.items():
            if key in DIAGNOSTIC_ONLY or key in unavailable:
                continue
            spec = specs.get(key)
            if spec is None:
                continue
            band, resolved, alt = _resolve_band(m, spec)
            floored = _below_noise_floor(m, spec)
            # No advice from a reading the instrument cannot resolve, and none
            # from a reading that sits in the reference band.
            advice = ([] if (floored or band == BAND_REFERENCE)
                      else ADVICE_ZH.get(key, []))
            verdicts[key] = MetricVerdict(
                measurement=m, band=band, band_label_zh=BAND_LABELS_ZH[band],
                spec=spec, resolved=resolved and not floored,
                alternative_band=alt, below_noise_floor=floored,
                advice_zh=advice)

    return Assessment(ok=not is_blocked, view=view, measurements=measurements,
                      verdicts=verdicts, findings=findings, blocked=is_blocked,
                      noise=noise, unavailable=unavailable)


def assess_file(path: str, **kw) -> Assessment:
    return assess(L.load_rgb(path), **kw)
