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
    "forward_head": [
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
        "站立时把重心放在双脚中段，避免整天靠向一侧或前脚掌。",
        "如果长期单侧负重（背单肩包、抱孩子），尝试两侧交替。",
    ],
    "knee_deviation": [
        "站立时保持膝关节微微解锁，不要把膝盖向后顶死。",
        "加强大腿前后侧力量有助于膝关节在中立位稳定。",
    ],
    "shoulder_tilt": [
        "注意日常是否总用同一侧肩膀背包或提重物。",
        "两侧交替进行单侧负重的活动。",
    ],
    "pelvis_tilt": [
        "留意站立时是否习惯把重心压在一条腿上。",
        "久坐时避免总是翘同一条腿。",
    ],
    "head_tilt": [
        "注意用耳朵夹手机接电话这类习惯。",
        "检查屏幕是否摆在正前方而非偏向一侧。",
    ],
    "head_vs_shoulder_tilt": [
        "这一项排除了相机倾斜的影响，如果只有它偏大，更可能是真实的头部侧倾。",
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
    advice_zh: list[str] = field(default_factory=list)

    @property
    def display_band_zh(self) -> str:
        if self.resolved:
            return self.band_label_zh
        other = BAND_LABELS_ZH.get(self.alternative_band or "", "")
        return f"{self.band_label_zh} / {other}（读数落在阈值的不确定度范围内，无法区分）"


@dataclass
class Assessment:
    ok: bool
    view: ViewEstimate | None
    measurements: dict[str, Measurement]
    verdicts: dict[str, MetricVerdict]
    findings: list[G.GuardFinding]
    blocked: bool
    noise: NoiseModel
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

    specs = load_thresholds()
    verdicts: dict[str, MetricVerdict] = {}
    if not is_blocked:
        for key, m in measurements.items():
            if key in DIAGNOSTIC_ONLY:
                continue
            spec = specs.get(key)
            if spec is None:
                continue
            band, resolved, alt = _resolve_band(m, spec)
            verdicts[key] = MetricVerdict(
                measurement=m, band=band, band_label_zh=BAND_LABELS_ZH[band],
                spec=spec, resolved=resolved, alternative_band=alt,
                advice_zh=ADVICE_ZH.get(key, []) if band != BAND_REFERENCE else [])

    return Assessment(ok=not is_blocked, view=view, measurements=measurements,
                      verdicts=verdicts, findings=findings, blocked=is_blocked,
                      noise=noise)


def assess_file(path: str, **kw) -> Assessment:
    return assess(L.load_rgb(path), **kw)
