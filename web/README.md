# web/ — 浏览器端移植

`posture.js` 是 `posture/` 这个 Python 包的逐行移植：视角判定、指标计算、
逐关键点的不确定度传播、守卫、阈值分档。**小程序要移植的就是这一份。**

## 为什么可以直接用

MediaPipe 的 JS 版（`@mediapipe/tasks-vision`）用的是同一个 BlazePose 33 点
模型，所以 `reports/landmark_noise.json` 的逐关键点噪声和
`reports/reference_distribution.json` 的分位数阈值**直接适用，不需要重新标定**。

换成别的姿态模型就不成立了——见 `reports/DEPLOYMENT.md`。

## 一致性是验证过的，不是声称的

`scripts/export_web_demo.py` 会：

1. 用 Python 版对示例照片跑检测，导出关键点 + 常量
2. 在 node 里用 `posture.js` 跑同一批关键点
3. 逐项比对视角、阻断、警告、停用指标、每个读数值、每个不确定度、每个档位

当前结果：**6 张样本、全部字段、0 差异**。改动 `posture.js` 或 Python 任一侧
之后都应该重跑一次。

比对用 `deep_guards=False`，因为浏览器端没有那两道需要额外模型推理的守卫
（独立人体检测、关键点稳定性）。这是有意的差异，demo 页面底部也写明了。

## 用法

```js
// 关键点来自 MediaPipe tasks-vision 的 PoseLandmarker
// P: 33 个 [x, y, visibility]，像素坐标
const V = estimateView(P, W, H);
const mets = computeAll(P, V);
const { findings, oof } = guards(P, V, mets, W, H, V.scale, otherBoxes, origScale);
```

`C` 是常量对象（噪声模型 + 阈值 + 建议文案），由导出脚本生成。

## 文件

| 文件 | 作用 |
|---|---|
| `posture.js` | **移植的核心逻辑**，小程序复用这个 |
| `demo-render.js` | 演示页的画布与刻度尺渲染 |
| `demo-styles.html` / `demo-gauge.css` / `demo-body.html` | 演示页的样式与结构 |

演示页面用 `scripts/export_web_demo.py` 组装。
