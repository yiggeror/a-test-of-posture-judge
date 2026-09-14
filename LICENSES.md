# 许可合规

硬性约束：只用 Apache-2.0 或 MIT 授权的模型；禁止 YOLO-pose（AGPL-3.0）
与 OpenPose（学术许可）。

## 实际使用

| 组件 | 版本 | 许可 | 用途 |
|------|------|------|------|
| MediaPipe（`mediapipe` PyPI 包） | 1.0.1 | **Apache-2.0** | 推理运行时 |
| MediaPipe Pose Landmarker 权重（BlazePose GHUM，lite/full/heavy） | float16/1 | **Apache-2.0** | 33 关键点模型 |
| OpenCV（`opencv-python-headless`） | 5.0.0.93 | **Apache-2.0** | 图像解码与绘制 |
| Flask | 3.1.3 | **BSD-3-Clause** | Web 服务 |
| NumPy | 2.4.6 | **BSD-3-Clause** | 数组运算 |

模型权重取自 Google 官方发布地址
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/…`，
随 MediaPipe 项目以 Apache-2.0 发布。

BSD-3-Clause（Flask / NumPy）属宽松许可，与 Apache-2.0/MIT 同类，
且约束针对的是「模型」，这两者是库不是模型。

## 明确未使用

- **YOLO-pose / Ultralytics YOLOv8-pose** —— AGPL-3.0，**已排除**。
- **OpenPose（CMU）** —— 非商业学术许可，**已排除**。

代码中无任何对上述两者的引用。核验：

```bash
grep -ri "yolo\|ultralytics\|openpose" --include=*.py --include=*.js \
  --include=*.html --include=*.txt . | grep -v LICENSES.md
```

## 测试图片

`testdata/` 里的图片来自 Google 公开的 `mediapipe-assets` GCS bucket
（MediaPipe 项目自用的测试资源）。这些图片**未随本仓库分发**——
`.gitignore` 已排除，由 `scripts/fetch_testdata.py` 现取。
如需对外分发，请自行确认每张图片的具体授权。
