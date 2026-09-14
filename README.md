# 姿态评估网页 Demo

上传一张全身照（侧面优先），叠加 33 个骨骼关键点，给出四项姿态**参考读数**。

> **这是技术 Demo，不是医疗器械，不提供诊断。**
> 所有阈值都是演示默认值，未经任何标注数据集校准。
> 三项矢状面指标目前**没有任何真实照片验证**（见下）。

---

## 先读这个：现在能信什么

| 部分 | 状态 |
|------|------|
| 关键点提取（MediaPipe，真实照片） | 可信 |
| 几何运算（角度、归一化、符号） | 可信，21 项单元测试解析验证 |
| 高低肩在全身正面图上的表现 | n=1，弱证据 |
| 头前引角 / 圆肩 / 矢状对齐 | **未验证**——一张侧面测试图都没有 |
| 全部阈值 | 未校准，逐项标注了依据等级 |
| 视角自动判定 | 启发式，侧面分支从未被真实图触发 |

详见 [`reports/PHASE0.md`](reports/PHASE0.md)（环境探测）与
[`reports/PHASE2.md`](reports/PHASE2.md)（验证，含失败案例）。

## 安装运行

```bash
./scripts/setup.sh          # 系统库 + venv + 依赖 + 模型 + 测试
./.venv/bin/python app.py   # http://127.0.0.1:5000
```

`setup.sh` 里的 apt 步骤不能省：mediapipe 的 wheel 即便只做 CPU 推理也链接
EGL/GLES，headless 容器缺 `libEGL.so.1` / `libGLESv2.so.2` 会在创建
landmarker 时崩溃。

```bash
./.venv/bin/python -m pytest tests/ -q                        # 单元测试
./.venv/bin/python scripts/validate.py --save-overlays        # 验证
./.venv/bin/python scripts/fetch_testdata.py                  # 取测试图
```

## 四项指标

所有几何都在**像素坐标**下计算。MediaPipe 的 x 按图宽归一、y 按图高归一，
直接用归一化坐标算角度会把图像长宽比混进每一个角度——`posture/engine.py`
在唯一的入口处换算成像素，`tests/test_geometry.py` 有对应回归测试。

| 指标 | 定义 | 需要 | 阈值依据 |
|------|------|------|----------|
| 头前引角 | 肩峰→耳连线相对铅垂线的夹角 | 侧面 | 几何推算 |
| 圆肩 | 肩峰相对耳垂的水平偏移 ÷ 躯干长 | 侧面 | **纯猜测** |
| 躯干矢状面对齐 | 180° − 肩-髋-踝夹角 | 侧面 | **纯猜测** |
| 高低肩 | 左右肩峰连线相对水平线的倾角 | 正面 | 几何推算 |
| 铅垂线偏移（补充） | 耳/肩/髋相对踝部铅垂线的前后偏移 | 侧面 | 无阈值，只给数 |

阈值常量集中在 [`posture/thresholds.py`](posture/thresholds.py)，
每个都带 `basis` 字段（`guess` / `geometric-estimate` / `literature-adjacent`）
并在 UI 上原样展示。

### 两个必须说清楚的定义问题

**「骨盆倾斜」名不副实。** 真正的骨盆前后倾由 ASIS-PSIS 连线定义，
MediaPipe **不提供这两个关键点**，每侧只给一个大转子附近的髋点。
所以本项目实现的是任务书字面定义的「肩-髋-踝三点矢状面对齐偏差」，
并在代码和界面里统一改名为「躯干矢状面对齐」。真正的骨盆倾斜用这套关键点测不了。

**头前引和圆肩不是两个独立指标。** 两者都只用「耳」和「肩峰」两个点：
一个算夹角、一个算水平偏移，沿前后轴互为反号。用这两个点无法区分
「头更靠前」和「肩更靠前」。因此额外提供了 `plumb_offsets`——
以踝部铅垂线为共同基准（临床铅垂线惯例）分别度量耳和肩峰的前移，
那一组才是相互独立的。

## 已知限制

- **单张照片不可标定**，没有已知尺寸参照物，只能给归一化比值，给不出厘米。
- **相机姿态直接污染读数**：翻滚角 1:1 叠加到高低肩；俯仰与镜头畸变改变头前引角。单张照片无法分离。
- **MediaPipe 的 `visibility` 不是「这是不是人」**。它估计的是单点遮挡。
  验证中模型对一张**手掌照片**输出了 vis≈1.0 的完整骨架
  （详见 `reports/PHASE2.md`）。已加人体比例合理性检查兜底，但那不是人体检测器。
- **耳部遮挡会安静地失真**：头发/帽子/口罩盖住耳朵时，模型照样给出一个位置，
  且 visibility 可能依然很高。而两项侧面指标全靠耳点。
- **无重复性数据**：同一个人、同一姿势、不同照片之间的波动有多大，完全未知。

## 项目结构

```
app.py                    Flask 服务（/ 与 /api/analyze）
posture/landmarks.py      33 关键点表；Pt（像素坐标）
posture/thresholds.py     全部阈值常量 + 依据等级标注
posture/metrics.py        几何运算 + 视角判定 + 合理性检查
posture/engine.py         MediaPipe 封装（像素换算的唯一入口）
posture/render.py         骨架与测量构造线叠加
scripts/setup.sh          一键安装
scripts/fetch_models.py   下载权重并校验 sha256
scripts/fetch_testdata.py 下载测试图
scripts/validate.py       阶段 2 验证脚本
tests/test_geometry.py    21 项几何单元测试
reports/                  阶段 0 / 阶段 2 报告
```

## 许可

模型：MediaPipe Pose Landmarker（BlazePose GHUM），**Apache-2.0**。
未使用 YOLO-pose（AGPL-3.0）或 OpenPose（学术许可）。详见 [`LICENSES.md`](LICENSES.md)。
