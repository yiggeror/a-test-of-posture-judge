# 姿态评估网页 Demo

上传一张全身照（侧面优先），叠加 33 个骨骼关键点，给出四项姿态**参考读数**。

> **这是技术 Demo，不是医疗器械，不提供诊断。**
> 所有阈值都是演示默认值，未经任何标注数据集校准。
> 三项矢状面指标目前**没有任何真实照片验证**（见下）。

---

## 先读这个：现在能信什么

| 部分 | 状态 |
|------|------|
| 关键点提取（MediaPipe，构图合规的真实照片） | 可信 |
| 几何运算（角度、归一化、符号） | 可信，37 项单元测试解析验证 |
| 视角自动判定（躯干偏航角） | 真实数据验证，n=12，两类间余量宽 |
| 指标方向性（符号对不对） | 有证据：头后仰→负值，整体前倾→铅垂偏移大 |
| 高低肩 | n=4 真实正面，量级合理 |
| 头前引角 / 圆肩 / 矢状对齐 | **n=2**，只够验证代码与符号，不够校准 |
| 全部阈值 | **仍未校准**，逐项标注了依据等级 |
| 假阳性防护 | 负样本泄漏 49%→29%；对人像雕塑与肢体特写**无效** |

详见 [`reports/PHASE0.md`](reports/PHASE0.md)（环境探测）与
[`reports/PHASE2.md`](reports/PHASE2.md)（验证，含失败案例）。

**换环境接手请先读 [`reports/HANDOFF.md`](reports/HANDOFF.md)。**

第二轮真实数据的处理与发现见 [`reports/PHASE3.md`](reports/PHASE3.md)
（含两例已证实的假读数：头发遮耳、小腿特写）。

**要补齐验证缺口，见 [`reports/IMAGE_BRIEF.md`](reports/IMAGE_BRIEF.md)** ——
可直接交给采集 agent 的图片需求书。采集方无需访问本仓库；
图片收回来后在本地跑 `scripts/triage_images.py` 筛查（退出码 0 = 数据达标）。

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
| 头前引角 | 肩峰→耳连线相对铅垂线的夹角 | 侧面 | 几何推算（读数带 ±，见下） |
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
- **MediaPipe 的 `visibility` 不是「这是不是人」**，也不是「这个点位置对不对」。
  已证实两例：一张**手掌照片**和一张**穿袜小腿特写**都被贴上 vis≈1.0 的完整人体骨架，
  后者通过了全部几何合理性检查（详见 `reports/PHASE2.md`、`PHASE3.md`）。
  **直立的人像雕塑、人体模特与玩偶同样无法用几何手段排除**——需要真正的分类器。
- **耳部遮挡会安静地失真，已证实**：长发盖住耳朵时 ear 关键点落在头发上、
  visibility 仍报 1.000，应用据此输出「头前引角 +25.39°，明显倾向」。两项侧面指标全靠耳点。
- **读数精度低于阈值分档**：耳点平移 10px（1400px 高的图）→ 头前引角摆动 ±4.4°，
  而分档间距只有 10°。界面已给出 ± 并说明不要按小数位解读。
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
scripts/triage_images.py  候选图自动筛查（ACCEPT/REJECT + 达标统计）
scripts/fetch_from_manifest.py  按采集方的 sources.csv 批量下载候选图
scripts/validate.py       阶段 2 验证脚本
tests/test_geometry.py    21 项几何单元测试
reports/                  阶段 0 / 阶段 2 报告
```

## 许可

模型：MediaPipe Pose Landmarker（BlazePose GHUM），**Apache-2.0**。
未使用 YOLO-pose（AGPL-3.0）或 OpenPose（学术许可）。详见 [`LICENSES.md`](LICENSES.md)。
