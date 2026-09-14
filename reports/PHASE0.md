# 阶段 0：环境探测报告

执行环境：Ubuntu 24.04.4 LTS / x86_64 / Python 3.11.15 / 4 vCPU / 16 GB RAM
执行日期：2026-09-14
出口网络：经 Anthropic agent proxy（策略化 egress），TLS 在代理处重新终结。

## 结论速览

| # | 项目 | 结果 | 说明 |
|---|------|------|------|
| 1 | 网络出口（pypi / storage.googleapis / github） | **部分通过** | pypi ✅、GCS ✅、`github.com` 与 `api.github.com` ❌ 403 被策略拦截，但 `raw.githubusercontent.com` ✅ |
| 2 | `pip install mediapipe opencv-python-headless flask` | **通过** | 另需补装系统库，见下 |
| 3 | 下载 Pose 模型权重并核对大小 | **通过** | 三个变体全部下载并校验 sha256 |
| 4 | 真实人像推理，输出 33 个关键点 | **通过** | 见坐标样本 |
| 5 | 获取 ≥5 张全身站姿测试图（正面+侧面） | **失败** | 实际只拿到 2 张全身图，**侧面图 0 张** |

**4 项通过 / 1 项失败。** 按任务书「四项以上通过才进入阶段 1」的门槛，已进入阶段 1；
同时按任务书对第 5 项的预案「若无法自动获取，明确告知，我会手动上传」处理。

## 1. 网络出口

```
https://pypi.org/simple/                  -> 200
https://files.pythonhosted.org/           -> 200
https://storage.googleapis.com/…task      -> 200（5,777,746 bytes 实际下载成功）
https://raw.githubusercontent.com/…       -> 200
https://github.com/google-ai-edge/mediapipe    -> 403（CONNECT 被 egress 策略拒绝）
https://api.github.com/repos/…                 -> 403（同上）
```

两点初读时的误判，已更正：

- `https://storage.googleapis.com/` 根路径返回 400，内容是
  `<Code>MissingSecurityHeader</Code>`——那是 GCS 拒绝匿名列举 bucket 根，
  **不是**网络被拦。实际对象下载 200 且字节数正确。
- `https://github.com/` 返回 400 `Request path could not be canonicalized`，
  来自代理的路径规范化层。真正的仓库路径返回的是 403（策略拦截）。

GitHub 的 HTML/API 面被策略拦截属预期：本会话的 GitHub 访问走 MCP 通道。
模型与测试资源全部可从 pypi + GCS + raw.githubusercontent 获取，不影响后续步骤。

## 2. 依赖安装

```
Successfully installed mediapipe-1.0.1 opencv-python-headless-5.0.0.93
flask-3.1.3 numpy-2.4.6 …（共 27 个包）
```

**需要补装系统库**，否则 import 成功但创建 landmarker 时崩溃。mediapipe 的 wheel
即使只做 CPU 图像推理也链接了 EGL/GLES，headless 容器默认没有：

```
OSError: libEGL.so.1: cannot open shared object file
OSError: libGLESv2.so.2: cannot open shared object file
```

修复：`apt-get install libegl1 libgl1 libgles2`（已写入 `scripts/setup.sh`）。
注意需先 `apt-get update`——镜像里的 apt 索引是过期的，直接装会 404。

## 3. 模型权重

来源：`https://storage.googleapis.com/mediapipe-models/pose_landmarker/…`（Google 官方发布）

| 变体 | 字节数 | sha256（前 16） |
|------|--------|------------------|
| lite | 5,777,746 | `59929e1d1ee95287` |
| full | 9,398,198 | `5134a3aad27a58b9` |
| heavy | 30,664,242 | `64437af838a65d18` |

大小与官方公布一致；文件为 zip 容器（TFLite bundle），可正常解析。
`scripts/fetch_models.py` 已把上述 sha256 固化，重下会校验。

## 4. 真实人像推理

输入：`testdata/pose.jpg`（1000×667，真实人像照片）
模型：`pose_landmarker_full.task`

输出 **1 个人体、33 个关键点**，节选（像素坐标）：

```
11 left_shoulder    px=( 544.6, 321.4)  z=-0.1085  vis=1.000
12 right_shoulder   px=( 449.7, 325.0)  z=-0.0210  vis=1.000
23 left_hip         px=( 522.1, 471.9)  z=-0.0455  vis=0.997
24 right_hip        px=( 464.5, 465.9)  z=+0.0454  vis=0.999
27 left_ankle       px=( 701.2, 616.1)  z=-0.0572  vis=0.993
28 right_ankle      px=( 351.2, 607.0)  z=-0.0035  vis=0.987
32 right_foot_index px=( 309.0, 632.7)  z=-0.0679  vis=0.960
```

同时返回 33 个 `pose_world_landmarks`（以髋中心为原点的米制 3D 坐标）。

附带发现：mediapipe 1.0.1 **移除了旧的 `mp.solutions.*` API**。
`mp.solutions.pose.PoseLandmark` 已不存在，关键点名称需自行维护
（见 `posture/landmarks.py`）。网上大量教程仍在用旧 API，会直接报错。

## 5. 测试图获取 —— 未通过

外部图源全部被 egress 策略拦截：

```
images.cocodataset.org      -> CONNECT tunnel failed, 403
upload.wikimedia.org        -> CONNECT tunnel failed, 403
images.pexels.com           -> CONNECT tunnel failed, 403
images.unsplash.com         -> CONNECT tunnel failed, 403
commons.wikimedia.org       -> CONNECT tunnel failed, 403
```

按代理文档，403 属组织 egress 策略拒绝，不应绕行，故如实上报。

可达的替代源只有 Google 自己的 `mediapipe-assets` public bucket
（可匿名列举，共 916 个对象、68 张根级图片）与 `raw.githubusercontent.com`。
把其中所有疑似人像的图片全部下载并逐张跑检测后，结果是：

| 文件 | 尺寸 | 实际内容 | 可用性 |
|------|------|----------|--------|
| `male_full_height_hands.jpg` | 638×1000 | 全身、**正面**站姿；戴帽+口罩 | 勉强可用 |
| `pose.jpg` | 1000×667 | 全身，但是**瑜伽战士二式大弓步** | 非中立站姿 |
| `business-person.png` | 958×1358 | 躯干，踝部出画（vis 0.02） | 不可用 |
| `portrait.jpg` | 820×1024 | 头肩特写 | 不可用 |
| `test.jpg` | 3456×5184 | **一只手的特写照片**（非人体） | 不可用 |
| `sergey.png` | 600×600 | 头肩特写 | 不可用 |
| `man-woman-okay.jpg` | 640×426 | 半身 | 不可用 |
| `hand-woman-man.jpg` | 640×317 | 检测不到人体 | 不可用 |
| `woman_hands.jpg` | 640×960 | 检测不到人体 | 不可用 |

**要点：全身图只有 2 张，真正的侧面（矢状面）图 0 张。**

这直接决定了阶段 2 的结论：任务书四项指标里，
「头前引角」「圆肩」「骨盆倾斜」三项**都要求侧面照**，
因此这三项在现有素材上**一次都没有被真实验证过**。

### 需要你做什么

请手动上传 **5–10 张全身照**，其中：

- **侧面照 ≥3 张**（最关键，三项矢状面指标全靠它）：正对侧方 90°，
  手臂自然下垂或轻抱于胸前以免遮挡躯干轮廓，**耳部不要被头发/帽子遮住**。
- **正面照 ≥2 张**：用于高低肩。
- 通用要求：全身入镜（头顶到脚底都在画面内）、相机高度约在髋部、
  镜头尽量与身体保持水平（避免俯仰）、背景简单、光线均匀、
  贴身或轮廓清晰的衣着。
- 若能同时提供「同一个人、同一次拍摄、相机不动」的多张照片，
  就能额外检验重复性（repeatability），这是目前完全缺失的一环。

放到 `testdata/` 下，然后运行 `./.venv/bin/python scripts/validate.py --save-overlays`。
