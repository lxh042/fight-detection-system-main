# Orange Pi AI Pro 部署教程

## 1. 适用范围

本教程只针对 `deploy/` 目录，也就是项目的最终部署成果。其目标是在 Orange Pi AI Pro 开发板上完成以下事项：

1. 安装依赖与准备运行环境。
2. 配置 ONNX Runtime、视频源与模型路径。
3. 构建并运行 C++ 推理程序。
4. 启动 Web 仪表盘进行可视化展示。
5. 在需要时切换到 `CANN EP` 使用 NPU。

## 2. 部署交付物说明

`deploy/` 目录中的关键内容如下：

| 路径 | 作用 |
| --- | --- |
| `deploy/models/version3.onnx` | 板端分类模型 |
| `deploy/yolo11n-pose.onnx` | 板端姿态模型 |
| `deploy/cpp_demo/` | 板端 C++ 推理引擎 |
| `deploy/install_deps_orangepi.sh` | 一键安装构建依赖与 ONNX Runtime |
| `deploy/dashboard.py` | Flask Web 仪表盘 |
| `deploy/stream_video.py` | 可选视频流模拟器 |
| `deploy/artifacts/` | 板端运行摘要与事件记录 |

## 3. 部署架构

```mermaid
flowchart LR
    A[摄像头 / RTSP / HTTP流 / 本地视频] --> B[deploy/cpp_demo C++引擎]
    B --> C[YOLO Pose ONNX]
    C --> D[分类 ONNX]
    D --> E[/dev/shm/preview.jpg]
    D --> F[/dev/shm/status.json]
    D --> G[artifacts/*.json / event_*.mp4]
    E --> H[deploy/dashboard.py]
    F --> H
    G --> H
    H --> I[浏览器演示页面]
```

## 4. 前置条件

### 4.1 硬件

1. Orange Pi AI Pro 开发板。
2. 稳定电源。
3. 可用网络连接。
4. 显示器、键盘、鼠标，或可用 SSH 登录方式。

### 4.2 软件

1. Ubuntu 22.04 LTS aarch64。
2. `build-essential`、`cmake`、`pkg-config`、`libopencv-dev`。
3. ONNX Runtime aarch64 版本。
4. 若使用 Web 面板，需安装 `python3-flask`。
5. 若使用 `stream_video.py`，建议安装 `python3-opencv`。

## 5. 上传部署目录

推荐把 `deploy/` 整个目录拷贝到开发板，例如：

```bash
scp -r deploy orangepi@<board-ip>:~/Desktop/
```

上传完成后的推荐目录：

```text
~/Desktop/deploy/
```

## 6. 安装依赖

进入部署目录后执行：

```bash
cd ~/Desktop/deploy
chmod +x install_deps_orangepi.sh
./install_deps_orangepi.sh
```

该脚本会自动执行：

1. `apt-get update`
2. 安装 `build-essential`、`cmake`、`pkg-config`、`libopencv-dev`、`wget`、`tar`
3. 下载并解压 `onnxruntime-linux-aarch64-1.24.3`

如果你计划使用面板或视频流模拟器，建议额外安装：

```bash
sudo apt-get install -y python3-flask python3-opencv
```

## 7. 配置环境文件

### 7.1 复制模板

```bash
cd ~/Desktop/deploy/cpp_demo/scripts
cp orangepi_aipro.env.example orangepi_aipro.env
```

### 7.2 修改关键变量

重点检查以下变量：

| 变量 | 说明 | 建议 |
| --- | --- | --- |
| `ONNXRUNTIME_ROOT` | ONNX Runtime 根目录 | 指向实际解压路径 |
| `ONNXRUNTIME_INCLUDE_DIR` | ONNX Runtime 头文件目录 | 通常用默认拼接即可 |
| `ONNXRUNTIME_LIBRARY` | `libonnxruntime.so` 路径 | 必须存在 |
| `VIDEO_PATH` | 本地视频路径 | 适用于文件模式 |
| `SOURCE_TYPE` | `video` 或 `camera` | 根据输入源选择 |
| `CAMERA_INDEX` | 摄像头索引或流地址 | 支持整数和 HTTP / RTSP 地址 |
| `FEATURE_SOURCE` | 特征来源 | 推荐 `yolo-onnx` |
| `POSE_MODEL_PATH` | 姿态模型路径 | 指向 `../yolo11n-pose.onnx` |
| `MODEL_PATH` | 分类模型路径 | 指向 `../models/version3.onnx` |
| `ONNX_EP` | 执行提供器 | 建议先 `cpu`，后 `cann` |
| `SUMMARY_JSON` | 摘要输出路径 | 建议保留 |

### 7.3 重要注意事项

1. 模板文件默认写的是 `onnxruntime-linux-aarch64-1.23.2`。
2. 安装脚本默认下载的是 `onnxruntime-linux-aarch64-1.24.3`。
3. 因此你必须按实际解压目录修改 `ONNXRUNTIME_ROOT`，不能直接照抄模板默认值。

一个更符合当前安装脚本结果的示例配置如下：

```bash
ONNXRUNTIME_ROOT=../../third_party/onnxruntime-linux-aarch64-1.24.3
ONNXRUNTIME_INCLUDE_DIR=${ONNXRUNTIME_ROOT}/include
ONNXRUNTIME_LIBRARY=${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
BUILD_DIR=build-orangepi-onnx
ENABLE_ONNX=ON
ENABLE_MINDSPORE=OFF
VIDEO_PATH=../fn.mp4
MODEL_PATH=../models/version3.onnx
POSE_MODEL_PATH=../yolo11n-pose.onnx
FEATURE_SOURCE=yolo-onnx
ONNX_EP=cpu
ONNX_DEVICE_ID=0
SUMMARY_JSON=../artifacts/cpp_demo_onnx_orangepi_summary.json
MAX_FRAMES=20
LOG_EVERY=10
```

## 8. 先做环境自检

在正式运行前，建议先执行：

```bash
cd ~/Desktop/deploy/cpp_demo
bash scripts/check_cann_env.sh scripts/orangepi_aipro.env
```

该脚本会检查：

1. `libonnxruntime.so` 是否存在。
2. `npu-smi` 是否可用。
3. Ascend / CANN 相关环境变量。
4. 当前 ONNX Runtime 是否真的导出 `CANN EP` 所需符号。

如果脚本提示当前库没有 `CANN EP` 符号，则必须先更换为带 CANN 执行提供器的 ONNX Runtime，再把 `ONNX_EP` 切到 `cann`。

## 9. 推荐运行顺序

### 9.1 阶段一：CPU 冒烟测试

先验证“工程能编译、模型能加载、参数能跑通”：

```bash
cd ~/Desktop/deploy/cpp_demo
ONNX_EP=cpu FEATURE_SOURCE=synthetic MAX_FRAMES=20 bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```

该模式不依赖真实姿态提取，适合快速排除编译和运行时错误。

### 9.2 阶段二：完整 ONNX 链路测试

切回真实姿态模型与分类模型：

```bash
cd ~/Desktop/deploy/cpp_demo
ONNX_EP=cpu FEATURE_SOURCE=yolo-onnx MAX_FRAMES=0 OUTPUT_VIDEO=../artifacts/result.mp4 EVENT_LOG=../artifacts/incidents.json bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```

该模式会：

1. 编译 `fight_detection_demo`
2. 加载姿态 ONNX 与分类 ONNX
3. 处理输入视频或视频流
4. 生成事件日志与摘要文件

### 9.3 阶段三：切换到 CANN EP / NPU

只有在你已经确认当前 ONNX Runtime 带有 `CANN EP` 时，才建议执行：

```bash
cd ~/Desktop/deploy/cpp_demo
ONNX_EP=cann ONNX_DEVICE_ID=0 MAX_FRAMES=0 bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```

若运行时报错提示 `SessionOptionsAppendExecutionProvider_CANN` 相关问题，说明当前 ONNX Runtime 仍不是 CANN 版本。

## 10. 输入源配置方式

### 10.1 本地视频文件

```bash
SOURCE_TYPE=video
VIDEO_PATH=../fn.mp4
```

### 10.2 USB 摄像头

```bash
SOURCE_TYPE=camera
CAMERA_INDEX=0
```

### 10.3 HTTP / RTSP 视频流

该工程支持把 `CAMERA_INDEX` 当作流地址使用，例如：

```bash
SOURCE_TYPE=camera
CAMERA_INDEX=http://<board-ip>:5555/video_feed
```

这是因为当前 `openCapture` 逻辑会优先判断 `CAMERA_INDEX` 是否为纯数字；若不是纯数字，则按网络流地址处理。

## 11. 启动 Web 仪表盘

在新的终端中执行：

```bash
cd ~/Desktop/deploy
python3 dashboard.py
```

浏览器访问：

```text
http://<board-ip>:5000
```

仪表盘读取的关键文件为：

- `/dev/shm/preview.jpg`
- `/dev/shm/status.json`

因此只有在 C++ 推理引擎运行后，仪表盘才会展示实时结果。

### 11.1 让仪表盘显示事件视频列表

`dashboard.py` 默认从 `~/Desktop/deploy/cpp_demo/artifacts` 枚举 `event_*.mp4`。如果你希望列表正常显示，推荐把事件日志路径显式设置到该目录，例如：

```bash
cd ~/Desktop/deploy/cpp_demo
ONNX_EP=cpu FEATURE_SOURCE=yolo-onnx EVENT_LOG=../cpp_demo/artifacts/incidents.json bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```

这样生成的 `event_*.mp4` 会与仪表盘扫描目录保持一致。

## 12. 可选：启动视频流模拟器

如果你没有外部摄像头，但想模拟一个 HTTP 视频流，可执行：

```bash
cd ~/Desktop/deploy
python3 stream_video.py
```

默认服务地址：

```text
http://<board-ip>:5555/video_feed
```

随后把环境文件中的输入源改为：

```bash
SOURCE_TYPE=camera
CAMERA_INDEX=http://<board-ip>:5555/video_feed
```

## 13. 运行结果与验收方式

### 13.1 重点产物

| 文件 | 说明 |
| --- | --- |
| `deploy/artifacts/cpp_demo_onnx_orangepi_summary.json` | 板端 ONNX 路线摘要 |
| `deploy/artifacts/cpp_demo_npu_summary.json` | 板端 NPU 演示摘要 |
| `deploy/artifacts/incidents.json` | 事件日志 |
| `deploy/artifacts/incidents_npu.json` | NPU 演示事件日志 |
| `deploy/artifacts/result.mp4` | 输出视频 |
| `deploy/cpp_demo/artifacts/event_*.mp4` | 事件片段视频 |

### 13.2 仓库内现有样例结果

| 场景 | 产物文件 | 结果 |
| --- | --- | --- |
| 开发板视频文件模式 | `deploy/artifacts/cpp_demo_onnx_orangepi_summary.json` | 1221 帧，60 次事件，约 1.39 FPS |
| 开发板 NPU 流模式 | `deploy/artifacts/cpp_demo_npu_summary.json` | 3235 帧，9 次事件，约 11.38 FPS |

这些结果主要用于说明工程链路已经打通。实际部署阈值、误报率与输入源质量仍需结合现场环境做进一步标定。

## 14. 常见问题

### 14.1 找不到 ONNX Runtime 头文件或动态库

请优先检查：

1. `ONNXRUNTIME_ROOT` 是否写对。
2. `ONNXRUNTIME_INCLUDE_DIR` 与 `ONNXRUNTIME_LIBRARY` 是否和实际解压目录匹配。
3. 模板中的版本号是否已经根据安装结果改成 `1.24.3` 或你的实际版本。

### 14.2 `onnx-ep=cann` 启动失败

最常见原因是当前 ONNX Runtime 是 CPU 版。先执行 `check_cann_env.sh`，确认是否真的存在 `CANN EP` 导出符号。

### 14.3 仪表盘打开但没有图像

请检查：

1. C++ 推理程序是否正在运行。
2. `/dev/shm/preview.jpg` 与 `/dev/shm/status.json` 是否被持续更新。
3. 浏览器访问的是否是开发板实际 IP。

### 14.4 视频流无法连接

若使用 `http://`、`rtsp://` 或 `rtmp://` 地址，请优先在板端通过其他工具验证该流确实可访问，再检查 `CAMERA_INDEX` 配置。

### 14.5 输出视频编码失败

如果输出视频时报 GStreamer 或编码错误，请检查输出文件扩展名是否正确，必要时安装常见编解码插件。

### 14.6 检测结果误报较多

优先从以下参数入手：

1. `THRESHOLD`
2. `SMOOTH_K`
3. 输入视频质量与拍摄视角

## 15. 交付建议

当你准备提交开发板版本说明文档或答辩材料时，建议至少附上以下证据：

1. 开发板实机照片。
2. `cpp_demo_npu_summary.json` 的关键字段截图。
3. 仪表盘页面截图。
4. 一段事件片段视频或实时流演示视频。

如果你需要，我可以在下一步继续为这份部署教程补一版“答辩 / 汇报用简化版”。