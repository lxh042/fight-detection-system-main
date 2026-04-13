# Orange Pi AI Pro (Ubuntu) 部署与运行指南

本文档指导如何在 Orange Pi AI Pro 开发板上配置环境并运行本项目。

## 1. 准备工作

### 硬件要求
- Orange Pi AI Pro 开发板
- 电源适配器 (推荐使用原装电源，保证供电稳定)
- 网线或 WiFi 连接 (需访问外网下载依赖)
- 显示器/键盘/鼠标 (或通过 SSH 远程连接)

### 系统要求
- 推荐使用官方 Ubuntu 22.04 LTS 镜像。
- 确保系统时间正确：`sudo date -s "2024-03-17 10:00:00"` (根据实际情况调整)。

## 2. 安装依赖

为了简化安装过程，我们提供了一键安装脚本 `install_deps_orangepi.sh`。

### 自动安装 (推荐)
在开发板终端进入本目录，执行：

```bash
chmod +x install_deps_orangepi.sh
./install_deps_orangepi.sh
```

该脚本会自动完成以下操作：
1. 更新 apt 软件源。
2. 安装编译工具 (`build-essential`, `cmake`, `pkg-config`)。
3. 安装 OpenCV 开发库 (`libopencv-dev`)。
4. 在当前目录下创建 `third_party` 文件夹。
5. 下载并解压适配 ARM64 架构的 ONNX Runtime (v1.24.3)。

### 手动安装 (如果自动脚本失败)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libopencv-dev wget tar

mkdir -p third_party && cd third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-aarch64-1.24.3.tgz
tar -xzvf onnxruntime-linux-aarch64-1.24.3.tgz
```

## 3. 配置环境

进入代码目录：
```bash
cd cpp_demo
```

### 复制并编辑配置文件
复制模板文件：
```bash
cp scripts/orangepi_aipro.env.example scripts/orangepi_aipro.env
```

**关键步骤**：编辑 `scripts/orangepi_aipro.env`，确保 `ONNXRUNTIME_ROOT` 指向刚才下载解压的路径。

如果使用自动脚本安装，默认路径应修改为：
```bash
# 修改为实际解压路径，注意 ../../third_party 表示上上级目录下的 third_party
ONNXRUNTIME_ROOT=../../third_party/onnxruntime-linux-aarch64-1.24.3
```
或者使用绝对路径：
```bash
ONNXRUNTIME_ROOT=/home/orangepi/Desktop/deploy/third_party/onnxruntime-linux-aarch64-1.24.3
```

## 4. 编译与运行

在 `cpp_demo` 目录下执行：

```bash
bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```

说明：默认会使用 `FEATURE_SOURCE=yolo-onnx` 和 `POSE_MODEL_PATH=../yolo11n-pose.onnx`，即基于姿态关键点进行真实检测。

完整视频检测（不截断帧数）：
```bash
MAX_FRAMES=0 bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```

方案一（ONNX Runtime + CANN EP，NPU）运行参数：
```bash
ONNX_EP=cann ONNX_DEVICE_ID=0 MAX_FRAMES=0 bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```
默认环境模板已将 `ONNX_EP` 设为 `cann`。如果当前 ONNX Runtime 仍是 CPU 版本，程序会明确报错提示你切换到 CANN EP 版本的 ONNX Runtime。

方案一落地步骤（推荐按顺序执行）：
```bash
# 1) 先自检当前环境（CPU/CANN 版 ORT、Ascend 依赖）
bash scripts/check_cann_env.sh scripts/orangepi_aipro.env

# 2) 将 ONNXRUNTIME_ROOT 指向 CANN-EP 版 ONNX Runtime（需你自行准备该版本）
#    例如：ONNXRUNTIME_ROOT=/home/HwHiAiUser/onnxruntime-cann-aarch64

# 3) 切到 CANN EP 再跑
ONNX_EP=cann ONNX_DEVICE_ID=0 MAX_FRAMES=0 bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```

保存可视化结果视频：
```bash
MAX_FRAMES=0 OUTPUT_VIDEO=../artifacts/result.mp4 bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```

### 预期输出
- 脚本会自动调用 CMake 进行编译。
- 编译成功后，会自动运行 `fight_detection_demo`。
- 程序将加载 `models/version3.onnx` 和 `fn.mp4`，输出推理日志。
- 结束后会生成性能统计文件 `artifacts/cpp_demo_onnx_orangepi_summary.json`。

## 5. 常见问题

**Q: 找不到 libonnxruntime.so?**
A: 请检查 `scripts/orangepi_aipro.env` 中的 `ONNXRUNTIME_LIBRARY` 路径是否正确。它通常位于 `${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so`。

**Q: 运行提示 Permission denied?**
A: 检查脚本是否有执行权限：`chmod +x scripts/*.sh`。

**Q: 缺少 OpenCV?**
A: 确保执行了 `sudo apt-get install libopencv-dev`。

**Q: 推理速度慢?**
A: 本 Demo 已支持 `ONNX_EP=cann`（NPU）与 `ONNX_EP=cpu` 两种模式。若速度仍慢，先运行 `bash scripts/check_cann_env.sh scripts/orangepi_aipro.env` 确认 `libonnxruntime.so` 导出了 CANN EP 符号，并检查 Ascend/CANN 环境变量是否正确。

**Q: 视频里有打斗，但 `incident_count` 仍然是 0?**
A: 先确认你没有使用 `synthetic` 特征。执行后日志中应出现 `--feature-source yolo-onnx`。同时可尝试降低阈值：
`THRESHOLD=0.35 MAX_FRAMES=0 bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env`。

**Q: 输出视频时报 GStreamer 编码错误？**
A: 先确认输出文件扩展名正确（例如 `result.mp4`，不要写成 `result.mp`）。若仍报错，请安装编码插件：
`sudo apt-get install -y gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav`。
