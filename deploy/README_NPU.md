# Orange Pi Ai Pro 暴力行为检测系统 (竞赛演示版)

本项目是专为 Orange Pi Ai Pro (NPU) 打造的高性能暴力行为检测系统，经过深度优化，适用于实时竞赛演示。

## 核心亮点 (Key Features)

*   **极致性能优化**：
    *   **多线程并行采集**：引入独立的视频采集线程，实现帧率与推理解耦，确保总是处理最新帧，将端到端延迟降至最低。
    *   **高效内存交互**：通过 `/dev/shm` 共享内存传输 640x360 分辨率的预览流，大幅降低 I/O 开销，提升 Web 仪表盘流畅度。
    *   **NPU 硬件加速**：底层采用华为昇腾 CANN 架构加速推理，充分释放板端算力。
*   **竞赛级可视化**：
    *   提供基于 Web 的实时监控仪表盘，直观展示检测画面、FPS、置信度及报警状态。
    *   支持检测结果实时叠加渲染。

## 快速启动 (Quick Start)

### 1. 环境与配置
确保开发板已连接网络，且摄像头或视频流地址可用。
配置文件位于 `cpp_demo/scripts/orangepi_aipro.env`。默认配置已针对竞赛演示优化：
```bash
# 示例配置
SOURCE_TYPE=camera
CAMERA_INDEX="http://10.174.159.83:5555/video_feed"
```

### 2. 启动核心检测引擎 (C++ NPU Backend)
该程序负责从视频源读取画面、在 NPU 上执行推理，并将结果写入共享内存。

```bash
cd ~/Desktop/deploy/cpp_demo
bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```
*注意：首次运行会进行编译，之后将直接启动。若看到 "Connection refused" 错误，请检查视频流服务是否已开启。*

### 3. 启动可视化仪表盘 (Python Web Frontend)
打开一个新的终端窗口，启动 Web 服务器以展示实时检测效果：

```bash
python3 ~/Desktop/deploy/dashboard.py
```

### 4. 浏览演示界面
在浏览器中访问：
**http://10.174.159.83:5000**

## 系统架构
*   **后端 (C++)**: 负责视频解码、图像预处理、NPU 模型推理 (YOLO+LSTM)、共享内存写入。
*   **前端 (Python/Flask)**: 负责从共享内存读取预览帧、渲染 Web 界面、展示报警信息。

## 故障排查
*   **画面延迟大**：系统已启用多线程采集，若仍有延迟，请检查网络带宽或尝试降低输入流分辨率。
*   **无法连接视频流**：请确保 `http://10.174.159.83:5555/video_feed` 可访问。可使用 VLC 播放器在 PC 上测试该地址。
