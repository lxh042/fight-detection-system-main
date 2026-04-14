# Fight Detection System

基于人体关键点序列的实时暴力行为检测系统，采用 YOLO11 Pose 提取人体姿态特征，并结合轻量时序分类模型完成暴力 / 非暴力识别。该仓库同时保留了本地研发验证链路与 Orange Pi AI Pro 板端部署成果，适合用于课程项目、工程演示、端侧推理验证和部署实践。

## 项目介绍

本项目的核心思路是先做人，再做行为判断。

具体流程如下：

1. 使用 YOLO11 Pose 检测人体 17 个关键点。
2. 每帧最多保留 3 个人体目标，不足部分补零。
3. 将单帧关键点展开为 153 维特征。
4. 使用连续 41 帧构成时序窗口，输入轻量分类器。
5. 输出暴力概率，并结合阈值、平滑和冷却时间生成事件告警。

这个仓库不是单一的“训练代码”或“部署代码”，而是一条完整工程链路：

- 本地部分：用于数据准备、特征提取、模型训练、离线评估、无界面推理和 Streamlit 原型展示。
- 板端部分：用于 Orange Pi AI Pro 上的 C++ 推理、ONNX Runtime 接入、NPU 路线验证和 Web Dashboard 展示。

## 系统架构

```mermaid
flowchart LR
    A[视频 / 摄像头 / 网络流] --> B[YOLO11 Pose]
    B --> C[41 x 153 时序特征]
    C --> D[GRU / ONNX 分类模型]
    D --> E[暴力概率]
    E --> F[阈值 + 平滑 + 事件冷却]
    F --> G[可视化展示 / 事件日志 / 板端告警]
```

## 仓库分区

| 分区 | 说明 |
| --- | --- |
| `deploy/` | 最终部署成果，面向 Orange Pi AI Pro |
| 其余目录 | 本地研发、训练、验证、模型导出与 C++ 对照工程 |

这个划分很重要：`deploy/` 是最终交付形态，其余目录是项目从原型到部署的研发过程。

## 主要特性

- 基于姿态关键点而非原始 RGB 直接分类，输入更轻量，适合时序建模。
- 支持本地 Python 原型，包括训练、离线评估、无界面推理和 Streamlit 演示。
- 支持 ONNX 导出与本地 C++ 对照验证。
- 支持 Orange Pi AI Pro 上的 C++ 推理引擎和 Web Dashboard。
- 板端链路支持 `ONNX Runtime + YOLO Pose ONNX + C++`，可进一步切换到 `CANN EP` 做 NPU 路线验证。

## 核心特征规格

| 项目 | 数值 |
| --- | --- |
| 单人关键点数 | 17 |
| 每帧最多人数量 | 3 |
| 单帧特征维度 | 153 |
| 序列长度 | 41 |
| 模型输入形状 | `(41, 153)` |

## 目录结构

```text
.
├── app.py                    # Streamlit 原型界面入口
├── src/                      # 无界面推理、评估、导出与对齐工具
├── models/                   # 训练结果、导出模型与训练脚本
├── cpp_demo/                 # 本地 C++ 对照工程
├── deploy/                   # Orange Pi AI Pro 最终部署成果
├── docs/                     # 项目说明文档
├── artifacts/                # 本地验证输出样例
├── data/                     # 数据集目录
├── preprocessed/             # 特征缓存目录
├── utills/                   # 关键点提取与缓存工具
├── yolo11n-pose.pt           # 本地姿态模型权重
└── yolo11n-pose.onnx         # 导出的姿态 ONNX 模型
```

## 快速开始

### 1. 创建 conda 环境

```bash
conda create -n fight-detection python=3.10 -y
conda activate fight-detection
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 准备数据集

推荐数据集：

- [Real Life Violence and Non-Violence Data](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

目录结构：

```text
data/
  train/
    violence/
    non-violence/
  test/
    violence/
    non-violence/
```

### 3. 训练模型

```bash
python models/training/train_model.py
```

训练完成后会生成：

- `models/version3.ckpt`
- `models/version3.mindir`

### 4. 离线评估

```bash
python src/predictions.py
```

### 5. 运行无界面推理

```bash
python src/run_inference.py --source-type video --video-path fn.mp4 --max-frames 60 --log-every 20 --output-video artifacts/annotated_demo.mp4 --event-log artifacts/incidents.json --summary-json artifacts/summary.json
```

### 6. 启动本地 Web 原型

```bash
streamlit run app.py
```

## Orange Pi AI Pro 部署

板端最终成果位于 `deploy/`，推荐按以下顺序阅读和执行：

1. 阅读 [deploy/SETUP.md](deploy/SETUP.md)
2. 阅读 [docs/ORANGEPI_AI_PRO_DEPLOYMENT_GUIDE.md](docs/ORANGEPI_AI_PRO_DEPLOYMENT_GUIDE.md)
3. 在开发板上运行 `deploy/install_deps_orangepi.sh`
4. 使用 `deploy/cpp_demo/scripts/orangepi_build_and_run.sh` 完成构建与运行
5. 使用 `deploy/dashboard.py` 启动 Web 面板

## 文档导航

- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)：项目总说明
- [docs/LOCAL_DEVELOPMENT_GUIDE.md](docs/LOCAL_DEVELOPMENT_GUIDE.md)：本地电脑研发与运行教程
- [docs/ORANGEPI_AI_PRO_DEPLOYMENT_GUIDE.md](docs/ORANGEPI_AI_PRO_DEPLOYMENT_GUIDE.md)：Orange Pi AI Pro 部署教程
- [docs/notes/deployment_breadth_first_notes.md](docs/notes/deployment_breadth_first_notes.md)：部署方向学习笔记
- [docs/notes/inference_refactor_notes.md](docs/notes/inference_refactor_notes.md)：推理链路重构笔记

## 已验证样例结果

以下结果来自仓库中的现有样例产物，主要用于说明工程链路已经打通：

| 场景 | 结果 |
| --- | --- |
| 本地 Python 无界面推理 | 60 帧，1 次事件，约 13.69 FPS |
| 本地 C++ 伪后端结构验证 | 60 帧，1 次事件，约 93.42 FPS |
| 本地 C++ ONNX + Python 导出特征 | 45 帧，0 次事件，约 68.98 FPS |
| 本地纯 C++ Pose + ONNX 分类 | 45 帧，0 次事件，约 9.71 FPS |
| 开发板 NPU 演示流 | 3235 帧，9 次事件，约 11.38 FPS |

## 当前限制

1. `MindIR` 评估链路当前采用单样本推理，以兼容当前导出模型的图行为。
2. 板端最终路线以 `ONNX Runtime + YOLO Pose ONNX + C++` 为主，不是 MindSpore Lite 完整部署链路。
3. 纯 C++ 姿态前处理虽然已跑通，但与 Python / Ultralytics 的结果仍在持续对齐中。
4. 仓库中的 `utills/` 目录名沿用了原始拼写，后续可继续重构为 `utils/`。

## 适用场景

- 课程设计与毕业设计
- 轻量行为识别原型验证
- 边缘设备部署与国产化适配练习
- Python 与 C++ 协同推理工程学习

## 使用说明

本项目目前更适合研究、教学和工程演示用途。若用于实际安防场景，需要进一步补充数据闭环、误报控制、隐私合规和现场标定工作。

## 更详细教程可以看以下网站 (https://blog.csdn.net/lxh042/category_13153830.html)
