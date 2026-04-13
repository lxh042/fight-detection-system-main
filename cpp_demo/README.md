# 最小 C++ + OpenCV 对照工程

这个子工程的目标不是复现完整的 Python 模型推理，而是先搭出一个最小的 C++ 视频处理程序，对照当前的无界面 Python 入口：

- 读取视频或摄像头
- 维护 41 帧“预热”窗口
- 打印伪推理结果
- 预留真实 runtime 后端接口
- 可选保存标注视频
- 输出摘要日志
- 可选导出 incidents JSON
- 可选导出 summary JSON

这样做的价值是先学会 C++ 版的程序结构，而不是一开始就碰模型转换和板端运行时。

## 工程结构

- CMakeLists.txt
- src/inference_core.h
- src/inference_core.cpp
- src/main.cpp

## 构建方式

在 cpp_demo 目录下执行：

```powershell
cmake -S . -B build
cmake --build build --config Release
```

如果你在 Ubuntu / WSL 上构建，推荐先安装依赖并显式指定 ONNX Runtime 路径（OpenCV 用系统包）：

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev

cmake -S . -B build-linux \
  -DFIGHT_DEMO_ENABLE_MINDSPORE=OFF \
  -DONNXRUNTIME_INCLUDE_DIR=/usr/local/include/onnxruntime \
  -DONNXRUNTIME_LIBRARY=/usr/local/lib/libonnxruntime.so
cmake --build build-linux -j
```

如果你的 ONNX Runtime 不在 `/usr/local`，把上面两个 `ONNXRUNTIME_*` 改成你实际路径。

也可以直接使用仓库内的本地 ONNX Runtime（已验证可用）：

```bash
cd cpp_demo
mkdir -p third_party/onnxruntime-linux-x64-1.23.2
wget -c -O third_party/onnxruntime-linux-x64-1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
tar -xzf third_party/onnxruntime-linux-x64-1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
  -C third_party/onnxruntime-linux-x64-1.23.2 --strip-components=1

cmake -S . -B build-wsl-onnx-auto \
  -DFIGHT_DEMO_ENABLE_MINDSPORE=OFF \
  -DFIGHT_DEMO_ENABLE_ONNXRUNTIME=ON
cmake --build build-wsl-onnx-auto -j
```

推荐使用一键脚本（同样已验证）：

```bash
cd cpp_demo
bash scripts/wsl_build_onnx.sh
```

可选环境变量：

- `ORT_VERSION`（默认 `1.23.2`）
- `BUILD_DIR`（默认 `build-wsl-onnx-auto`）
- `ENABLE_ONNX`（默认 `ON`）
- `ENABLE_MINDSPORE`（默认 `OFF`）

如果你的机器和当前项目一样，终端里没有全局 cmake 命令，但已经安装了 Visual Studio Build Tools，可以直接使用下面这组命令：

```powershell
& 'D:/App/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' -S . -B build-vs -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="D:/App/Anaconda3/envs/pytorch/Library/cmake"
& 'D:/App/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' --build build-vs --config Release
```

当前仓库已经在这台机器上按上面这组命令验证成功。

如果 CMake 找不到 OpenCV，通常需要显式指定 OpenCV_DIR，例如：

```powershell
cmake -S . -B build -DOpenCV_DIR="D:/path/to/opencv/build"
```

## 运行示例

处理本地视频：

```powershell
.\build\Release\fight_detection_demo.exe --video-path ..\fn.mp4 --max-frames 60 --log-every 20 --output-video ..\artifacts\cpp_demo.mp4
```

如果你使用的是 build-vs 目录，并且 OpenCV DLL 在 conda 环境里，可以先把 DLL 路径加到 PATH，再运行：

```powershell
$env:PATH = 'D:/App/Anaconda3/envs/pytorch/Library/bin;' + $env:PATH
.\build-vs\Release\fight_detection_demo.exe --video-path ..\fn.mp4 --max-frames 60 --log-every 20 --output-video ..\artifacts\cpp_demo.mp4 --event-log ..\artifacts\cpp_demo_incidents.json --summary-json ..\artifacts\cpp_demo_summary.json
```

显式指定后端：

```powershell
.\build-vs\Release\fight_detection_demo.exe --video-path ..\fn.mp4 --backend pseudo --max-frames 60
```

测试 MindIR 适配器骨架：

```powershell
.\build-vs\Release\fight_detection_demo.exe --video-path ..\fn.mp4 --backend mindir --model-path ..\models\version3.mindir --max-frames 60
```

测试 ONNX Runtime 后端：

```powershell
$env:PATH = 'D:/App/Anaconda3/envs/pytorch/Library/bin;D:/App/Anaconda3/envs/pytorch/Lib/site-packages/onnxruntime/capi;' + $env:PATH
.\build-vs\Release\fight_detection_demo.exe --video-path ..\fn.mp4 --backend onnx --model-path ..\models\version3.onnx --max-frames 60 --output-video ..\artifacts\cpp_demo_onnx.mp4 --event-log ..\artifacts\cpp_demo_onnx_incidents.json --summary-json ..\artifacts\cpp_demo_onnx_summary.json
```

Ubuntu / WSL 下运行 ONNX 后端（按需设置 `LD_LIBRARY_PATH`）：

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
./build-linux/fight_detection_demo --video-path ../fn.mp4 --backend onnx --model-path ../models/version3.onnx --max-frames 60 --output-video ../artifacts/cpp_demo_onnx_linux.mp4 --event-log ../artifacts/cpp_demo_onnx_linux_incidents.json --summary-json ../artifacts/cpp_demo_onnx_linux_summary.json
```

如果你使用上面的 `build-wsl-onnx-auto`，命令如下（已验证）：

```bash
export LD_LIBRARY_PATH=$PWD/third_party/onnxruntime-linux-x64-1.23.2/lib:$LD_LIBRARY_PATH
./build-wsl-onnx-auto/fight_detection_demo --video-path ../fn.mp4 --backend onnx --model-path ../models/version3.onnx --max-frames 20 --log-every 10 --summary-json ../artifacts/cpp_demo_onnx_wsl_summary.json
```

也可以使用一键运行脚本：

```bash
cd cpp_demo
bash scripts/wsl_run_onnx_demo.sh
```

可选环境变量：

- `BUILD_DIR`（默认 `build-wsl-onnx-auto`）
- `VIDEO_PATH`（默认 `../fn.mp4`）
- `MODEL_PATH`（默认 `../models/version3.onnx`）
- `SUMMARY_JSON`（默认 `../artifacts/cpp_demo_onnx_wsl_summary.json`）
- `MAX_FRAMES`（默认 `20`）
- `LOG_EVERY`（默认 `10`）

Orange Pi AI Pro（Ubuntu/aarch64）建议先复制模板再替换路径：

```bash
cd cpp_demo/scripts
cp orangepi_aipro.env.example orangepi_aipro.env
```

模板文件包含 ONNX Runtime SDK 路径、构建参数和运行参数：

- `scripts/orangepi_aipro.env.example`

Orange Pi 上推荐直接用一键脚本（会自动 configure + build + run）：

```bash
cd cpp_demo
bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env
```

如果你想给 demo 追加参数（例如输出视频），可以用 `--` 透传：

```bash
bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env -- --output-video ../artifacts/cpp_demo_onnx_orangepi.mp4 --event-log ../artifacts/cpp_demo_onnx_orangepi_incidents.json
```

导出真实 YOLO Pose 特征，再交给 C++ ONNX 后端：

```powershell
& 'D:/App/Anaconda3/envs/pytorch/python.exe' ..\src\export_video_features.py --video-path ..\fn.mp4 --output-csv ..\artifacts\fn_features.csv --max-frames 60
$env:PATH = 'D:/App/Anaconda3/envs/pytorch/Library/bin;D:/App/Anaconda3/envs/pytorch/Lib/site-packages/onnxruntime/capi;' + $env:PATH
.\build-vs\Release\fight_detection_demo.exe --video-path ..\fn.mp4 --backend onnx --model-path ..\models\version3.onnx --feature-source csv --feature-file ..\artifacts\fn_features.csv --max-frames 60 --output-video ..\artifacts\cpp_demo_onnx_real_features.mp4 --event-log ..\artifacts\cpp_demo_onnx_real_features_incidents.json --summary-json ..\artifacts\cpp_demo_onnx_real_features_summary.json
```

纯 C++ 实时前处理：直接在 C++ 中运行 YOLO Pose ONNX，再交给 C++ ONNX 分类器：

```powershell
$env:PATH = 'D:/App/Anaconda3/envs/pytorch/Library/bin;D:/App/Anaconda3/envs/pytorch/Lib/site-packages/onnxruntime/capi;' + $env:PATH
.\build-vs\Release\fight_detection_demo.exe --video-path ..\fn.mp4 --backend onnx --model-path ..\models\version3.onnx --feature-source yolo-onnx --pose-model-path ..\yolo11n-pose.onnx --max-frames 60 --output-video ..\artifacts\cpp_demo_onnx_pure_cpp_pose.mp4 --event-log ..\artifacts\cpp_demo_onnx_pure_cpp_pose_incidents.json --summary-json ..\artifacts\cpp_demo_onnx_pure_cpp_pose_summary.json
```

逐帧对齐 Python 和 C++ 的姿态特征：

```powershell
& 'D:/App/Anaconda3/envs/pytorch/python.exe' ..\src\export_video_features.py --video-path ..\fn.mp4 --output-csv ..\artifacts\fn_features_python_ref.csv --max-frames 45
$env:PATH = 'D:/App/Anaconda3/envs/pytorch/Library/bin;D:/App/Anaconda3/envs/pytorch/Lib/site-packages/onnxruntime/capi;' + $env:PATH
.\build-vs\Release\fight_detection_demo.exe --video-path ..\fn.mp4 --backend onnx --model-path ..\models\version3.onnx --feature-source yolo-onnx --pose-model-path ..\yolo11n-pose.onnx --feature-dump ..\artifacts\fn_features_cpp_yolo_onnx.csv --max-frames 45
& 'D:/App/Anaconda3/envs/pytorch/python.exe' ..\src\compare_feature_csv.py --reference-csv ..\artifacts\fn_features_python_ref.csv --candidate-csv ..\artifacts\fn_features_cpp_yolo_onnx.csv
```

读取摄像头：

```powershell
.\build\Release\fight_detection_demo.exe --source-type camera --camera-index 0 --max-frames 120
```

## 它和 Python 无界面入口的对应关系

- Python 入口: src/run_inference.py
- Python 核心: src/inference_core.py
- C++ 入口: cpp_demo/src/main.cpp
- C++ 核心: cpp_demo/src/inference_core.cpp

当前 C++ 版本先用伪概率代替真实模型输出，目的是先把这些能力练熟：

- 视频读取
- 帧循环
- 预热窗口
- 概率平滑
- 标签判断
- 日志打印
- 结果视频保存
- 独立事件日志输出
- 结构化摘要输出

这条伪推理链路现在已经进一步抽成了分类器接口层：

- `ISequenceClassifier`：序列分类器抽象接口
- `PseudoClassifier`：当前默认的伪实现
- `MindIRClassifier`：MindSpore C++ runtime 适配器骨架
- `OnnxRuntimeClassifier`：已接入真实 ONNX Runtime 推理

这样后面如果你要继续学真实部署，可以把 `PseudoClassifier` 替换成新的实现，例如 `MindSporeClassifier`、`ONNXRuntimeClassifier` 或某个开发板 NPU runtime 适配器，而不用先改视频读取和主循环。

现在的接口已经不再只吃 `frameIndex`，而是接收一个 `SequenceWindow`：

- `lastFrameIndex`：当前滑窗末帧编号
- `sequenceLength`：序列长度，默认 41
- `featureDim`：每帧特征维度，默认 153
- `flattenedFeatures`：留给真实 runtime 的扁平特征输入

这让你后面把 Python 里的 `(41, 153)` 输入张量迁到 C++ 时，接口形状基本已经对齐。

## 当前限制

- `--backend pseudo`：当前可直接运行，仍然使用伪概率逻辑
- `--backend mindir`：已经接好参数和模型路径校验，但会在分类调用时明确报“尚未接入 MindSpore C++ runtime”
- `--backend onnx`：已经能真实加载 `models/version3.onnx` 并运行分类
- `--feature-source synthetic`：仍然使用示例特征，适合做接口调试
- `--feature-source csv`：可以读取 Python 导出的真实 YOLO Pose 特征 CSV，并构造真实 `(41, 153)` 序列窗口
- `--feature-source yolo-onnx`：可以直接在 C++ 中运行 `yolo11n-pose.onnx`，做实时姿态估计、NMS 和 153 维特征构造
- 当前纯 C++ 姿态前处理是最小可运行版本，解析逻辑对齐了当前导出的 `yolo11n-pose.onnx` `(1,56,8400)` 输出，但还没有做更完整的精度对齐与可视化调试
- 当前对齐工具已经可用：C++ 可通过 `--feature-dump` 导出逐帧特征，Python 可用 `src/compare_feature_csv.py` 做逐帧误差统计

也就是说，当前这一步已经完成了“纯 C++ 实时前处理 + 纯 C++ ONNX 分类”的最小闭环。

等这条链路理解清楚以后，再继续替换成真实推理引擎会更稳。

## 本机验证结果

当前机器已经完成下面这条验证链路：

- 使用 Visual Studio Build Tools 自带 CMake 成功配置工程
- 成功编译出 fight_detection_demo.exe
- 成功读取项目根目录下的 fn.mp4
- 成功输出伪推理日志和摘要
- 成功生成结果视频 artifacts/cpp_demo.mp4
- 可以额外生成 incidents JSON，例如 artifacts/cpp_demo_incidents.json
- 可以额外生成 summary JSON，例如 artifacts/cpp_demo_summary.json
- 成功训练并导出 models/version3.onnx
- 成功导出 yolo11n-pose.onnx
- 成功在 C++ 中使用 ONNX Runtime 加载 models/version3.onnx
- 成功生成 ONNX 版本结果文件 artifacts/cpp_demo_onnx.mp4、artifacts/cpp_demo_onnx_incidents.json、artifacts/cpp_demo_onnx_summary.json
- 成功用 Python 导出 fn.mp4 的真实 YOLO Pose 特征 CSV: artifacts/fn_features.csv
- 成功让 C++ ONNX 后端读取真实特征 CSV，并生成 artifacts/cpp_demo_onnx_real_features.mp4、artifacts/cpp_demo_onnx_real_features_incidents.json、artifacts/cpp_demo_onnx_real_features_summary.json
- 成功让 C++ 直接运行 yolo11n-pose.onnx 并生成 artifacts/cpp_demo_onnx_pure_cpp_pose.mp4、artifacts/cpp_demo_onnx_pure_cpp_pose_incidents.json、artifacts/cpp_demo_onnx_pure_cpp_pose_summary.json
- 成功完成 Python 与 C++ 的逐帧特征对齐评估，并确认当前纯 C++ 姿态前处理仍需继续做精度对齐

运行时可能看到 GStreamer 插件警告，但当前程序仍然能正常完成读取和输出。这说明在这台机器上，OpenCV 已经具备可用的视频处理能力。