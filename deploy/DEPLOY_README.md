# Orange Pi AI Pro 部署包

这个目录是从项目中提取的板端部署最小集合。

## 目录结构

- `cpp_demo/`：C++ 工程（`CMakeLists.txt`、`src/`、部署脚本）
- `models/version3.onnx`：分类模型
- `yolo11n-pose.onnx`：姿态模型（可选，用于 `--feature-source yolo-onnx`）
- `fn.mp4`：本地测试视频（可选）

## 板端部署

详细步骤请参考 [SETUP.md](SETUP.md)。

1. **环境安装**：执行 `./install_deps_orangepi.sh` 安装依赖 (OpenCV, CMake, ONNX Runtime ARM64)。
2. **配置文件**：编辑 `cpp_demo/scripts/orangepi_aipro.env` 并指向你的 ONNX Runtime 路径。
3. **编译运行**：`cd cpp_demo && bash scripts/orangepi_build_and_run.sh --env scripts/orangepi_aipro.env`。

> 注意：此部署包已针对 Ubuntu 22.04 LTS (aarch64) 测试优化。
