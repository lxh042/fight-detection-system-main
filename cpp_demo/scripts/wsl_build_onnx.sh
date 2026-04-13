#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ORT_VERSION="${ORT_VERSION:-1.23.2}"
ORT_DIR="${CPP_DEMO_DIR}/third_party/onnxruntime-linux-x64-${ORT_VERSION}"
ORT_TGZ="${ORT_DIR}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"

BUILD_DIR="${BUILD_DIR:-build-wsl-onnx-auto}"
ENABLE_ONNX="${ENABLE_ONNX:-ON}"
ENABLE_MINDSPORE="${ENABLE_MINDSPORE:-OFF}"

mkdir -p "${ORT_DIR}"

if [[ ! -f "${ORT_DIR}/include/onnxruntime_c_api.h" || ! -f "${ORT_DIR}/lib/libonnxruntime.so" ]]; then
  echo "[step] downloading ONNX Runtime ${ORT_VERSION} ..."
  wget -c -O "${ORT_TGZ}" "${ORT_URL}"
  tar -xzf "${ORT_TGZ}" -C "${ORT_DIR}" --strip-components=1
fi

if [[ ! -f "${ORT_DIR}/include/onnxruntime_c_api.h" || ! -f "${ORT_DIR}/lib/libonnxruntime.so" ]]; then
  echo "[error] ONNX Runtime package missing include/lib after extraction: ${ORT_DIR}" >&2
  exit 1
fi

echo "[step] configuring CMake (${BUILD_DIR}) ..."
cmake -S "${CPP_DEMO_DIR}" -B "${CPP_DEMO_DIR}/${BUILD_DIR}" \
  -DFIGHT_DEMO_ENABLE_MINDSPORE="${ENABLE_MINDSPORE}" \
  -DFIGHT_DEMO_ENABLE_ONNXRUNTIME="${ENABLE_ONNX}" \
  -DONNXRUNTIME_INCLUDE_DIR="${ORT_DIR}/include" \
  -DONNXRUNTIME_LIBRARY="${ORT_DIR}/lib/libonnxruntime.so"

echo "[step] building ..."
cmake --build "${CPP_DEMO_DIR}/${BUILD_DIR}" -j

echo "[ok] build done: ${CPP_DEMO_DIR}/${BUILD_DIR}/fight_detection_demo"
