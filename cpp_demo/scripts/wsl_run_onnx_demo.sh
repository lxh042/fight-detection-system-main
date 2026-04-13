#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CPP_DEMO_DIR}/.." && pwd)"

ORT_VERSION="${ORT_VERSION:-1.23.2}"
ORT_DIR="${CPP_DEMO_DIR}/third_party/onnxruntime-linux-x64-${ORT_VERSION}"
BUILD_DIR="${BUILD_DIR:-build-wsl-onnx-auto}"
BIN_PATH="${CPP_DEMO_DIR}/${BUILD_DIR}/fight_detection_demo"

VIDEO_PATH="${VIDEO_PATH:-${PROJECT_ROOT}/fn.mp4}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/version3.onnx}"
SUMMARY_JSON="${SUMMARY_JSON:-${PROJECT_ROOT}/artifacts/cpp_demo_onnx_wsl_summary.json}"
MAX_FRAMES="${MAX_FRAMES:-20}"
LOG_EVERY="${LOG_EVERY:-10}"

if [[ ! -f "${BIN_PATH}" ]]; then
  echo "[error] binary not found: ${BIN_PATH}" >&2
  echo "run scripts/wsl_build_onnx.sh first" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[error] model not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "[error] video not found: ${VIDEO_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ORT_DIR}/lib/libonnxruntime.so" ]]; then
  echo "[error] ONNX Runtime .so not found: ${ORT_DIR}/lib/libonnxruntime.so" >&2
  echo "run scripts/wsl_build_onnx.sh first" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${ORT_DIR}/lib:${LD_LIBRARY_PATH:-}"

"${BIN_PATH}" \
  --video-path "${VIDEO_PATH}" \
  --backend onnx \
  --model-path "${MODEL_PATH}" \
  --max-frames "${MAX_FRAMES}" \
  --log-every "${LOG_EVERY}" \
  --summary-json "${SUMMARY_JSON}" \
  "$@"
