#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CPP_DEMO_DIR}/.." && pwd)"

ENV_FILE="${SCRIPT_DIR}/orangepi_aipro.env"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      echo "usage: bash scripts/orangepi_build_and_run.sh [--env scripts/orangepi_aipro.env] [-- <extra args>]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[error] env file not found: ${ENV_FILE}" >&2
  echo "copy scripts/orangepi_aipro.env.example to scripts/orangepi_aipro.env and edit it first" >&2
  exit 1
fi

set -a
source <(tr -d '\r' < "${ENV_FILE}")
set +a

ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-}"
ONNXRUNTIME_INCLUDE_DIR="${ONNXRUNTIME_INCLUDE_DIR:-}"
ONNXRUNTIME_LIBRARY="${ONNXRUNTIME_LIBRARY:-}"
BUILD_DIR="${BUILD_DIR:-build-orangepi-onnx}"
ENABLE_ONNX="${ENABLE_ONNX:-ON}"
ENABLE_MINDSPORE="${ENABLE_MINDSPORE:-OFF}"
VIDEO_PATH="${VIDEO_PATH:-../fn.mp4}"
MODEL_PATH="${MODEL_PATH:-../models/version3.onnx}"
POSE_MODEL_PATH="${POSE_MODEL_PATH:-../yolo11n-pose.onnx}"
FEATURE_SOURCE="${FEATURE_SOURCE:-yolo-onnx}"
THRESHOLD="${THRESHOLD:-0.5}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-41}"
FEATURE_DIM="${FEATURE_DIM:-153}"
SMOOTH_K="${SMOOTH_K:-5}"
SOURCE_TYPE="${SOURCE_TYPE:-video}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
ONNX_EP="${ONNX_EP:-cann}"
ONNX_DEVICE_ID="${ONNX_DEVICE_ID:-0}"
SUMMARY_JSON="${SUMMARY_JSON:-../artifacts/cpp_demo_onnx_orangepi_summary.json}"
EVENT_LOG="${EVENT_LOG:-}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-}"
MAX_FRAMES="${MAX_FRAMES:-20}"
LOG_EVERY="${LOG_EVERY:-10}"

if [[ -n "${ONNXRUNTIME_ROOT}" ]]; then
  ONNXRUNTIME_INCLUDE_DIR="${ONNXRUNTIME_INCLUDE_DIR:-${ONNXRUNTIME_ROOT}/include}"
  ONNXRUNTIME_LIBRARY="${ONNXRUNTIME_LIBRARY:-${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so}"
fi

if [[ -z "${ONNXRUNTIME_INCLUDE_DIR}" || -z "${ONNXRUNTIME_LIBRARY}" ]]; then
  echo "[error] ONNXRUNTIME_INCLUDE_DIR or ONNXRUNTIME_LIBRARY is empty" >&2
  exit 1
fi

if [[ "${ONNXRUNTIME_INCLUDE_DIR}" != /* ]]; then
  ONNXRUNTIME_INCLUDE_DIR="${CPP_DEMO_DIR}/${ONNXRUNTIME_INCLUDE_DIR}"
fi
if [[ "${ONNXRUNTIME_LIBRARY}" != /* ]]; then
  ONNXRUNTIME_LIBRARY="${CPP_DEMO_DIR}/${ONNXRUNTIME_LIBRARY}"
fi
if [[ "${VIDEO_PATH}" != /* ]]; then
  VIDEO_PATH="${CPP_DEMO_DIR}/${VIDEO_PATH}"
fi
if [[ "${MODEL_PATH}" != /* ]]; then
  MODEL_PATH="${CPP_DEMO_DIR}/${MODEL_PATH}"
fi
if [[ "${POSE_MODEL_PATH}" != /* ]]; then
  POSE_MODEL_PATH="${CPP_DEMO_DIR}/${POSE_MODEL_PATH}"
fi
if [[ "${SUMMARY_JSON}" != /* ]]; then
  SUMMARY_JSON="${CPP_DEMO_DIR}/${SUMMARY_JSON}"
fi
if [[ -n "${EVENT_LOG}" && "${EVENT_LOG}" != /* ]]; then
  EVENT_LOG="${CPP_DEMO_DIR}/${EVENT_LOG}"
fi
if [[ -n "${OUTPUT_VIDEO}" && "${OUTPUT_VIDEO}" != /* ]]; then
  OUTPUT_VIDEO="${CPP_DEMO_DIR}/${OUTPUT_VIDEO}"
fi

if [[ ! -f "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" ]]; then
  echo "[error] ONNX Runtime header not found: ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" >&2
  exit 1
fi
if [[ ! -f "${ONNXRUNTIME_LIBRARY}" ]]; then
  echo "[error] ONNX Runtime shared library not found: ${ONNXRUNTIME_LIBRARY}" >&2
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
if [[ "${FEATURE_SOURCE}" == "yolo-onnx" && ! -f "${POSE_MODEL_PATH}" ]]; then
  echo "[error] pose model not found: ${POSE_MODEL_PATH}" >&2
  echo "[hint] set POSE_MODEL_PATH in env, or run with '-- --feature-source synthetic' for smoke test only" >&2
  exit 1
fi

echo "[step] configure: ${BUILD_DIR}"
cmake -S "${CPP_DEMO_DIR}" -B "${CPP_DEMO_DIR}/${BUILD_DIR}" \
  -DFIGHT_DEMO_ENABLE_MINDSPORE="${ENABLE_MINDSPORE}" \
  -DFIGHT_DEMO_ENABLE_ONNXRUNTIME="${ENABLE_ONNX}" \
  -DONNXRUNTIME_INCLUDE_DIR="${ONNXRUNTIME_INCLUDE_DIR}" \
  -DONNXRUNTIME_LIBRARY="${ONNXRUNTIME_LIBRARY}"

echo "[step] build"
cmake --build "${CPP_DEMO_DIR}/${BUILD_DIR}" -j

export LD_LIBRARY_PATH="$(dirname "${ONNXRUNTIME_LIBRARY}"):${LD_LIBRARY_PATH:-}"

BIN_PATH="${CPP_DEMO_DIR}/${BUILD_DIR}/fight_detection_demo"
if [[ ! -f "${BIN_PATH}" ]]; then
  echo "[error] binary not found: ${BIN_PATH}" >&2
  exit 1
fi

echo "[step] run ONNX demo"
RUN_ARGS=(
  --source-type "${SOURCE_TYPE}"
  --camera-index "${CAMERA_INDEX}"
  --onnx-ep "${ONNX_EP}"
  --onnx-device-id "${ONNX_DEVICE_ID}"
  --video-path "${VIDEO_PATH}"
  --backend onnx
  --model-path "${MODEL_PATH}"
  --feature-source "${FEATURE_SOURCE}"
  --pose-model-path "${POSE_MODEL_PATH}"
  --threshold "${THRESHOLD}"
  --sequence-length "${SEQUENCE_LENGTH}"
  --feature-dim "${FEATURE_DIM}"
  --smooth-k "${SMOOTH_K}"
  --max-frames "${MAX_FRAMES}"
  --log-every "${LOG_EVERY}"
  --summary-json "${SUMMARY_JSON}"
)

if [[ -n "${EVENT_LOG}" ]]; then
  RUN_ARGS+=(--event-log "${EVENT_LOG}")
fi
if [[ -n "${OUTPUT_VIDEO}" ]]; then
  RUN_ARGS+=(--output-video "${OUTPUT_VIDEO}")
fi

"${BIN_PATH}" "${RUN_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "[ok] done. summary: ${SUMMARY_JSON}"
