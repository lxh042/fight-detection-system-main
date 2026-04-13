#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${1:-${SCRIPT_DIR}/orangepi_aipro.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[error] env file not found: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source <(tr -d '\r' < "${ENV_FILE}")
set +a

ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-}"
ONNXRUNTIME_INCLUDE_DIR="${ONNXRUNTIME_INCLUDE_DIR:-}"
ONNXRUNTIME_LIBRARY="${ONNXRUNTIME_LIBRARY:-}"

if [[ -n "${ONNXRUNTIME_ROOT}" ]]; then
  ONNXRUNTIME_INCLUDE_DIR="${ONNXRUNTIME_INCLUDE_DIR:-${ONNXRUNTIME_ROOT}/include}"
  ONNXRUNTIME_LIBRARY="${ONNXRUNTIME_LIBRARY:-${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so}"
fi

if [[ "${ONNXRUNTIME_INCLUDE_DIR}" != /* ]]; then
  ONNXRUNTIME_INCLUDE_DIR="${CPP_DEMO_DIR}/${ONNXRUNTIME_INCLUDE_DIR}"
fi
if [[ "${ONNXRUNTIME_LIBRARY}" != /* ]]; then
  ONNXRUNTIME_LIBRARY="${CPP_DEMO_DIR}/${ONNXRUNTIME_LIBRARY}"
fi

echo "[check] ONNX include: ${ONNXRUNTIME_INCLUDE_DIR}"
echo "[check] ONNX library: ${ONNXRUNTIME_LIBRARY}"

if [[ ! -f "${ONNXRUNTIME_LIBRARY}" ]]; then
  echo "[error] libonnxruntime.so not found" >&2
  exit 1
fi

if command -v npu-smi >/dev/null 2>&1; then
  echo "[check] npu-smi found"
  npu-smi info || true
else
  echo "[warn] npu-smi not found in PATH"
fi

echo "[check] Ascend env vars"
echo "  ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<empty>}"
echo "  ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<empty>}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"

echo "[check] ORT CANN EP exported symbol (strong check)"
HAS_CANN_EP_SYMBOL=0
if command -v nm >/dev/null 2>&1; then
  if nm -D "${ONNXRUNTIME_LIBRARY}" 2>/dev/null | grep -E "SessionOptionsAppendExecutionProvider_CANN|OrtSessionOptionsAppendExecutionProvider_CANN" >/dev/null 2>&1; then
    HAS_CANN_EP_SYMBOL=1
  fi
elif command -v readelf >/dev/null 2>&1; then
  if readelf -Ws "${ONNXRUNTIME_LIBRARY}" 2>/dev/null | grep -E "SessionOptionsAppendExecutionProvider_CANN|OrtSessionOptionsAppendExecutionProvider_CANN" >/dev/null 2>&1; then
    HAS_CANN_EP_SYMBOL=1
  fi
fi

if [[ "${HAS_CANN_EP_SYMBOL}" -eq 1 ]]; then
  echo "[ok] found CANN EP exported symbol in libonnxruntime.so"
else
  echo "[warn] CANN EP exported symbol not found (this is typically a CPU-only ONNX Runtime build)"
  if command -v strings >/dev/null 2>&1; then
    if strings "${ONNXRUNTIME_LIBRARY}" | grep -Ei "CANN|Ascend|acl" >/dev/null 2>&1; then
      echo "[note] plain strings contains CANN/Ascend words, but that alone is not sufficient for CANN EP availability"
    fi
  fi
fi

echo "[check] ORT dynamic deps (heuristic)"
if command -v ldd >/dev/null 2>&1; then
  ldd "${ONNXRUNTIME_LIBRARY}" | grep -Ei "ascend|acl|cann|ge_runner|graph" || echo "[warn] no Ascend/CANN deps found by ldd"
else
  echo "[warn] ldd not found; skip dependency check"
fi

echo "[done] If CANN EP exported symbol is missing, replace ONNXRUNTIME_ROOT with a CANN-EP build before setting ONNX_EP=cann."
