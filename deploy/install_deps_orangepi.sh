#!/bin/bash
set -e

# 1. Update package list
echo "Updating package list..."
sudo apt-get update

# 2. Install build tools and OpenCV
echo "Installing build essentials and OpenCV..."
sudo apt-get install -y build-essential cmake pkg-config libopencv-dev wget tar

# 3. Create third_party directory
echo "Setting up third_party directory..."
mkdir -p third_party
cd third_party

# 4. Download ONNX Runtime for aarch64
ONNX_VERSION="1.24.3"
ONNX_FILE="onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_FILE}"

if [ ! -d "onnxruntime-linux-aarch64-${ONNX_VERSION}" ]; then
    if [ ! -f "${ONNX_FILE}" ]; then
        echo "Downloading ONNX Runtime v${ONNX_VERSION}..."
        wget -c "${ONNX_URL}"
    fi
    echo "Extracting ONNX Runtime..."
    tar -xzvf "${ONNX_FILE}"
else
    echo "ONNX Runtime already exists in third_party/"
fi

# 5. Clean up
# rm "${ONNX_FILE}"  # Optional: keep it or remove it

echo "Dependencies installed successfully!"
echo "Next step: Check SETUP_AND_RUN.md to configure and run the demo."
