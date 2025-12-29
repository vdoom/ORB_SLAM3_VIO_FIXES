#!/bin/bash

# Build script for Pear VIO Camera Library
# Builds both shared and static libraries into lib/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "Building Pear VIO Camera Library..."
echo "  Source: ${SCRIPT_DIR}"
echo "  Build:  ${BUILD_DIR}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo ""
echo "Build complete. Libraries in: ${SCRIPT_DIR}/lib/"
ls -la "${SCRIPT_DIR}/lib/"
