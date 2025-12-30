#!/bin/bash
# Build script for RPi Camera Viewer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== RPi Camera Viewer Build ==="

# Check dependencies
echo "Checking dependencies..."

check_pkg() {
    if ! pkg-config --exists "$1" 2>/dev/null; then
        echo "ERROR: $1 not found. Install with: sudo apt install $2"
        return 1
    fi
    return 0
}

MISSING=0
check_pkg "Qt6Widgets" "qt6-base-dev" || MISSING=1
check_pkg "libcamera" "libcamera-dev" || MISSING=1

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Install missing dependencies:"
    echo "  sudo apt install qt6-base-dev libqt6openglwidgets6 libcamera-dev cmake build-essential"
    exit 1
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo ""
echo "Configuring..."
cmake ..

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "Run with: $BUILD_DIR/rpi_camera_viewer"
