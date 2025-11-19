#!/bin/bash

# ============================================================================
# MPCC Build Script with Configurable Options
# ============================================================================
# This script allows easy building with different Neural Network precision settings
#
# Usage:
#   ./build_with_options.sh [preset]
#
# Presets:
#   default    - Default build (Eigen::bfloat16 with FP32 accumulators)
#   fp32       - FlexFloat with FP32 precision (E8M23)
#   bf16       - FlexFloat with BF16 precision (E8M7)
#   fp16       - FlexFloat with FP16 precision (E5M10)
#   e8m6       - FlexFloat with E8M6 precision
#   e8m5       - FlexFloat with E8M5 precision
#   e8m4       - FlexFloat with E8M4 precision
#   custom     - Custom precision (will prompt for exponent/mantissa bits)
#   clean      - Clean build directory
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Parse preset
PRESET=${1:-default}

echo "============================================================================"
echo "MPCC Build Script - Neural Network Precision Configuration"
echo "============================================================================"
echo ""

# Clean build if requested
if [ "$PRESET" = "clean" ]; then
    print_info "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    print_success "Build directory cleaned"
    exit 0
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure based on preset
print_info "Configuring build with preset: ${PRESET}"
echo ""

case "$PRESET" in
    default)
        print_info "Default configuration:"
        print_info "  - Eigen::bfloat16 weights/activations"
        print_info "  - FP32 accumulators"
        print_info "  - No FlexFloat"
        echo ""
        cmake .. \
            -DNN_USE_FLEXFLOAT=OFF
        ;;

    fp32)
        print_info "FlexFloat FP32 (E8M23) configuration:"
        print_info "  - Exponent bits: 8"
        print_info "  - Mantissa bits: 23"
        print_info "  - Equivalent to IEEE 754 single precision"
        echo ""
        cmake .. \
            -DNN_USE_FLEXFLOAT=ON \
            -DNN_FF_EXPONENT_BITS=8 \
            -DNN_FF_MANTISSA_BITS=23
        ;;

    bf16)
        print_info "FlexFloat BF16 (E8M7) configuration:"
        print_info "  - Exponent bits: 8"
        print_info "  - Mantissa bits: 7"
        print_info "  - Google Brain Float 16"
        echo ""
        cmake .. \
            -DNN_USE_FLEXFLOAT=ON \
            -DNN_FF_EXPONENT_BITS=8 \
            -DNN_FF_MANTISSA_BITS=7
        ;;

    fp16)
        print_info "FlexFloat FP16 (E5M10) configuration:"
        print_info "  - Exponent bits: 5"
        print_info "  - Mantissa bits: 10"
        print_info "  - IEEE 754 half precision"
        echo ""
        cmake .. \
            -DNN_USE_FLEXFLOAT=ON \
            -DNN_FF_EXPONENT_BITS=5 \
            -DNN_FF_MANTISSA_BITS=10
        ;;

    e8m6)
        print_info "FlexFloat E8M6 configuration:"
        print_info "  - Exponent bits: 8"
        print_info "  - Mantissa bits: 6"
        echo ""
        cmake .. \
            -DNN_USE_FLEXFLOAT=ON \
            -DNN_FF_EXPONENT_BITS=8 \
            -DNN_FF_MANTISSA_BITS=6
        ;;

    e8m5)
        print_info "FlexFloat E8M5 configuration:"
        print_info "  - Exponent bits: 8"
        print_info "  - Mantissa bits: 5"
        echo ""
        cmake .. \
            -DNN_USE_FLEXFLOAT=ON \
            -DNN_FF_EXPONENT_BITS=8 \
            -DNN_FF_MANTISSA_BITS=5
        ;;

    e8m4)
        print_info "FlexFloat E8M4 configuration:"
        print_info "  - Exponent bits: 8"
        print_info "  - Mantissa bits: 4"
        echo ""
        cmake .. \
            -DNN_USE_FLEXFLOAT=ON \
            -DNN_FF_EXPONENT_BITS=8 \
            -DNN_FF_MANTISSA_BITS=4
        ;;

    custom)
        print_info "Custom FlexFloat configuration"
        echo ""
        read -p "Enter exponent bits (1-8): " EXP_BITS
        read -p "Enter mantissa bits (1-23): " MANT_BITS

        print_info "Custom configuration:"
        print_info "  - Exponent bits: ${EXP_BITS}"
        print_info "  - Mantissa bits: ${MANT_BITS}"
        echo ""

        cmake .. \
            -DNN_USE_FLEXFLOAT=ON \
            -DNN_FF_EXPONENT_BITS=${EXP_BITS} \
            -DNN_FF_MANTISSA_BITS=${MANT_BITS}
        ;;

    *)
        print_error "Unknown preset: ${PRESET}"
        echo ""
        echo "Available presets:"
        echo "  default    - Default build (Eigen::bfloat16 + FP32 accumulators)"
        echo "  fp32       - FlexFloat FP32 (E8M23)"
        echo "  bf16       - FlexFloat BF16 (E8M7)"
        echo "  fp16       - FlexFloat FP16 (E5M10)"
        echo "  e8m6       - FlexFloat E8M6"
        echo "  e8m5       - FlexFloat E8M5"
        echo "  e8m4       - FlexFloat E8M4"
        echo "  custom     - Custom precision"
        echo "  clean      - Clean build directory"
        exit 1
        ;;
esac

# Check if CMake configuration was successful
if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

# Build
echo ""
print_info "Building..."
# Only build test_validation_inference target (not full MPCC_LIB)
make -j$(nproc) test_validation_inference

# Check if build was successful
if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

# Success
echo ""
echo "============================================================================"
print_success "Build completed successfully!"
echo "============================================================================"
echo ""
print_info "Executables:"
print_info "  - ${BUILD_DIR}/MPCC_EXE"
print_info "  - ${BUILD_DIR}/MPCC_TEST"
print_info "  - ${BUILD_DIR}/test_validation_inference"
echo ""
print_info "To run C++/Python validation:"
echo "  cd ${BUILD_DIR}"
echo "  ./test_validation_inference"
echo ""
print_info "To rebuild with different precision:"
echo "  ./build_with_options.sh [preset]"
echo ""
