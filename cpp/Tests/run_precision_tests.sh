#!/bin/bash

# QDLDL Precision Test Runner
# Builds QDLDL with different precision settings and runs tests

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(dirname "$SCRIPT_DIR")"
RESULT_DIR="$CPP_DIR/../result"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}QDLDL Precision Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if sample directory exists
if [ -z "$1" ]; then
    echo -e "${RED}Error: Sample directory not specified${NC}"
    echo "Usage: $0 <sample_directory>"
    echo "Example: $0 result/qdldl_samples_double/sample_000000"
    exit 1
fi

SAMPLE_DIR="$1"
if [ ! -d "$SAMPLE_DIR" ]; then
    echo -e "${RED}Error: Sample directory does not exist: $SAMPLE_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Input sample: $SAMPLE_DIR${NC}"
echo ""

# Precision configurations to test
# Format: "NAME:USE_FLEXFLOAT:EXPONENT:MANTISSA"
PRECISIONS=(
    "FP32:OFF:0:0"
    "E8M23:ON:8:23"   # Standard float precision with FlexFloat
    "E8M22:ON:8:22"
    "E8M21:ON:8:21"
    "E8M20:ON:8:20"
    "E8M19:ON:8:19"
    "E8M15:ON:8:15"   # Reduced precision
    "E8M10:ON:8:10"   # Low precision
)

# Function to build QDLDL with specific precision
build_qdldl_precision() {
    local name=$1
    local use_flexfloat=$2
    local exp_bits=$3
    local man_bits=$4

    echo -e "${YELLOW}Building QDLDL with precision: $name${NC}"

    # Clean previous build
    cd "$CPP_DIR/External/qdldl"
    rm -rf build
    mkdir -p build
    cd build

    # Configure CMake
    if [ "$use_flexfloat" = "ON" ]; then
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DQDLDL_USE_FLEXFLOAT=ON \
            -DFF_EXPONENT_BITS=$exp_bits \
            -DFF_MANTISSA_BITS=$man_bits \
            -DCMAKE_INSTALL_PREFIX="$CPP_DIR/lib"
    else
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DQDLDL_USE_FLEXFLOAT=OFF \
            -DCMAKE_INSTALL_PREFIX="$CPP_DIR/lib"
    fi

    # Build and install
    make -j$(nproc)
    make install

    echo -e "${GREEN}✓ QDLDL built successfully with $name${NC}"
}

# Function to build test program
build_test_program() {
    local use_flexfloat=$1
    local exp_bits=$2
    local man_bits=$3

    echo -e "${YELLOW}Building test program...${NC}"

    cd "$SCRIPT_DIR"
    rm -rf build
    mkdir -p build
    cd build

    # Configure CMake with same precision settings as QDLDL
    if [ "$use_flexfloat" = "ON" ]; then
        cmake .. \
            -DQDLDL_USE_FLEXFLOAT=ON \
            -DFF_EXPONENT_BITS=$exp_bits \
            -DFF_MANTISSA_BITS=$man_bits
    else
        cmake .. \
            -DQDLDL_USE_FLEXFLOAT=OFF
    fi

    make -j$(nproc)
    make install

    echo -e "${GREEN}✓ Test program built successfully${NC}"
}

# Main test loop
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running precision tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for precision_config in "${PRECISIONS[@]}"; do
    IFS=':' read -r name use_ff exp man <<< "$precision_config"

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Testing precision: $name${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Build QDLDL with this precision
    build_qdldl_precision "$name" "$use_ff" "$exp" "$man"

    # Build test program with same precision settings
    build_test_program "$use_ff" "$exp" "$man"

    # Run test
    OUTPUT_DIR="$RESULT_DIR/qdldl_test_$name"
    echo -e "${YELLOW}Running test...${NC}"
    export LD_LIBRARY_PATH="$CPP_DIR/lib/lib:$LD_LIBRARY_PATH"
    "$CPP_DIR/bin/qdldl_precision_test" "$SAMPLE_DIR" "$OUTPUT_DIR"

    echo -e "${GREEN}✓ Test completed for $name${NC}"
    echo -e "  Results saved to: $OUTPUT_DIR"
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All precision tests completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results directories:"
for precision_config in "${PRECISIONS[@]}"; do
    IFS=':' read -r name _ _ _ <<< "$precision_config"
    echo "  - $RESULT_DIR/qdldl_test_$name"
done
echo ""
