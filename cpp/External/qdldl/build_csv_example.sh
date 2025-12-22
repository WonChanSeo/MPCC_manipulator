#!/bin/bash
set -e

# ========================================
# QDLDL CSV Example Build Script
# ========================================
# Build and run QDLDL with CSV matrix input
#
# Usage:
#   ./build_csv_example.sh <input.csv> [output_dir]
#
# Examples:
#   ./build_csv_example.sh ../../reference_matrix/permuted_A_upper.csv
#   ./build_csv_example.sh ../../reference_matrix/permuted_A_upper.csv my_output
#
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <input.csv> [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 ../../reference_matrix/permuted_A_upper.csv"
    exit 1
fi

INPUT_CSV="$1"
OUTPUT_DIR="${2:-qdldl_csv_output}"

# Convert to absolute path if relative
if [[ ! "$INPUT_CSV" = /* ]]; then
    INPUT_CSV="$SCRIPT_DIR/$INPUT_CSV"
fi

if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"
fi

echo "========================================="
echo "QDLDL CSV Example Build"
echo "========================================="
echo "Input CSV:  $INPUT_CSV"
echo "Output dir: $OUTPUT_DIR"
echo "========================================="

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input file not found: $INPUT_CSV"
    exit 1
fi

# Build directory
BUILD_DIR="build_csv"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake (standard double precision)
echo ""
echo "Configuring CMake..."
cmake .. \
    -DQDLDL_USE_FLEXFLOAT=OFF \
    -DQDLDL_BUILD_STATIC_LIB=ON

# Build QDLDL library
echo ""
echo "Building QDLDL library..."
make -j$(nproc)

# Compile example_csv.c
echo ""
echo "Compiling example_csv.c..."
CFLAGS="-O3 -I$SCRIPT_DIR/include -I./include"
LDFLAGS="-L./out -lqdldl -lm"

gcc $CFLAGS -c "$SCRIPT_DIR/examples/example_csv.c" -o example_csv.o
gcc example_csv.o $LDFLAGS -o qdldl_csv

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
echo "Running qdldl_csv..."
echo ""

# Run
export LD_LIBRARY_PATH="./out:$LD_LIBRARY_PATH"
./qdldl_csv "$INPUT_CSV" "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "Done! Results in: $OUTPUT_DIR"
echo "========================================="
