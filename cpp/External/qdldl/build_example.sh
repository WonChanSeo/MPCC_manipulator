#!/bin/bash
set -e

# ========================================
# QDLDL Example Build Script
# ========================================
# Build QDLDL example with FlexFloat precision control
#
# Usage:
#   ./build_example.sh                    # Build with double precision (default)
#   ./build_example.sh E8M23              # Build with float precision (E8M23)
#   ./build_example.sh E8M7               # Build with bfloat16-like precision
#   ./build_example.sh E5M10              # Build with half precision
#   QDLDL_USE_FLEXFLOAT=ON FF_EXPONENT_BITS=8 FF_MANTISSA_BITS=23 ./build_example.sh
#
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default: FlexFloat OFF (use standard double)
QDLDL_USE_FLEXFLOAT=${QDLDL_USE_FLEXFLOAT:-OFF}
FF_EXPONENT_BITS=${FF_EXPONENT_BITS:-11}
FF_MANTISSA_BITS=${FF_MANTISSA_BITS:-52}

# Parse command line argument for precision preset
if [ -n "$1" ]; then
    case "$1" in
        E11M52|double)
            QDLDL_USE_FLEXFLOAT=ON
            FF_EXPONENT_BITS=11
            FF_MANTISSA_BITS=52
            ;;
        E8M23|float)
            QDLDL_USE_FLEXFLOAT=ON
            FF_EXPONENT_BITS=8
            FF_MANTISSA_BITS=23
            ;;
        E8M7|bfloat16)
            QDLDL_USE_FLEXFLOAT=ON
            FF_EXPONENT_BITS=8
            FF_MANTISSA_BITS=7
            ;;
        E5M10|half)
            QDLDL_USE_FLEXFLOAT=ON
            FF_EXPONENT_BITS=5
            FF_MANTISSA_BITS=10
            ;;
        E5M2)
            QDLDL_USE_FLEXFLOAT=ON
            FF_EXPONENT_BITS=5
            FF_MANTISSA_BITS=2
            ;;
        *)
            # Try to parse ExMy format
            if [[ "$1" =~ ^E([0-9]+)M([0-9]+)$ ]]; then
                QDLDL_USE_FLEXFLOAT=ON
                FF_EXPONENT_BITS=${BASH_REMATCH[1]}
                FF_MANTISSA_BITS=${BASH_REMATCH[2]}
            else
                echo "Unknown precision preset: $1"
                echo "Available presets: E11M52 (double), E8M23 (float), E8M7 (bfloat16), E5M10 (half), E5M2"
                exit 1
            fi
            ;;
    esac
fi

# Output directory name based on precision
if [ "$QDLDL_USE_FLEXFLOAT" = "ON" ]; then
    OUTPUT_DIR="qdldl_output_E${FF_EXPONENT_BITS}M${FF_MANTISSA_BITS}"
else
    OUTPUT_DIR="qdldl_output_double"
fi

echo "========================================="
echo "QDLDL Example Build Configuration"
echo "========================================="
echo "  QDLDL_USE_FLEXFLOAT: $QDLDL_USE_FLEXFLOAT"
if [ "$QDLDL_USE_FLEXFLOAT" = "ON" ]; then
    echo "  FF_EXPONENT_BITS:   $FF_EXPONENT_BITS"
    echo "  FF_MANTISSA_BITS:   $FF_MANTISSA_BITS"
fi
echo "  Output directory:   $OUTPUT_DIR"
echo "========================================="

# FlexFloat paths (relative to SCRIPT_DIR which is qdldl directory)
FLEXFLOAT_DIR="$SCRIPT_DIR/../flexfloat"

# Build FlexFloat library if needed
if [ "$QDLDL_USE_FLEXFLOAT" = "ON" ]; then
    echo ""
    echo "Building FlexFloat library..."
    echo ""

    FLEXFLOAT_BUILD_DIR="$FLEXFLOAT_DIR/build"
    mkdir -p "$FLEXFLOAT_BUILD_DIR"
    pushd "$FLEXFLOAT_BUILD_DIR" > /dev/null

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DDISABLE_ROUNDING=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF

    make -j$(nproc)
    popd > /dev/null

    echo "FlexFloat library built successfully."
    echo ""
fi

# Build directory
BUILD_DIR="build_example"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
cmake .. \
    -DQDLDL_USE_FLEXFLOAT=$QDLDL_USE_FLEXFLOAT \
    -DFF_EXPONENT_BITS=$FF_EXPONENT_BITS \
    -DFF_MANTISSA_BITS=$FF_MANTISSA_BITS \
    -DQDLDL_BUILD_DEMO_EXE=ON \
    -DQDLDL_BUILD_STATIC_LIB=ON

# Build QDLDL library
make -j$(nproc)

# Compile example_save.c directly with FlexFloat flags
echo ""
echo "Compiling example_save.c..."

# Use absolute paths (FLEXFLOAT_DIR defined earlier)
CFLAGS="-O3 -I$SCRIPT_DIR/include -I./include"
LDFLAGS="-L./out -lqdldl -lm"

if [ "$QDLDL_USE_FLEXFLOAT" = "ON" ]; then
    CFLAGS="$CFLAGS -DQDLDL_USE_FLEXFLOAT -DFF_exponent_bits=$FF_EXPONENT_BITS -DFF_mantissa_bits=$FF_MANTISSA_BITS"
    CFLAGS="$CFLAGS -I$FLEXFLOAT_DIR/include"
    LDFLAGS="$LDFLAGS -L$FLEXFLOAT_DIR/build -lflexfloat"
fi

gcc $CFLAGS -c $SCRIPT_DIR/examples/example_save.c -o example_save.o
gcc example_save.o $LDFLAGS -o qdldl_example_save

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
echo "Running example_save..."
echo ""

# Run example with output directory
export LD_LIBRARY_PATH="./out:$FLEXFLOAT_DIR/build:$LD_LIBRARY_PATH"
./qdldl_example_save "$SCRIPT_DIR/$OUTPUT_DIR"

echo ""
echo "========================================="
echo "Results saved to: $SCRIPT_DIR/$OUTPUT_DIR"
echo "========================================="
echo ""
echo "Files created:"
ls -la "$SCRIPT_DIR/$OUTPUT_DIR/"
