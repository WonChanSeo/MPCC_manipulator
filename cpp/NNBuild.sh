#!/bin/bash
set -e

# ========================================
# NN FlexFloat Configuration
# ========================================
# Set NN_USE_FLEXFLOAT=ON to enable FlexFloat precision testing for NN inference
# Set NN_USE_FLEXFLOAT=OFF to use standard bfloat16 precision
NN_USE_FLEXFLOAT=${NN_USE_FLEXFLOAT:-OFF}

# FlexFloat precision configuration (only used when NN_USE_FLEXFLOAT=ON)
NN_FF_EXPONENT_BITS=${NN_FF_EXPONENT_BITS:-8}
NN_FF_MANTISSA_BITS=${NN_FF_MANTISSA_BITS:-23}

echo "========================================="
echo "NN Build Configuration:"
echo "  NN_USE_FLEXFLOAT:        $NN_USE_FLEXFLOAT"
if [ "$NN_USE_FLEXFLOAT" = "ON" ]; then
    echo "  NN_FF_EXPONENT_BITS:     $NN_FF_EXPONENT_BITS"
    echo "  NN_FF_MANTISSA_BITS:     $NN_FF_MANTISSA_BITS"
fi
echo "========================================="

# Build MPCC with NN FlexFloat configuration
rm -rf build
mkdir -p build
cd build

cmake .. \
    -DNN_USE_FLEXFLOAT=$NN_USE_FLEXFLOAT \
    -DNN_FF_EXPONENT_BITS=$NN_FF_EXPONENT_BITS \
    -DNN_FF_MANTISSA_BITS=$NN_FF_MANTISSA_BITS

make -j8

echo "========================================="
echo "NN Build Complete!"
echo "========================================="
