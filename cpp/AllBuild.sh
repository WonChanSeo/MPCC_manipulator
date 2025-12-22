#!/bin/bash
set -e

# ========================================
# FlexFloat Configuration for OSQP/QDLDL
# ========================================
# Set OSQP_USE_FLEXFLOAT=ON to enable FlexFloat precision testing
# Set OSQP_USE_FLEXFLOAT=OFF to use standard float precision
OSQP_USE_FLEXFLOAT=${OSQP_USE_FLEXFLOAT:-OFF}

# Set QDLDL_USE_FLEXFLOAT=OFF to disable FlexFloat only in QDLDL
# (defaults to same as OSQP_USE_FLEXFLOAT if not specified)
QDLDL_USE_FLEXFLOAT=${QDLDL_USE_FLEXFLOAT:-$OSQP_USE_FLEXFLOAT}

# OSQP FlexFloat precision configuration: E11M52 (double precision equivalent)
FF_EXPONENT_BITS=${FF_EXPONENT_BITS:-11}
FF_MANTISSA_BITS=${FF_MANTISSA_BITS:-52}

# ========================================
# QDLDL Sample Logging Configuration
# ========================================
# Set QDLDL_ENABLE_SAMPLE_LOGGING=ON to save QDLDL samples for precision analysis
# Samples will be saved to QDLDL_SAMPLE_OUTPUT_DIR
QDLDL_ENABLE_SAMPLE_LOGGING=${QDLDL_ENABLE_SAMPLE_LOGGING:-OFF}
QDLDL_SAMPLE_OUTPUT_DIR=${QDLDL_SAMPLE_OUTPUT_DIR:-"../../result/qdldl_samples"}

# ========================================
# FlexFloat Configuration for NN Inference
# ========================================
# Set NN_USE_FLEXFLOAT=ON to enable FlexFloat for NN inference
NN_USE_FLEXFLOAT=${NN_USE_FLEXFLOAT:-OFF}

# NN FlexFloat precision configuration: E8M7
NN_FF_EXPONENT_BITS=${NN_FF_EXPONENT_BITS:-8}
NN_FF_MANTISSA_BITS=${NN_FF_MANTISSA_BITS:-7}

echo "========================================="
echo "Build Configuration:"
echo "  OSQP/QDLDL FlexFloat:"
echo "    OSQP_USE_FLEXFLOAT:   $OSQP_USE_FLEXFLOAT"
echo "    QDLDL_USE_FLEXFLOAT:  $QDLDL_USE_FLEXFLOAT"
if [ "$OSQP_USE_FLEXFLOAT" = "ON" ]; then
    echo "    FF_EXPONENT_BITS:     $FF_EXPONENT_BITS"
    echo "    FF_MANTISSA_BITS:     $FF_MANTISSA_BITS"
fi
echo ""
echo "  QDLDL Sample Logging:"
echo "    QDLDL_ENABLE_SAMPLE_LOGGING: $QDLDL_ENABLE_SAMPLE_LOGGING"
if [ "$QDLDL_ENABLE_SAMPLE_LOGGING" = "ON" ]; then
    echo "    QDLDL_SAMPLE_OUTPUT_DIR:    $QDLDL_SAMPLE_OUTPUT_DIR"
fi
echo ""
echo "  NN Inference FlexFloat:"
echo "    NN_USE_FLEXFLOAT:     $NN_USE_FLEXFLOAT"
if [ "$NN_USE_FLEXFLOAT" = "ON" ]; then
    echo "    NN_FF_EXPONENT_BITS:  $NN_FF_EXPONENT_BITS"
    echo "    NN_FF_MANTISSA_BITS:  $NN_FF_MANTISSA_BITS"
fi
echo "========================================="

cd External
cd osqp
rm -rf build
rm -rf lib
mkdir -p build lib
cd build
cmake .. \
    -DOSQP_USE_FLOAT=OFF \
    -DOSQP_USE_FLEXFLOAT=$OSQP_USE_FLEXFLOAT \
    -DQDLDL_USE_FLEXFLOAT=$QDLDL_USE_FLEXFLOAT \
    -DFF_EXPONENT_BITS=$FF_EXPONENT_BITS \
    -DFF_MANTISSA_BITS=$FF_MANTISSA_BITS \
    -DQDLDL_ENABLE_SAMPLE_LOGGING=$QDLDL_ENABLE_SAMPLE_LOGGING \
    -DCMAKE_INSTALL_PREFIX=$(realpath ../lib)
make
make install
EXPORT_LINE="export LD_LIBRARY_PATH=\"$(realpath ../lib/lib)\":\$LD_LIBRARY_PATH"
if ! grep -Fxq "$EXPORT_LINE" ~/.bashrc
then
    echo "$EXPORT_LINE" >> ~/.bashrc
    echo "[INFO] Added LD_LIBRARY_PATH to ~/.bashrc"
else
    echo "[INFO] LD_LIBRARY_PATH already exists in ~/.bashrc. Skipping."
fi
. ~/.bashrc

cd ~/git/MPCC_manipulator/cpp/External/osqp_eigen
rm -rf build
rm -rf lib
mkdir -p build lib
cd build
cmake .. -DOSQP_IS_V1=ON -DCMAKE_INSTALL_PREFIX:PATH=$(realpath ../lib) -DCMAKE_PREFIX_PATH=$(realpath ../../osqp/lib) -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make
make install
EXPORT_LINE="export OsqpEigen_DIR=\"$(realpath ..)\""
if ! grep -Fxq "$EXPORT_LINE" ~/.bashrc
then
    echo "$EXPORT_LINE" >> ~/.bashrc
    echo "[INFO] Added OsqpEigen_DIR to ~/.bashrc"
else
    echo "[INFO] OsqpEigen_DIR already exists in ~/.bashrc. Skipping."
fi
. ~/.bashrc


cd ~/git/MPCC_manipulator/cpp
export LD_LIBRARY_PATH="/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp/lib/lib":$LD_LIBRARY_PATH
    export OsqpEigen_DIR="/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp_eigen"
    echo "MPCC_manipulator Env set activated."

# 기존 build 디렉토리를 삭제하고 새로 생성
rm -rf build
mkdir build
cd build

# cmake로 빌드 설정 및 make 실행 (NN FlexFloat 및 QDLDL 샘플 로깅 옵션 포함)
cmake .. \
    -DNN_USE_FLEXFLOAT=$NN_USE_FLEXFLOAT \
    -DNN_FF_EXPONENT_BITS=$NN_FF_EXPONENT_BITS \
    -DNN_FF_MANTISSA_BITS=$NN_FF_MANTISSA_BITS \
    -DQDLDL_ENABLE_SAMPLE_LOGGING=$QDLDL_ENABLE_SAMPLE_LOGGING
make -j8

# 빌드 디렉토리에서 상위 디렉토리로 이동
cd ..

