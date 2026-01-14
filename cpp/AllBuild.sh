#!/bin/bash
set -e

# ========================================
# OSQP/QDLDL Float Type Configuration
# ========================================
# OSQP uses single precision float (32-bit) - hardcoded ON
OSQP_USE_FLOAT=ON

# ========================================
# Truncate Mode Configuration
# ========================================
# Truncation mode enabled for all components - hardcoded ON
OSQP_USE_TRUNCATE=ON
NN_USE_TRUNCATE=ON
CONSTRAINTS_USE_TRUNCATE=ON

# ========================================
# QDLDL Sample Logging Configuration
# ========================================
# Set QDLDL_ENABLE_SAMPLE_LOGGING=ON to save QDLDL samples for precision analysis
# Samples will be saved to QDLDL_SAMPLE_OUTPUT_DIR
QDLDL_ENABLE_SAMPLE_LOGGING=${QDLDL_ENABLE_SAMPLE_LOGGING:-OFF}
QDLDL_SAMPLE_OUTPUT_DIR=${QDLDL_SAMPLE_OUTPUT_DIR:-"../../result/qdldl_samples"}

echo "========================================="
echo "Build Configuration:"
echo "  OSQP/QDLDL Float Type:"
echo "    OSQP_USE_FLOAT:       $OSQP_USE_FLOAT"
echo ""
echo "  Truncate Mode:"
echo "    OSQP_USE_TRUNCATE:        $OSQP_USE_TRUNCATE"
echo "    NN_USE_TRUNCATE:          $NN_USE_TRUNCATE"
echo "    CONSTRAINTS_USE_TRUNCATE: $CONSTRAINTS_USE_TRUNCATE"
echo ""
echo "  QDLDL Sample Logging:"
echo "    QDLDL_ENABLE_SAMPLE_LOGGING: $QDLDL_ENABLE_SAMPLE_LOGGING"
if [ "$QDLDL_ENABLE_SAMPLE_LOGGING" = "ON" ]; then
    echo "    QDLDL_SAMPLE_OUTPUT_DIR:    $QDLDL_SAMPLE_OUTPUT_DIR"
fi
echo "========================================="

cd External
cd osqp
rm -rf build
rm -rf lib
mkdir -p build lib
cd build
cmake .. \
    -DOSQP_USE_FLOAT=$OSQP_USE_FLOAT \
    -DOSQP_USE_TRUNCATE=$OSQP_USE_TRUNCATE \
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

# cmake로 빌드 설정 및 make 실행 (OSQP, Truncate 및 QDLDL 샘플 로깅 옵션 포함)
cmake .. \
    -DOSQP_USE_FLOAT=$OSQP_USE_FLOAT \
    -DOSQP_USE_TRUNCATE=$OSQP_USE_TRUNCATE \
    -DNN_USE_TRUNCATE=$NN_USE_TRUNCATE \
    -DCONSTRAINTS_USE_TRUNCATE=$CONSTRAINTS_USE_TRUNCATE \
    -DQDLDL_ENABLE_SAMPLE_LOGGING=$QDLDL_ENABLE_SAMPLE_LOGGING
make -j8

# 빌드 디렉토리에서 상위 디렉토리로 이동
cd ..
