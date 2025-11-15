set -e

# ========================================
# FlexFloat Configuration for OSQP/QDLDL
# ========================================
# Set USE_FLEXFLOAT=ON to enable FlexFloat precision testing
# Set USE_FLEXFLOAT=OFF to use standard float precision
USE_FLEXFLOAT=${USE_FLEXFLOAT:-OFF}

# Set QDLDL_USE_FLEXFLOAT=OFF to disable FlexFloat only in QDLDL
# (defaults to same as USE_FLEXFLOAT if not specified)
QDLDL_USE_FLEXFLOAT=${QDLDL_USE_FLEXFLOAT:-$USE_FLEXFLOAT}

# FlexFloat precision configuration (only used when USE_FLEXFLOAT=ON)
FF_EXPONENT_BITS=${FF_EXPONENT_BITS:-8}
FF_MANTISSA_BITS=${FF_MANTISSA_BITS:-23}

# ========================================
# FlexFloat Configuration for NN Inference
# ========================================
# Set NN_USE_FLEXFLOAT=ON to enable FlexFloat for NN inference
NN_USE_FLEXFLOAT=${NN_USE_FLEXFLOAT:-OFF}

# NN FlexFloat precision configuration (only used when NN_USE_FLEXFLOAT=ON)
NN_FF_EXPONENT_BITS=${NN_FF_EXPONENT_BITS:-8}
NN_FF_MANTISSA_BITS=${NN_FF_MANTISSA_BITS:-23}

# ========================================
# OpenMP Configuration
# ========================================
# Set number of OpenMP threads (defaults to all available cores)
OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
export OMP_NUM_THREADS

# ========================================
# CUDA GPU Acceleration Configuration
# ========================================
# Set USE_CUDA=ON to enable GPU acceleration for NN inference
USE_CUDA=${USE_CUDA:-OFF}

# Set CUDA compiler path (required for CUDA builds)
if [ "$USE_CUDA" = "ON" ]; then
    export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
fi

echo "========================================="
echo "Build Configuration:"
echo "  OSQP/QDLDL FlexFloat:"
echo "    USE_FLEXFLOAT:        $USE_FLEXFLOAT"
echo "    QDLDL_USE_FLEXFLOAT:  $QDLDL_USE_FLEXFLOAT"
if [ "$USE_FLEXFLOAT" = "ON" ]; then
    echo "    FF_EXPONENT_BITS:     $FF_EXPONENT_BITS"
    echo "    FF_MANTISSA_BITS:     $FF_MANTISSA_BITS"
fi
echo ""
echo "  NN Inference FlexFloat:"
echo "    NN_USE_FLEXFLOAT:     $NN_USE_FLEXFLOAT"
if [ "$NN_USE_FLEXFLOAT" = "ON" ]; then
    echo "    NN_FF_EXPONENT_BITS:  $NN_FF_EXPONENT_BITS"
    echo "    NN_FF_MANTISSA_BITS:  $NN_FF_MANTISSA_BITS"
fi
echo ""
echo "  OpenMP Configuration:"
echo "    OMP_NUM_THREADS:      $OMP_NUM_THREADS"
echo ""
echo "  CUDA GPU Acceleration:"
echo "    USE_CUDA:             $USE_CUDA"
echo "========================================="

cd External
cd osqp
rm -rf build
rm -rf lib
mkdir -p build lib
cd build
cmake .. \
    -DOSQP_USE_FLOAT=ON \
    -DOSQP_USE_FLEXFLOAT=$USE_FLEXFLOAT \
    -DQDLDL_USE_FLEXFLOAT=$QDLDL_USE_FLEXFLOAT \
    -DFF_EXPONENT_BITS=$FF_EXPONENT_BITS \
    -DFF_MANTISSA_BITS=$FF_MANTISSA_BITS \
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
cmake .. -DOSQP_IS_V1=ON -DCMAKE_INSTALL_PREFIX:PATH=$(realpath ../lib) -DCMAKE_PREFIX_PATH=$(realpath ../../../$localFolder_osqp/lib) -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
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

# cmake로 빌드 설정 및 make 실행 (NN FlexFloat 및 CUDA 옵션 포함)
cmake .. \
    -DNN_USE_FLEXFLOAT=$NN_USE_FLEXFLOAT \
    -DNN_FF_EXPONENT_BITS=$NN_FF_EXPONENT_BITS \
    -DNN_FF_MANTISSA_BITS=$NN_FF_MANTISSA_BITS \
    -DUSE_CUDA=$USE_CUDA
make -j8

# 빌드 디렉토리에서 상위 디렉토리로 이동
cd ..

