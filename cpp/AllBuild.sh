set -e

# ========================================
# FlexFloat 설정
# ========================================
# FlexFloat 사용 여부 설정 (ON/OFF)
# - ON: 14-bit mantissa, 8-bit exponent (낮은 정밀도, 하드웨어 최적화 테스트용)
# - OFF: 표준 float32 (23-bit mantissa, 8-bit exponent)
USE_FLEXFLOAT=${1:-ON}  # 첫 번째 인자로 ON/OFF 지정 가능, 기본값 ON

if [ "$USE_FLEXFLOAT" = "ON" ]; then
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║   FlexFloat Mode: ENABLED                  ║"
    echo "║   Mantissa: 14 bits (~4-5 digits)          ║"
    echo "║   Exponent: 8 bits (same as float32)       ║"
    echo "║   Use case: Hardware optimization testing  ║"
    echo "╚════════════════════════════════════════════╝"
    echo ""
else
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║   FlexFloat Mode: DISABLED                 ║"
    echo "║   Using standard float32 precision         ║"
    echo "╚════════════════════════════════════════════╝"
    echo ""
fi

cd External
cd osqp
rm -rf build
rm -rf lib
mkdir -p build lib
cd build

# OSQP 빌드 with FlexFloat 옵션
echo "[1/3] Building OSQP..."

if [ "$USE_FLEXFLOAT" = "ON" ]; then
    cmake .. -DOSQP_USE_FLOAT=ON -DOSQP_USE_FLEXFLOAT=ON -DCMAKE_INSTALL_PREFIX=$(realpath ../lib)
else
    cmake .. -DOSQP_USE_FLOAT=ON -DCMAKE_INSTALL_PREFIX=$(realpath ../lib)
fi
make -j4
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

echo "[2/3] Building osqp-eigen..."

cd ~/git/MPCC_manipulator/cpp/External/osqp_eigen
rm -rf build
rm -rf lib
mkdir -p build lib
cd build
cmake .. -DOSQP_IS_V1=ON -DCMAKE_INSTALL_PREFIX:PATH=$(realpath ../lib) -DCMAKE_PREFIX_PATH=$(realpath ../../../$localFolder_osqp/lib) -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make -j4
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


echo "[3/3] Building MPCC_manipulator main project..."

cd ~/git/MPCC_manipulator/cpp
export LD_LIBRARY_PATH="/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp/lib/lib":$LD_LIBRARY_PATH
export OsqpEigen_DIR="/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp_eigen"
echo "MPCC_manipulator Env set activated."

# 기존 build 디렉토리를 삭제하고 새로 생성
rm -rf build
mkdir build
cd build

# cmake로 빌드 설정 및 make 실행
cmake ..
make -j8

# 빌드 디렉토리에서 상위 디렉토리로 이동
cd ..

# ========================================
# 빌드 완료 요약
# ========================================
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║   BUILD COMPLETED SUCCESSFULLY!            ║"
echo "╚════════════════════════════════════════════╝"
echo ""

if [ "$USE_FLEXFLOAT" = "ON" ]; then
    echo "✓ OSQP built with FlexFloat (14-bit mantissa)"
    echo "✓ All ADMM iterations use reduced precision"
    echo "✓ Expected effects:"
    echo "  - Lower numerical precision (~1e-4 to 1e-5)"
    echo "  - Potential for 10-50% more iterations"
    echo "  - Simulates custom FPU hardware"
    echo ""
    echo "To disable FlexFloat, run:"
    echo "  ./AllBuild.sh OFF"
else
    echo "✓ OSQP built with standard float32 precision"
    echo ""
    echo "To enable FlexFloat, run:"
    echo "  ./AllBuild.sh ON"
fi

echo ""
echo "Executable location:"
echo "  ~/git/MPCC_manipulator/cpp/build/"
echo ""

