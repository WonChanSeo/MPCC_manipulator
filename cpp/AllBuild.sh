set -e

cd External
cd osqp
rm -rf build
rm -rf lib
mkdir -p build lib
cd build
cmake .. -DOSQP_USE_FLOAT=ON -DCMAKE_INSTALL_PREFIX=$(realpath ../lib)
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


# 1. 프로젝트 루트 디렉토리로 이동
cd ~/git/MPCC_manipulator/cpp

# 2. 환경 변수 설정 (터미널 세션용)
#    - 이 부분은 스크립트 실행에 필수적이지는 않지만, 수동으로 디버깅할 때를 위해 유지합니다.
export LD_LIBRARY_PATH="/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp/lib/lib":$LD_LIBRARY_PATH
export OsqpEigen_DIR="/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp_eigen"
echo "MPCC_manipulator Env set activated."
echo "Using OsqpEigen from: $OsqpEigen_DIR" # 경로 확인용 출력

# 3. 기존 build 디렉토리를 완전히 삭제
echo "Removing old build directory..."
rm -rf build
mkdir build
cd build

# 4. cmake 실행 시 경로를 직접 인자로 전달 (가장 중요한 부분!)
#    - CMAKE_PREFIX_PATH: osqp 라이브러리 경로를 알려줌
#    - OsqpEigen_DIR: OsqpEigen 경로를 알려줌
echo "Running cmake with explicit paths..."
cmake .. \
    -DCMAKE_PREFIX_PATH="/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp/lib" \
    -DOsqpEigen_DIR="/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp_eigen"

# 5. 빌드 실행
echo "Building project..."
make -j8

echo "Build complete."

# 빌드 디렉토리에서 상위 디렉토리로 이동
cd ..

