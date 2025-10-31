set -e

cd External
cd osqp
rm -rf build
rm -rf lib
mkdir -p build lib
cd build
cmake .. -DOSQP_USE_FLOAT=ON -DOSQP_USE_FLEXFLOAT=ON -DCMAKE_INSTALL_PREFIX=$(realpath ../lib)
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

# cmake로 빌드 설정 및 make 실행
cmake ..
make -j8

# 빌드 디렉토리에서 상위 디렉토리로 이동
cd ..

