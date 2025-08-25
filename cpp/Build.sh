#!/bin/bash
set -e  # 명령어가 실패할 경우 스크립트 종료

export LD_LIBRARY_PATH="/home/mms-wonchan/MPCC_manipulator/cpp/External/osqp/lib/lib":$LD_LIBRARY_PATH
    export OsqpEigen_DIR="/home/mms-wonchan/MPCC_manipulator/cpp/External/osqp_eigen"
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
