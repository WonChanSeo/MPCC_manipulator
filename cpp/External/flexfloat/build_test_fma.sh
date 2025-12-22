#!/bin/bash

# Build FlexFloat FMA test
# Usage: ./build_test_fma.sh [debug|release]

BUILD_TYPE=${1:-release}

CC=gcc
CFLAGS="-Wall -Wextra -I./include -std=c99"

if [ "$BUILD_TYPE" == "debug" ]; then
    CFLAGS="$CFLAGS -g -O0"
else
    CFLAGS="$CFLAGS -O2"
fi

# Link with math library
LDFLAGS="-lm"

echo "Building FlexFloat FMA test ($BUILD_TYPE)..."
echo "CFLAGS: $CFLAGS"

$CC $CFLAGS -o test_fma test_fma.c src/flexfloat.c $LDFLAGS

if [ $? -eq 0 ]; then
    echo "Build successful: test_fma"
    echo ""
    echo "Run with: ./test_fma [testdata_dir]"
    echo "Default testdata_dir: testdata"
else
    echo "Build failed!"
    exit 1
fi
