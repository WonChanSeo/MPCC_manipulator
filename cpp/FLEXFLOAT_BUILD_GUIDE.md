# FlexFloat Build Guide

## Overview

This guide explains how to build OSQP with FlexFloat precision control for testing numerical precision.

## Build Options

### Environment Variables

- `USE_FLEXFLOAT`: Enable/disable FlexFloat precision control
  - `ON`: Enable FlexFloat (default: `OFF`)
  - `OFF`: Use standard float precision

- `FF_EXPONENT_BITS`: Number of exponent bits (default: `8`)
- `FF_MANTISSA_BITS`: Number of mantissa bits (default: `23` - IEEE 754 single precision)

## Build Examples

### 1. Standard Build (FlexFloat OFF)

```bash
./AllBuild.sh
```

This builds with standard float precision (32-bit IEEE 754).

### 2. Build with FlexFloat (Default Precision - IEEE 754 Single)

```bash
USE_FLEXFLOAT=ON ./AllBuild.sh
```

This enables FlexFloat with:
- Exponent: 8 bits
- Mantissa: 23 bits (same as IEEE 754 single precision for verification)

### 3. Build with Custom Precision

```bash
USE_FLEXFLOAT=ON FF_EXPONENT_BITS=6 FF_MANTISSA_BITS=10 ./AllBuild.sh
```

This enables FlexFloat with:
- Exponent: 6 bits
- Mantissa: 10 bits

### 4. Test Different Precisions

#### IEEE 754 Half Precision (binary16)
```bash
USE_FLEXFLOAT=ON FF_EXPONENT_BITS=5 FF_MANTISSA_BITS=10 ./AllBuild.sh
```

#### BFloat16 Precision
```bash
USE_FLEXFLOAT=ON FF_EXPONENT_BITS=8 FF_MANTISSA_BITS=7 ./AllBuild.sh
```

#### 14-bit Mantissa (Custom Low Precision)
```bash
USE_FLEXFLOAT=ON FF_MANTISSA_BITS=14 ./AllBuild.sh
```

## How It Works

### FlexFloat Precision Configuration Flow

1. **AllBuild.sh** reads environment variables:
   - `USE_FLEXFLOAT`, `FF_EXPONENT_BITS`, `FF_MANTISSA_BITS`

2. **OSQP CMakeLists.txt** receives the configuration:
   - Sets compile definitions: `-DFF_exponent_bits=...`, `-DFF_mantissa_bits=...`

3. **QDLDL CMakeLists.txt** inherits the configuration:
   - Receives `FF_EXPONENT_BITS` and `FF_MANTISSA_BITS` from OSQP
   - Applies the same definitions to QDLDL object library

4. **Source Code** uses the configuration:
   - `qdldl.c` and `vector.c` use `FF_exponent_bits` and `FF_mantissa_bits` macros
   - All FlexFloat operations use the configured precision

## Testing Workflow

### Step 1: Run with Standard Precision (Baseline)
```bash
./AllBuild.sh
./build/MPCC
```
Save the results as baseline.

### Step 2: Run with FlexFloat (Low Precision)
```bash
USE_FLEXFLOAT=ON FF_MANTISSA_BITS=10 ./AllBuild.sh
./build/MPCC
```
Compare results with baseline to see precision impact.

### Step 3: Sweep Different Precisions
```bash
for bits in 7 10 14 16 20 23; do
    echo "Testing mantissa bits: $bits"
    USE_FLEXFLOAT=ON FF_MANTISSA_BITS=$bits ./AllBuild.sh
    ./build/MPCC > results_mantissa_${bits}.txt
done
```

## Key Files Modified

- `AllBuild.sh`: Reads environment variables and passes to CMake
- `osqp/CMakeLists.txt`: Configures FlexFloat for OSQP
- `qdldl/CMakeLists.txt`: Configures FlexFloat for QDLDL
- `osqp/algebra/_common/lin_sys/qdldl/qdldl.cmake`: Passes config to QDLDL
- `qdldl/src/qdldl.c`: Uses FlexFloat in reciprocal and factorization
- `osqp/algebra/builtin/vector.c`: Uses FlexFloat in vector operations

## Common Precision Configurations

| Configuration | Exponent | Mantissa | Description |
|--------------|----------|----------|-------------|
| Half (FP16) | 5 | 10 | IEEE 754 half precision |
| BFloat16 | 8 | 7 | Brain Float 16 |
| TensorFloat-32 | 8 | 10 | NVIDIA TF32 |
| Custom Low | 8 | 14 | Low precision test |
| Single (FP32) | 8 | 23 | IEEE 754 single precision (default) |

## Notes

- FlexFloat emulates lower precision on double precision hardware
- Performance will be slower due to emulation overhead
- All arithmetic operations (add, mul, fma) respect the configured precision
- The magic number for reciprocal approximation remains unchanged (optimized for FP32)
- Newton-Raphson iterations are increased for lower precision (4 iterations instead of 3)
