# FlexFloat Implementation Status

## Overview
FlexFloat has been integrated into OSQP to enable reduced-precision arithmetic testing for hardware optimization research.

## Current Configuration
- **Mantissa bits**: 23 (equivalent to float32)
- **Exponent bits**: 8 (equivalent to float32)
- **Application scope**: ADMM vector operations ONLY
- **Linear solver (QDLDL)**: Uses standard double precision (FlexFloat DISABLED)

## Implementation Strategy

### What FlexFloat Does
FlexFloat is applied **ONLY** to vector arithmetic operations in the ADMM solver loop:
- `OSQPVectorf_add_scaled()` - Scaled vector addition
- `OSQPVectorf_norm_inf()` - Infinity norm calculation
- `OSQPVectorf_dot_prod()` - Dot product
- All other vector operations in `algebra/builtin/vector.c`

### What Remains Full Precision
The following operations use standard **double precision**:
1. **QP Problem Setup** (`osqp_setup`)
   - Matrix construction (P, A)
   - Bound vectors (l, u)
   - Constraint validation

2. **Linear System Solver** (QDLDL)
   - LDL^T factorization
   - Forward/backward substitution
   - System matrix operations

3. **Memory Operations**
   - Simple assignment `bv[i] = av[i]`
   - Memory allocation
   - Data copying

## Known Issues

### Current Problem
Even with 23-bit mantissa (float32 equivalent precision), the system fails with:
```
ERROR in validate_data: Lower bound at index 142 is greater than upper bound: 1.5967e-314 > 5.3568e-315
```

### Analysis
The error values (~10^-314) are far beyond FlexFloat underflow range (~10^-38 for float32). This suggests:

1. **Possible Root Causes**:
   - The input data to OSQP contains corrupted/uninitialized values
   - Your MPCC code generates these extreme values when FlexFloat is enabled
   - Memory corruption or timing issue exposed by FlexFloat code path changes

2. **Evidence**:
   - Works correctly with FlexFloat OFF
   - Fails even at full float32 precision (23-bit mantissa)
   - Error occurs during QP validation (before ADMM iterations)
   - QDLDL uses full double precision, so not a factorization issue

## Files Modified

### Core FlexFloat Implementation
- `include/private/flexfloat_wrapper.h` - FlexFloat type and operations
- `include/private/flexfloat.h` - Compatibility header

### OSQP Integration
- `algebra/builtin/vector.c` - FlexFloat wrappers for all vector operations
- `src/osqp_api.c` - Debug printf at ADMM start
- `CMakeLists.txt` - `OSQP_USE_FLEXFLOAT` option

### QDLDL Configuration
- `algebra/_common/lin_sys/qdldl/qdldl.cmake` - FlexFloat explicitly DISABLED for QDLDL

### Build System
- `AllBuild.sh` - Toggle FlexFloat with `./AllBuild.sh ON|OFF`

## Next Steps

### Immediate Debugging
1. **Check Input Data**: Add logging in your MPCC code to verify QP bounds before calling `osqp_setup()`
2. **Memory Validation**: Run with valgrind or sanitizers to detect memory corruption
3. **Reduce Mantissa**: Try 14-bit or 20-bit mantissa to see if behavior changes

### Recommended Approach
```cpp
// In your MPCC code, before osqp_setup():
for (int i = 0; i < m; i++) {
    if (std::abs(lower[i]) < 1e-200 || std::abs(upper[i]) < 1e-200) {
        printf("WARNING: Extreme bound value at index %d: l=%.4e, u=%.4e\n",
               i, lower[i], upper[i]);
    }
    if (lower[i] > upper[i]) {
        printf("ERROR: Invalid bounds at index %d: l=%.4e > u=%.4e\n",
               i, lower[i], upper[i]);
    }
}
```

## Configuration Changes

### To Adjust Mantissa Bits
Edit [`flexfloat_wrapper.h:21`](include/private/flexfloat_wrapper.h#L21):
```c
#define FLEXFLOAT_MANTISSA_BITS 14  // Change this value
```

Values to try:
- 14 bits: Very reduced precision
- 20 bits: Medium precision
- 23 bits: Float32 equivalent (current)

### To Enable FlexFloat in QDLDL (Not Recommended)
Edit [`qdldl.cmake:54-74`](algebra/_common/lin_sys/qdldl/qdldl.cmake#L54-L74) and uncomment the FlexFloat section.

**Warning**: This will apply FlexFloat to linear system factorization, which may cause numerical instability.

## Build and Test

```bash
# Enable FlexFloat
cd /home/mms-wonchan/git/MPCC_manipulator
./AllBuild.sh ON

# Run test
./test

# Disable FlexFloat
./AllBuild.sh OFF
```

## References
- FlexFloat paper: "FlexFloat: A Software Library for Transprecision Computing"
- OSQP documentation: https://osqp.org/
- Implementation discussion: See `FLEXFLOAT_README.md`
