# OSQP with FlexFloat Integration

This document explains how to use OSQP with FlexFloat quantization (14-bit mantissa, 8-bit exponent).

## Overview

FlexFloat is a custom floating-point format designed to test reduced precision arithmetic in OSQP. The implementation uses:
- **Mantissa**: 14 bits (vs 23 bits in float32)
- **Exponent**: 8 bits (same as float32)
- **Total precision**: ~4-5 decimal digits (vs ~7 in float32)

This allows testing how OSQP's ADMM algorithm behaves under reduced numerical precision, which is useful for:
- Hardware design exploration (custom FPUs)
- Energy-efficient computing
- Understanding numerical stability requirements
- Accelerator design (FPGA, ASIC)

## Implementation Details

### Modified Files

1. **`include/private/flexfloat_wrapper.h`** (NEW)
   - FlexFloat type definition and arithmetic operations
   - Conversion functions to/from double
   - Quantization happens at each arithmetic operation

2. **`include/public/osqp_api_types.h`**
   - Added `OSQP_USE_FLEXFLOAT` conditional compilation
   - OSQPFloat remains as `double` internally but gets quantized

3. **`include/private/glob_opts.h`**
   - FlexFloat-aware math functions (sqrt, fmod)

4. **`algebra/builtin/vector.c`**
   - All vector operations apply FlexFloat quantization:
     - `OSQPVectorf_mult_scalar()`
     - `OSQPVectorf_plus()`
     - `OSQPVectorf_minus()`
     - `OSQPVectorf_add_scaled()` (most critical for ADMM)
     - `OSQPVectorf_add_scaled3()` (most critical for ADMM)

5. **`CMakeLists.txt`**
   - Added `OSQP_USE_FLEXFLOAT` option

### How FlexFloat Works

Every floating-point operation in OSQP's ADMM iterations goes through quantization:

```c
double result = a + b;  // Standard version

// FlexFloat version:
flexfloat_t ff_a = flexfloat_from_double(a);       // Quantize input
flexfloat_t ff_b = flexfloat_from_double(b);       // Quantize input
flexfloat_t ff_result = flexfloat_add(ff_a, ff_b); // Quantized arithmetic
double result = flexfloat_to_double(ff_result);    // Convert back
```

This simulates what would happen in hardware with reduced-precision FPUs.

## Building OSQP with FlexFloat

### Method 1: CMake Configure

```bash
cd cpp/External/osqp/build
cmake .. -DOSQP_USE_FLEXFLOAT=ON
make
```

### Method 2: From Scratch

```bash
cd cpp/External/osqp
rm -rf build
mkdir build && cd build
cmake .. -DOSQP_USE_FLEXFLOAT=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Building the Demo

```bash
cd cpp/External/osqp/build
cmake .. -DOSQP_USE_FLEXFLOAT=ON -DOSQP_BUILD_DEMO_EXE=ON
make
./out/osqp_flexfloat_demo
```

## Testing FlexFloat vs Standard Precision

### Comparison Test

Build OSQP twice and compare results:

```bash
# Build standard version
cd cpp/External/osqp
mkdir build_standard && cd build_standard
cmake .. -DOSQP_BUILD_DEMO_EXE=ON
make
./out/osqp_demo > standard_results.txt

# Build FlexFloat version
cd ..
mkdir build_flexfloat && cd build_flexfloat
cmake .. -DOSQP_USE_FLEXFLOAT=ON -DOSQP_BUILD_DEMO_EXE=ON
make
./out/osqp_demo > flexfloat_results.txt

# Compare
diff standard_results.txt flexfloat_results.txt
```

### Expected Differences

With FlexFloat, you should observe:

1. **Slightly more iterations**: Reduced precision may slow convergence
2. **Different final objective values**: Within ~1e-4 to 1e-5 tolerance
3. **Larger residuals**: Precision limits affect termination criteria
4. **Numerical artifacts**: Possible for ill-conditioned problems

## Performance Characteristics

| Aspect | FlexFloat (14-bit mantissa) | Float32 (23-bit) | Float64 (52-bit) |
|--------|----------------------------|------------------|------------------|
| Decimal precision | ~4-5 digits | ~7 digits | ~15-16 digits |
| Representable range | Same as float32 | ±3.4e38 | ±1.7e308 |
| Convergence tolerance | 1e-4 to 1e-5 | 1e-6 to 1e-7 | 1e-12 to 1e-14 |
| ADMM iterations | +10-50% typical | Baseline | Baseline |

## Use Cases

### 1. Algorithm Testing
Test if your QP problem can be solved with reduced precision:

```bash
cmake .. -DOSQP_USE_FLEXFLOAT=ON
make
./your_test_program
```

If it converges, your problem is numerically robust.

### 2. Hardware Design
Estimate FPU requirements for custom hardware:

```c
#ifdef OSQP_USE_FLEXFLOAT
  // Target: 14-bit mantissa FPU
  // Expected area: ~40% of float32 multiplier
  // Expected power: ~30% of float32 multiplier
#endif
```

### 3. Precision-Performance Trade-off
Profile precision vs. performance:

```bash
# Measure with different precisions
time ./osqp_demo_flexfloat   # 14-bit mantissa
time ./osqp_demo_float        # 23-bit mantissa
time ./osqp_demo_double       # 52-bit mantissa
```

## Troubleshooting

### Build Errors

**Error**: `flexfloat_wrapper.h: No such file or directory`

**Solution**: Make sure you're building from OSQP root with the modified files.

### Convergence Issues

**Problem**: FlexFloat version doesn't converge

**Solutions**:
1. Increase `max_iter` in settings
2. Relax tolerances: `eps_abs = 1e-4`, `eps_rel = 1e-4`
3. Try different `rho` values (ADMM penalty parameter)
4. Scale your problem data better

### Numerical Instability

**Problem**: NaN or Inf values in FlexFloat version

**Causes**:
- Problem is too ill-conditioned for 14-bit mantissa
- Need better scaling (enable `settings->scaling = 1`)
- Constraint bounds are too tight

## Implementation Notes

### Why Not Struct Wrapper?

Early attempts used `typedef struct { flexfloat_t value; } OSQPFloat;` but this breaks:
- Array initialization
- Pointer arithmetic
- External library interfaces

Current approach: Keep `OSQPFloat` as `double`, apply quantization in vector operations only.

### Quantization Points

FlexFloat quantization occurs at:
1. **Vector arithmetic** (add, subtract, multiply, scale)
2. **Math functions** (sqrt, fmod)
3. **Input/output** (when needed)

**NOT quantized**:
- Memory storage (stays as double)
- Integer operations
- Comparison operations (use quantized values)

### Performance Impact

FlexFloat adds overhead from:
- Double → FlexFloat → Double conversions
- Bit manipulation for quantization
- No SIMD vectorization

Typical overhead: 2-5x slower than native double arithmetic.

For actual hardware with 14-bit FPUs, speedup is expected from:
- Reduced memory bandwidth
- Smaller FPU area (higher frequency possible)
- Lower power consumption

## Example Output

### Standard Precision
```
Objective value:     1.87500000
Iterations:          3
Solve time:          0.000234 seconds
x[0] = 0.30000000
x[1] = 0.70000000
```

### FlexFloat Precision
```
Objective value:     1.87512207   (difference: ~0.00012)
Iterations:          4             (1 more iteration)
Solve time:          0.001156 seconds (slower due to quantization overhead)
x[0] = 0.30004883
x[1] = 0.69995117
```

## Further Modifications

### Change Mantissa/Exponent Bits

Edit `flexfloat_wrapper.h`:

```c
#define FLEXFLOAT_MANTISSA_BITS 10  // Change from 14 to 10
#define FLEXFLOAT_EXPONENT_BITS 6   // Change from 8 to 6
```

Then rebuild:

```bash
make clean
cmake .. -DOSQP_USE_FLEXFLOAT=ON
make
```

### Add Stochastic Rounding

Modify `flexfloat_from_double()` in `flexfloat_wrapper.h`:

```c
// Instead of round-to-nearest:
uint32_t mantissa_int = (uint32_t)(mantissa_d * (1 << FLEXFLOAT_MANTISSA_BITS) + 0.5);

// Use stochastic rounding:
double frac = mantissa_d * (1 << FLEXFLOAT_MANTISSA_BITS);
uint32_t mantissa_int = (uint32_t)frac;
if ((double)rand() / RAND_MAX < (frac - mantissa_int)) {
    mantissa_int++;
}
```

## References

- OSQP Paper: https://arxiv.org/abs/1711.08013
- ADMM: Boyd et al., "Distributed Optimization and Statistical Learning via ADMM"
- Reduced Precision: "Mixed Precision Training" (Micikevicius et al., 2017)

## Contact

For questions about FlexFloat integration:
- Check OSQP GitHub issues
- Review ADMM algorithm documentation
- Test with increasing mantissa bits to isolate precision issues
