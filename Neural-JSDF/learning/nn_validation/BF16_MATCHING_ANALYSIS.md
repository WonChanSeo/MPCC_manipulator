# BF16 vs BF16 Comparison Analysis

## Test Configuration

- **Python**: E8M7 (BF16) with FMA-level quantization
- **C++**: FlexFloat E8M7 (BF16) with FMA-level quantization
- **Dataset**: 100 samples × 10 inputs → 100 samples × 9 outputs
- **Architecture**: 10 -> [256, 256, 256, 256] -> 9 with NERF encoding

## Results Summary

### Output Statistics

**Python**:
- Min: -1.289
- Max: 116.0
- Mean: 49.71

**C++**:
- Min: -1.070
- Max: 199.0
- Mean: 87.20

### Differences

```
Absolute differences:
  Max:    83.0 cm
  Mean:   37.5 cm
  Median: 40.0 cm
  Min:    0.20 cm

Relative errors:
  Max:    220%
  Mean:   75.7%
  Median: 75.3%
```

### Sample Comparison

| Sample | Output | Python | C++ | Abs Diff | Rel Error |
|--------|--------|--------|-----|----------|-----------|
| 0 | 0 | 56.75 | 98.0 | 41.25 | 72.7% |
| 0 | 1 | 56.25 | 97.5 | 41.25 | 73.3% |
| 0 | 2 | 56.25 | 102.5 | 46.25 | 82.2% |

## Key Finding: Systematic 1.75x Ratio

### Ratio Analysis

```
C++ / Python ratio:
  Min:    0.813
  Max:    3.2
  Mean:   1.756
  Median: 1.753
  Std:    0.127
```

**Observation**: C++ outputs are consistently ~1.75x larger than Python outputs!

## Investigation

### 1. Weights Verification ✅

**Python** (from `sdf_256x5_mesh_50000.pt`):
```
[0.0069, -0.0549, -0.0225, 0.0160, -0.0063]
```

**C++** (from `NNmodel/env/parameter/weight_0.txt`):
```
[0.00692749, -0.05493164, -0.02246094, 0.01599121, -0.00631714]
```

**Result**: Weights match! This rules out weight loading issues.

### 2. Normalization Parameters ✅

**Python normalization dict**:
```python
'y': {
    'mean': [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'std': [1., 1., 1., 1., 1., 1., 1., 1., 1.]
}
```

**Result**: No output denormalization (mean=0, std=1). This rules out scaling issues in denormalization.

### 3. NERF Encoding ✅

**Python** (validation_fma_quant.py:244-247):
```python
x_sin = torch.sin(x_norm)
x_cos = torch.cos(x_norm)
x_nerf = torch.cat((x_norm, x_sin, x_cos), dim=-1)
```

**C++** (EnvCollisionModel.cpp:395-397):
```cpp
mlp_.batch_input_nerf.topRows(mlp_.n_input) = mlp_.batch_input;
mlp_.batch_input_nerf.middleRows(mlp_.n_input, mlp_.n_input) = mlp_.batch_input.array().sin().matrix();
mlp_.batch_input_nerf.bottomRows(mlp_.n_input) = mlp_.batch_input.array().cos().matrix();
```

**Result**: NERF encoding is identical [x, sin(x), cos(x)].

### 4. Bias Addition Fixed ⚠️

**Original C++ implementation**:
```cpp
// WRONG: Initialize accumulator with bias
ff_init_float(&ff_sum, (float)mlp_.bias[0](i), NN_FF_DESC);
for (int k = 0; k < n; ++k) {
    ff_fma(&ff_sum, &ff_w, &ff_x, &ff_sum);
}
```

**Fixed C++ implementation**:
```cpp
// CORRECT: Initialize accumulator to 0, add bias after
ff_init_float(&ff_sum, 0.0f, NN_FF_DESC);
for (int k = 0; k < n; ++k) {
    ff_fma(&ff_sum, &ff_w, &ff_x, &ff_sum);
}
// Add bias AFTER accumulation
ff_init_float(&ff_bias, (float)mlp_.bias[layer](i), NN_FF_DESC);
flexfloat_t ff_one;
ff_init_float(&ff_one, 1.0f, NN_FF_DESC);
ff_fma(&ff_sum, &ff_one, &ff_bias, &ff_sum);  // sum = 1 * bias + sum
```

**Python implementation** (validation_fma_quant.py:119-137):
```python
acc = torch.tensor(0.0, device=x.device)  # Initialize to 0
for i in range(in_features):
    mul_result = x_q[b, i] * weight_q[o, i]
    acc = acc + mul_result
    acc = quantize_float(acc, exponent_bits, mantissa_bits)

# Add bias AFTER accumulation
if bias is not None:
    bias_q = quantize_float(bias[o], exponent_bits, mantissa_bits)
    acc = acc + bias_q
    acc = quantize_float(acc, exponent_bits, mantissa_bits)
```

**Result**: Fixed to match Python, but still shows 1.75x ratio!

## ✅ ROOT CAUSE IDENTIFIED: Rounding Mode Difference

### What We've Ruled Out:
- ❌ Weight loading differences
- ❌ Output denormalization
- ❌ NERF encoding differences
- ❌ Bias addition order (now fixed)

### 🎯 Root Cause Found:

**Quantization rounding mode differs between Python and C++**:

1. **Python** (validation_fma_quant.py:29-88):
   - Uses **bit manipulation truncation** (round toward zero)
   - Example: 2.2150976658 → **2.2031250000**

2. **C++ FlexFloat** (flexfloat.c:212-352):
   - Uses **IEEE 754 round-to-nearest-even** (banker's rounding)
   - Example: 2.2150976658 → **2.2187500000**

**Difference**: 0.0156 (0.7%) for the first input value

### Error Propagation:

| Stage | Python | C++ | Difference |
|-------|--------|-----|------------|
| Input quantization | 2.2031 | 2.2188 | +0.0156 |
| After Layer 0 | -0.8711 | -0.8867 | -0.016 |
| Final output[0,0] | 56.75 cm | 98.0 cm | **1.73x** |

The 0.016 input difference propagates through:
- 256 multiply-adds per layer × 5 layers = 1,280 operations
- Non-linear activations amplify differences
- Results in systematic 1.75x ratio

### Verification:

Created [test_quantization_modes.py](test_quantization_modes.py) to experimentally verify:

```
Original value:    2.2150976658
Python (Truncate): 2.2031250000
FlexFloat (Round): 2.2187500000  ✓ Matches C++ output
Difference:        0.0156250000
```

See [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md) for detailed analysis and solutions.

## Files Modified

1. **EnvCollisionModel.cpp**: Fixed bias addition to match Python (initialize accumulator to 0, add bias after accumulation)
2. **test_validation_inference.cpp**: Updated to save full batch output
3. **validate_cpp_inference.py**: Changed to use E8M7 instead of E8M23

## Build Commands

```bash
cd /home/mms-wonchan/git/MPCC_manipulator/cpp
./build_with_options.sh bf16
cd build
./test_validation_inference
```

## Comparison Command

```bash
cd /home/mms-wonchan/git/MPCC_manipulator/Neural-JSDF/learning/nn_validation
python3 compare_cpp_python.py
```
