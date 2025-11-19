# C++ vs Python Inference Validation

This directory contains scripts and tools for validating C++ `calculateMlpOutputBatch` against Python implementation using the Neural-JSDF validation dataset.

## Overview

The validation system:
1. Generates the same test dataset used for quantization validation
2. Runs both Python and C++ inference on the same inputs
3. Compares outputs element-by-element
4. Creates visualizations of differences

## Files Created

### Python Scripts
- `validate_cpp_inference.py` - Generates test dataset and Python reference outputs
- `compare_cpp_python.py` - Compares C++ and Python outputs, creates visualization

### C++ Files
- `/cpp/test_validation_inference.cpp` - C++ test program
- `/cpp/include/Constraints/EnvCollision/EnvCollisionModel.h` - Added logging functions
- `/cpp/src/Constraints/EnvCollision/EnvCollisionModel.cpp` - Implemented logging functions

### Data Files (Generated)
- `cpp_validation_input.txt` - Input data (10 × 100, column-major)
- `python_validation_output.txt` - Python output (9 × 100, column-major)
- `python_validation_reference.npz` - NumPy archive with input/output
- `cpp_validation_output.txt` - C++ output (9 × 100, column-major)
- `cpp_validation_log.txt` - Detailed I/O log from C++
- `cpp_vs_python_comparison.png` - Visualization

## Usage

### Step 1: Generate Test Data

```bash
cd /path/to/Neural-JSDF/learning/nn_validation
python3 validate_cpp_inference.py
```

This will:
- Load the FMA validation model (E8M23 = FP32, no quantization)
- Generate 100 test samples (same as validation scripts)
- Run Python inference
- Export data for C++

### Step 2: Compile C++ Test

```bash
cd /path/to/cpp/build
cmake ..
make
```

### Step 3: Run C++ Test

```bash
cd /path/to/cpp/build
./test_validation_inference
```

This will:
- Load input data from `cpp_validation_input.txt`
- Initialize Neural-JSDF model (10 → [256, 256, 256, 256] → 9, NERF enabled)
- Run batch inference
- Save full batch output to `cpp_validation_output.txt`
- Save detailed log to `cpp_validation_log.txt`

### Step 4: Compare Results

```bash
cd /path/to/Neural-JSDF/learning/nn_validation
python3 compare_cpp_python.py
```

This will:
- Load Python and C++ outputs
- Calculate absolute and relative differences
- Check tolerance thresholds
- Create visualization (`cpp_vs_python_comparison.png`)

## Expected Results

### Perfect Match (FP32, no quantization)
If both use FP32 without quantization:
- Max difference: < 1e-5
- Mean difference: < 1e-6

### BF16 with FP32 Accumulators (Eigen::bfloat16)
If C++ uses Eigen::bfloat16 (default, no FlexFloat):
- Inputs/weights quantized to BF16
- Accumulators use FP32
- Expected difference: ~0.01-0.1 cm

### BF16 with FMA-level Quantization (FlexFloat)
If C++ uses FlexFloat with E8M7:
- Every FMA operation quantized
- Expected difference: Larger, ~1-10 cm

## Implementation Details

### C++ Additions

#### Header (`EnvCollisionModel.h`)
```cpp
// Input/Output logging functions
void saveInputOutputLog(const std::string& filename,
                        const Eigen::MatrixXd& inputs,
                        const Eigen::MatrixXd& outputs) const;
void enableLogging(bool enable);
bool isLoggingEnabled() const;
Eigen::MatrixXd getBatchOutput() const;  // Get full batch output
```

#### Implementation (`EnvCollisionModel.cpp`)
- `saveInputOutputLog()`: Saves inputs/outputs to text file with metadata
- `getBatchOutput()`: Returns `mlp_.batch_output` (full 9×100 matrix before min selection)

### Data Format

All text files use **row-major** format:
- Each row = one dimension (input/output feature)
- Each column = one sample
- Space-separated values
- 10 decimal precision

Example:
```
# Input dimension 0 (joint 0):
-1.234567890 0.987654321 ...  (100 values)
# Input dimension 1 (joint 1):
...
```

### Python Model

Uses `FMAQuantValidation` with E8M23 (FP32):
- Same architecture as quantization validation
- NERF encoding enabled
- No quantization applied (exponent_bits=8, mantissa_bits=23)

### C++ Model

Uses `EnvCollNNmodel` with:
- Architecture: 10 → [256, 256, 256, 256] → 9
- NERF encoding enabled
- Precision: Depends on compilation flags
  - Default: Eigen::bfloat16 with FP32 accumulators
  - With `NN_USE_FLEXFLOAT`: FlexFloat with configurable precision

## Troubleshooting

### "Failed to open input file"
- Make sure you run `validate_cpp_inference.py` first
- Check that `cpp_validation_input.txt` exists in `Neural-JSDF/learning/nn_validation/`

### "Output shape mismatch"
- Ensure C++ test is saving full batch output (not just min per row)
- Check that `getBatchOutput()` is called

### Large differences (> 1 cm)
- Check if FlexFloat is enabled in C++
- Verify quantization settings match
- Compare precision: Python FP32 vs C++ BF16

### Compilation errors
- Make sure all new functions are declared in header
- Check that `#include <iomanip>` is present in C++ test

## Notes

- The validation dataset uses **same random seed (42)** for reproducibility
- Input range matches validation scripts: q_min to q_max
- C++ `calculateMlpOutputBatch` normally returns **minimum per row** for MPCC
- For validation, we extract **full batch output** using `getBatchOutput()`
- Comparison is done on **all 900 values** (100 samples × 9 outputs)

## References

- Validation scripts: `validation_fma_quant.py`, `validation_per_op_quant.py`
- C++ implementation: `EnvCollisionModel.cpp`
- FlexFloat library: `cpp/External/flexfloat/`
