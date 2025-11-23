# Neural Network Validation Tests

This directory contains precision validation tests for the Neural-JSDF collision detection network.

## Directory Structure

```
nn_validation/
├── cpp_precision_results/       # Latest C++ precision test results (Nov 2025)
│   ├── cpp_precision_E8M23.npz  # FP32 results (9900 samples)
│   ├── cpp_precision_E8M7.npz   # BF16 results
│   ├── cpp_precision_E8M6.npz   # E8M6 results
│   ├── cpp_precision_E8M5.npz   # E8M5 results
│   ├── cpp_precision_E8M4.npz   # E8M4 results
│   ├── cpp_precision_summary.npz
│   └── comparison_report.txt    # Detailed analysis
│
├── old_experiments/             # Historical experiments
│   ├── exp_mantissa/           # Exponent & mantissa variation tests
│   ├── mantissa_only/          # Mantissa-only reduction tests
│   ├── fma_quant/              # FMA vs per-op quantization tests
│   ├── per_op_quant/           # Per-operation quantization tests
│   ├── misc_tests/             # Various other experiments
│   └── old_logs/               # Old log files
│
├── test_cpp_precision.py       # Main C++ precision test script
├── fk_num.py                   # Forward kinematics utilities
└── README.md                   # This file
```

## Latest Results (Nov 2025)

### C++ calculateMlpOutputBatch Precision Test

Tested 5 FlexFloat precision configurations with 9900 samples from data_mesh_test.mat:

| Config | Exp | Mant | Mean Error (cm) | Degradation vs FP32 |
|--------|-----|------|-----------------|---------------------|
| FP32   | 8   | 23   | 0.65            | baseline            |
| BF16   | 8   | 7    | 0.79            | +22.1%              |
| E8M6   | 8   | 6    | 1.21            | +87.2%              |
| E8M5   | 8   | 5    | 2.58            | +297.5%             |
| E8M4   | 8   | 4    | 6.75            | +939.7%             |

**Recommendation**: BF16 (E8M7) provides the best tradeoff with only 22% accuracy loss and 2x memory reduction.

See [cpp_precision_results/comparison_report.txt](cpp_precision_results/comparison_report.txt) for detailed analysis.

## Running Tests

### C++ Precision Test

```bash
cd /home/mms-wonchan/git/MPCC_manipulator/Neural-JSDF/learning/nn_validation
python3 test_cpp_precision.py
```

This will:
1. Build C++ library with each precision configuration
2. Run inference on 9900 test samples
3. Calculate errors vs ground truth
4. Save results to `cpp_precision_results/`

**Note**: Full test takes ~5 hours (each config ~1 hour)

## Key Findings

1. **BF16 is optimal** for current use case:
   - Mean error: 0.79 cm (acceptable for collision detection)
   - Memory: 50% reduction vs FP32
   - 95th percentile error: 2.23 cm

2. **E8M6 is marginal**:
   - 87% error increase
   - Only consider if memory extremely constrained

3. **E8M5 and E8M4 NOT recommended**:
   - E8M5: 2.58 cm mean error (4x worse than FP32)
   - E8M4: 6.75 cm mean error (10x worse, max 52 cm)

## Dependencies

- Python 3.8+
- numpy
- scipy
- torch (for old validation scripts)
- MPCC_WRAPPER (C++ Python bindings)

## Historical Experiments

See `old_experiments/` for previous studies:
- Exponent vs mantissa bit variations
- FMA vs per-operation quantization
- BF16 vs FP32 comparisons
- Various optimization attempts

These are kept for reference but superseded by the latest C++ precision tests.

## Network Architecture

- Input: 10 dims (7 joint angles + 3 query point)
- Hidden: 4 layers x 256 units each
- Output: 9 dims (SDF distances for 9 links)
- Activation: ReLU with NeRF encoding [x, sin(x), cos(x)]
- Training: 50K samples from mesh dataset

## Dataset

- Source: `Neural-JSDF/learning/data-sampling/datasets/data_mesh_test.mat`
- Samples: 9900
- Format: [7 joints | 3 query | 9 ground truth distances (m)]
- Ground truth: Mesh-based SDF calculation

## Contact

For questions about these tests, refer to the git history or contact the maintainer.
