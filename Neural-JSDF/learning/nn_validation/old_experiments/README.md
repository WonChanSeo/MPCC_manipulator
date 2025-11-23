# Historical Experiments Archive

This directory contains previous precision validation experiments. These have been superseded by the latest C++ precision tests in the parent directory.

## Experiment Categories

### 1. exp_mantissa/
**Exponent & Mantissa Variation Tests**

Tests various combinations of exponent and mantissa bits to understand their individual effects.

Files:
- `data_E*M*.npz` - Results for different exp/mant combinations
- `validation_exp_mantissa.py` - Test script
- `exp_mantissa_comparison_summary.npz` - Summary

Configurations tested:
- E4M4, E4M7, E5M5, E5M7, E5M10
- E6M6, E6M7
- E8M2, E8M4, E8M6, E8M7, E8M8, E8M12, E8M16, E8M23

### 2. mantissa_only/
**Mantissa-Only Reduction Tests (Python)**

Pure Python validation testing mantissa bit reduction while keeping exponent constant.

Files:
- `data_mantissa_*bit.npz` - Results for each mantissa bit count
- `validation_precision.py` - Main validation script
- `precision_comparison_summary.npz` - Summary
- `visualize_precision_comparison.py` - Visualization

Mantissa bits tested: 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 23

**Note**: These were Python-only tests using manual quantization. The latest C++ tests are more accurate as they test the actual C++ implementation.

### 3. fma_quant/
**FMA Quantization Tests**

Experiments comparing fused multiply-add (FMA) quantization vs per-operation quantization.

Files:
- `data_fma_E*.npz` - FMA quantization results
- `validation_fma_quant.py` - FMA validation script
- `test_fma_*.py` - Various FMA test scripts
- `debug_fma.py` - FMA debugging utilities
- `visualize_fma_*.py` - Visualization scripts
- `compare_perop_vs_fma.py` - Direct comparison

Key question: Does quantizing after FMA vs before each operation affect accuracy?

### 4. per_op_quant/
**Per-Operation Quantization Tests**

Tests quantizing after each individual operation (multiply, add) separately.

Files:
- `data_per_op_E*.npz` - Per-operation quantization results
- `validation_per_op_quant.py` - Validation script
- `visualize_per_op_quant.py` - Visualization

Compared configurations: E4M7, E5M7, E6M7, E8M2-E8M7, E8M23

### 5. misc_tests/
**Miscellaneous Tests and Utilities**

Various other experiments and analysis scripts.

Files:
- `compare_*.py` - Various comparison scripts
- `analyze*.py` - Analysis utilities
- `test_*.py` - Ad-hoc test scripts
- `validation*.py` - Old validation scripts
- `get_data*.py` - Data collection scripts
- `visualize*.py` - Old visualization scripts
- `bf16_summary.py` - BF16 specific analysis
- `optimize.py` - Optimization attempts

### 6. old_logs/
**Log Files**

Historical test output logs.

Files:
- `report.log` - Main validation report
- `fma_test_output*.log` - FMA test logs
- `test_output.log` - Various test outputs

## Why These Were Superseded

The latest C++ precision tests (`test_cpp_precision.py` in parent directory) are superior because:

1. **Tests actual C++ implementation**: Previous tests used Python simulations
2. **Full dataset**: 9900 samples vs smaller test sets
3. **Real-world accuracy**: Tests the actual `calculateMlpOutputBatch` function
4. **Comprehensive metrics**: Detailed error statistics including per-link analysis
5. **FlexFloat library**: Uses actual FlexFloat implementation, not manual quantization

## Historical Context

These experiments were conducted to:
- Understand precision vs accuracy tradeoffs
- Explore different quantization strategies
- Compare FMA vs per-op quantization
- Find optimal exponent/mantissa combinations
- Validate BF16 as a good default

The findings guided the decision to:
- Use BF16 (E8M7) as default precision
- Implement FlexFloat in C++ for configurable precision
- Focus on mantissa reduction (keeping 8 exponent bits)

## Preservation Rationale

These experiments are kept for:
1. **Reference**: Understanding how we arrived at current settings
2. **Methodology**: Showing different testing approaches
3. **Comparison**: Validating that C++ implementation matches Python predictions
4. **Education**: Learning about floating-point precision effects

If disk space is needed, these can be compressed or archived externally.

## Related Files

See parent directory's `QUICK_SUMMARY.txt` for a high-level overview of early findings.
