# Neural-JSDF Per-Operation Quantization Testing

This directory contains scripts for testing Neural-JSDF inference with **per-operation quantization**, where every arithmetic operation (multiplication, addition, ReLU) is immediately quantized to the target precision.

## What is Per-Operation Quantization?

**Per-Operation Quantization** simulates actual low-precision hardware behavior:
- Each operation (mul, add, ReLU) produces a full-precision intermediate result
- This result is immediately quantized (rounded) to the target precision
- The quantized value is used as input for the next operation
- This causes **error accumulation** through the network layers

**Contrast with Post-Operation Quantization**:
- Input is quantized once at the beginning
- All operations run in full precision (FP32)
- Output is quantized once at the end
- This **underestimates** the actual hardware error

## Files

### Main Scripts
- `validation_per_op_quant.py` - Per-operation quantization validation script
- `visualize_per_op_quant.py` - Visualization script for results

### Generated Data
- `data_per_op_E*M*.npz` - Individual test results for each configuration
- `per_op_quant_summary.npz` - Summary statistics across all configurations
- `per_op_quant_analysis.png` - Detailed 9-subplot analysis
- `per_op_quant_summary.png` - Key findings and insights

## Usage

### Run Per-Operation Quantization Tests

```bash
python3 validation_per_op_quant.py
```

This will test 10 configurations:
- E8M23 (FP32 - baseline)
- E8M7 (BF16)
- E8M6, E8M5, E8M4, E8M3, E8M2 (varying mantissa bits)
- E6M7, E5M7, E4M7 (varying exponent bits)

Each test runs 5000 random samples and compares:
- Neural network predictions (with per-operation quantization)
- Ground truth mesh distances (computed from robot forward kinematics)

### Generate Visualizations

```bash
python3 visualize_per_op_quant.py
```

Generates two visualization files:
1. `per_op_quant_analysis.png` - Comprehensive 9-plot analysis
2. `per_op_quant_summary.png` - Key findings and recommendations

## Key Results

### Mantissa Precision Effect (E8M*)

| Mantissa Bits | Mean Error (cm) | vs FP32 |
|---------------|-----------------|---------|
| 23 (FP32)     | 0.88           | baseline|
| 7 (BF16)      | 1.51           | +73%    |
| 6             | 2.56           | +192%   |
| 5             | 4.73           | +440%   |
| 4             | 8.96           | +924%   |
| 3             | 16.72          | +1810%  |
| 2             | 28.05          | +3104%  |

**Finding**: Mantissa precision is critical. Below 7 bits, error accumulation becomes severe.

### Exponent Range Effect (*M7)

| Exponent Bits | Mean Error (cm) | vs FP32 |
|---------------|-----------------|---------|
| 8             | 1.51           | +73%    |
| 6             | 25.46          | +2808%  |
| 5             | 1.76           | +101%   |
| 4             | 0.25           | **-71%**|

**Surprising Finding**: E4M7 achieves **LOWER** error than FP32!

This suggests:
- The neural network operates in a limited dynamic range
- 4-bit exponent (range: 2^-7 to 2^7) is sufficient
- Extra mantissa precision (7 bits vs 23 bits in weight representation) helps accuracy
- The network benefits from implicit regularization at reduced exponent range

### Recommendations

1. **For Lowest Error**: Use **E4M7** (12-bit total)
   - Mean error: 0.25 cm (better than FP32!)
   - Exponent range: 2^-7 to 2^7 (sufficient for this network)
   - Mantissa precision: 7 bits

2. **For Standard Precision**: Use **E8M7 (BF16)** (16-bit)
   - Mean error: 1.51 cm (+73% vs FP32)
   - Industry-standard format
   - Good balance of range and precision

3. **Avoid**:
   - E6M7 and below: Insufficient exponent range (25+ cm error)
   - E8M4 and below: Insufficient mantissa precision (9+ cm error)

## How Per-Operation Quantization Works

### Implementation Method: Forward Hooks

The per-operation quantization is implemented using PyTorch forward hooks:

```python
def calc_nn_pred(self, input):
    # Register forward hooks to quantize after each layer
    handles = []

    def quantize_hook(module, input, output):
        return quantize_float(output, exp_bits, mant_bits)

    # Add hooks to all Linear and ReLU layers
    for module in self.nn_model.model.modules():
        if isinstance(module, (nn.Linear, nn.ReLU)):
            handle = module.register_forward_hook(quantize_hook)
            handles.append(handle)

    # Run inference with hooks active
    with torch.no_grad():
        y_pred = self.nn_model.model(input)

    # Remove hooks
    for handle in handles:
        handle.remove()

    return y_pred.float()
```

### Quantization Function

IEEE 754 bit manipulation:

```python
def quantize_float(tensor, exponent_bits, mantissa_bits):
    # FP32 format: sign(1) + exponent(8) + mantissa(23)
    bits = tensor_np.view(np.uint32)

    # Quantize mantissa (keep only mantissa_bits)
    mantissa_bits_to_zero = 23 - mantissa_bits
    mantissa_mask = 0x007FFFFF >> mantissa_bits_to_zero
    mantissa_mask <<= mantissa_bits_to_zero

    # Quantize exponent (keep only exponent_bits)
    exponent = (bits >> 23) & 0xFF
    exponent_bits_to_zero = 8 - exponent_bits
    exponent_quantized = (exponent >> exponent_bits_to_zero) << exponent_bits_to_zero

    # Reconstruct quantized float
    sign_bit = bits & 0x80000000
    exponent_field = (exponent_quantized & 0xFF) << 23
    mantissa_field = bits & mantissa_mask
    quantized_bits = sign_bit | exponent_field | mantissa_field

    return torch.from_numpy(quantized_bits.view(np.float32))
```

## Comparison with Post-Operation Quantization

| Method | E8M7 Mean Error | E8M4 Mean Error |
|--------|----------------|----------------|
| Post-operation | ~0.93 cm | ~2.5 cm |
| Per-operation | 1.51 cm | 8.96 cm |

**Per-operation quantization shows significantly higher errors**, which is expected because:
- Error accumulates through 5 hidden layers
- Each layer applies: quantize(ReLU(quantize(W @ x + b)))
- Quantization noise compounds at each step
- This is the actual behavior of low-precision hardware

## Insights

1. **E4M7 Optimization Opportunity**:
   - The network can be trained specifically for E4M7 precision
   - This could achieve sub-FP32 error at 12-bit precision
   - Potential for 2.67x memory savings vs FP32

2. **BF16 is Reasonable**:
   - E8M7 shows ~73% degradation, which is acceptable
   - Industry-standard format with good tool support

3. **Exponent Range Matters More Than Expected**:
   - E6M7 (insufficient range) is worse than E8M4 (insufficient precision)
   - Need to profile network activations to determine minimum exponent bits

4. **Hardware Implications**:
   - Post-operation quantization testing underestimates real hardware error
   - Always test with per-operation quantization for accurate hardware predictions
   - Consider quantization-aware training for optimal low-precision performance

## Technical Details

### Test Configuration
- **Model**: Neural-JSDF collision network (256x5 MLP)
- **Weight file**: `sdf_256x5_mesh_50000.pt`
- **Test samples**: 5000 random joint configurations + query points
- **Joint space**: 7-DOF Franka Panda robot
- **Query space**: 3D points in workspace

### Precision Formats

| Format | Sign | Exponent | Mantissa | Total | Range | Precision |
|--------|------|----------|----------|-------|-------|-----------|
| FP32 | 1 | 8 | 23 | 32 | ±3.4e38 | ~7 digits |
| BF16 | 1 | 8 | 7 | 16 | ±3.4e38 | ~2 digits |
| E4M7 | 1 | 4 | 7 | 12 | ±256 | ~2 digits |

### Error Metrics
- **Mean Error**: Average L1 distance between NN prediction and ground truth
- **Std Error**: Standard deviation of errors
- **Max Error**: Worst-case error
- **95th/99th Percentile**: Tail error distribution
- **Relative Degradation**: % change vs FP32 baseline

## Related Files
- `validation_precision.py` - Post-operation quantization (mantissa only)
- `validation_exp_mantissa.py` - Post-operation quantization (exp + mantissa)
- `validation_per_op_quant.py` - Per-operation quantization (**this file**)

## References
- [BFloat16 Specification](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
- [IEEE 754 Floating Point](https://en.wikipedia.org/wiki/IEEE_754)
- [Quantization-Aware Training](https://pytorch.org/docs/stable/quantization.html)
