#!/usr/bin/env python3
"""
Debug script to compare FMA precision between Python and C++.
Tests the hypothesis that the difference comes from extended precision in C's fma().

This script compares:
1. Python with np.float64 FMA (current implementation)
2. Python with pure FP32 operations (no extended precision)
3. Expected C++ behavior with hardware fma() (80-bit or FMA hardware)
"""

import numpy as np
import torch
import struct

def quantize_float(x, exponent_bits=8, mantissa_bits=7):
    """Quantize to custom float format (E8M7 = bfloat16) using truncation."""
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = np.array(x, dtype=np.float32)

    # Ensure we have at least 1D array for view operations
    scalar_input = x_np.ndim == 0
    if scalar_input:
        x_np = x_np.reshape(1)

    # E8M7 truncation: keep 8 exponent bits and 7 mantissa bits
    # FP32 has 8 exponent and 23 mantissa bits
    # We need to zero out the lower 16 mantissa bits (23 - 7 = 16)
    x_bytes = x_np.view(np.uint32)
    mask = 0xFFFF0000  # Keep sign(1) + exponent(8) + mantissa(7), zero lower 16
    x_truncated = (x_bytes & mask).view(np.float32)

    if scalar_input:
        x_truncated = x_truncated[0]

    if isinstance(x, torch.Tensor):
        return torch.from_numpy(np.array(x_truncated)).to(x.device)
    return x_truncated

def fma_float64(a, b, c):
    """FMA using float64 for extended precision intermediate (current Python implementation)."""
    result = np.float64(a) * np.float64(b) + np.float64(c)
    return float(result)

def fma_float32(a, b, c):
    """FMA using pure float32 operations (no extended precision)."""
    return np.float32(np.float32(a) * np.float32(b)) + np.float32(c)

def load_validation_data():
    """Load the validation data."""
    data = np.load('python_validation_reference.npz')
    inputs = torch.from_numpy(data['input']).float()  # Shape: (100, 10)

    # Load weights from text files
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_dir = os.path.join(script_dir, '..', '..', '..', 'cpp', 'NNmodel', 'env', 'parameter') + '/'

    # Architecture: 10 -> [256, 256, 256, 256] -> 9
    # With NERF: 30 -> 256 -> 256 -> 256 -> 256 -> 9
    layer_sizes = [(256, 30), (256, 256), (256, 256), (256, 256), (9, 256)]

    weights = []
    biases = []

    for i, (out_size, in_size) in enumerate(layer_sizes):
        # Load weight
        w_data = np.loadtxt(f'{weight_dir}weight_{i}.txt')
        w = torch.from_numpy(w_data.reshape(out_size, in_size)).float()
        weights.append(w)

        # Load bias
        b_data = np.loadtxt(f'{weight_dir}bias_{i}.txt')
        b = torch.from_numpy(b_data).float()
        biases.append(b)

    return inputs, weights, biases

def trace_first_layer_fma(input_vec, weight, bias, use_float64_fma=True):
    """
    Trace FMA operations for the first layer.
    Returns detailed trace for each output neuron.
    """
    in_features = weight.shape[1]
    out_features = weight.shape[0]

    # Quantize inputs
    input_q = quantize_float(input_vec)
    weight_q = quantize_float(weight)
    bias_q = quantize_float(bias)

    traces = []

    for o in range(out_features):
        acc = 0.0
        step_trace = []

        for i in range(in_features):
            w_val = float(weight_q[o, i].item())
            x_val = float(input_q[i].item())

            acc_before = acc

            if use_float64_fma:
                acc = fma_float64(w_val, x_val, acc)
            else:
                acc = fma_float32(w_val, x_val, acc)

            acc_after = acc

            # Quantize accumulator
            acc_tensor = torch.tensor(acc)
            acc_tensor = quantize_float(acc_tensor)
            acc = float(acc_tensor.item())

            step_trace.append({
                'step': i,
                'w': w_val,
                'x': x_val,
                'w*x': w_val * x_val,
                'acc_before': acc_before,
                'acc_after': acc_after,
                'acc_quant': acc
            })

        # Add bias
        b_val = float(bias_q[o].item())
        acc_before = acc

        if use_float64_fma:
            acc = fma_float64(b_val, 1.0, acc)
        else:
            acc = fma_float32(b_val, 1.0, acc)

        acc_after = acc
        acc_tensor = torch.tensor(acc)
        acc_tensor = quantize_float(acc_tensor)
        acc = float(acc_tensor.item())

        step_trace.append({
            'step': 'bias',
            'bias': b_val,
            'acc_before': acc_before,
            'acc_after': acc_after,
            'acc_quant': acc
        })

        traces.append({
            'output': o,
            'final_pre_relu': acc,
            'trace': step_trace
        })

    return traces

def main():
    print("=" * 80)
    print("FMA Precision Debug Analysis")
    print("=" * 80)

    # Load data
    inputs, weights, biases = load_validation_data()

    # Focus on Sample 15 which has differences
    sample_idx = 15
    input_vec = inputs[sample_idx]

    # Apply NERF encoding
    nerf_input = torch.cat([input_vec, torch.sin(input_vec), torch.cos(input_vec)])

    print(f"\nSample {sample_idx} Input (first 5 values):")
    print(f"  {input_vec[:5].numpy()}")
    print(f"\nNERF-encoded input shape: {nerf_input.shape}")

    # Trace with float64 FMA (current implementation)
    print("\n" + "=" * 80)
    print("Testing with Float64 FMA (current Python implementation)")
    print("=" * 80)
    traces_f64 = trace_first_layer_fma(nerf_input, weights[0], biases[0], use_float64_fma=True)

    # Trace with float32 FMA (no extended precision)
    print("\n" + "=" * 80)
    print("Testing with Float32 FMA (no extended precision)")
    print("=" * 80)
    traces_f32 = trace_first_layer_fma(nerf_input, weights[0], biases[0], use_float64_fma=False)

    # Compare results
    print("\n" + "=" * 80)
    print("Comparison: Float64 vs Float32 FMA for Layer 0 Pre-activations")
    print("=" * 80)
    print(f"\n{'Output':>6} {'Float64':>15} {'Float32':>15} {'Diff':>15} {'Match':>8}")
    print("-" * 60)

    diff_count = 0
    for i in range(min(10, len(traces_f64))):  # Show first 10 outputs
        val_f64 = traces_f64[i]['final_pre_relu']
        val_f32 = traces_f32[i]['final_pre_relu']
        diff = abs(val_f64 - val_f32)
        match = "Yes" if diff == 0 else "No"
        if diff != 0:
            diff_count += 1
        print(f"{i:>6} {val_f64:>15.6f} {val_f32:>15.6f} {diff:>15.10f} {match:>8}")

    print(f"\nDifferences found in first 10 outputs: {diff_count}")

    # Find all differences
    all_diffs = []
    for i in range(len(traces_f64)):
        val_f64 = traces_f64[i]['final_pre_relu']
        val_f32 = traces_f32[i]['final_pre_relu']
        if val_f64 != val_f32:
            all_diffs.append((i, val_f64, val_f32, abs(val_f64 - val_f32)))

    print(f"\nTotal differences in Layer 0 pre-activations: {len(all_diffs)}/{len(traces_f64)}")

    if all_diffs:
        print("\nFirst 5 differences:")
        for i, (idx, f64, f32, diff) in enumerate(all_diffs[:5]):
            print(f"  Output {idx}: Float64={f64:.10f}, Float32={f32:.10f}, Diff={diff:.10e}")

    # Detailed trace for output 0 (which has a difference in final output)
    print("\n" + "=" * 80)
    print("Detailed FMA Trace for Output 0 (First 20 steps)")
    print("=" * 80)

    print(f"\n{'Step':>5} {'w_val':>14} {'x_val':>14} {'acc_f64':>14} {'acc_f32':>14} {'diff':>12}")
    print("-" * 75)

    for i in range(min(20, len(traces_f64[0]['trace']) - 1)):  # Exclude bias step
        t_f64 = traces_f64[0]['trace'][i]
        t_f32 = traces_f32[0]['trace'][i]

        diff = abs(t_f64['acc_quant'] - t_f32['acc_quant'])
        if diff > 0:
            marker = " *"
        else:
            marker = ""

        print(f"{i:>5} {t_f64['w']:>14.10f} {t_f64['x']:>14.10f} "
              f"{t_f64['acc_quant']:>14.10f} {t_f32['acc_quant']:>14.10f} "
              f"{diff:>12.2e}{marker}")

    # Summary
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)
    print("""
Key findings:

1. Float64 FMA: Uses 64-bit precision for intermediate multiply
   - This is what the current Python implementation uses
   - More precise than pure float32, but not as precise as x86 80-bit

2. Float32 FMA: Uses pure 32-bit operations
   - No extended precision for intermediate values
   - May accumulate more rounding errors

3. C++ hardware fma(): Uses 80-bit extended precision (on x86)
   - Most precise intermediate calculation
   - This is why C++ results may differ from Python float64

The difference between Python (float64) and C++ (fma) comes from:
- 80-bit extended precision has 64 mantissa bits
- float64 has 53 mantissa bits
- This ~11 bit difference in precision can cause different quantization results

To get exact match, options are:
1. Modify C++ to not use fma() - use separate multiply and add
2. Use Python with long double (80-bit) if available
3. Accept that 99.44% match is sufficient for practical purposes
""")

if __name__ == '__main__':
    main()
