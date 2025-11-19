#!/usr/bin/env python3
"""
Compare different FMA implementation approaches for Sample 15.

The goal is to understand why Python and C++ diverge for this specific sample.
"""

import numpy as np
import torch
import os
import struct

def float_to_hex(f):
    """Convert float to hex representation."""
    return hex(struct.unpack('>I', struct.pack('>f', f))[0])

def quantize_float_truncate(x):
    """Quantize to E8M7 using truncation (current implementation)."""
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = np.array(x, dtype=np.float32)

    scalar_input = x_np.ndim == 0
    if scalar_input:
        x_np = x_np.reshape(1)

    x_bytes = x_np.view(np.uint32)
    mask = 0xFFFF0000
    x_truncated = (x_bytes & mask).view(np.float32)

    if scalar_input:
        x_truncated = x_truncated[0]

    return x_truncated

def fma_f64(a, b, c):
    """FMA using float64."""
    return float(np.float64(a) * np.float64(b) + np.float64(c))

def fma_f80(a, b, c):
    """FMA using longdouble (80-bit on x86)."""
    return float(np.longdouble(a) * np.longdouble(b) + np.longdouble(c))

def load_data():
    """Load inputs and weights."""
    data = np.load('python_validation_reference.npz')
    inputs = data['input']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_dir = os.path.join(script_dir, '..', '..', '..', 'cpp', 'NNmodel', 'env', 'parameter')

    # Load first layer weights
    w0_data = np.loadtxt(os.path.join(weight_dir, 'weight_0.txt'))
    w0 = w0_data.reshape(256, 30).astype(np.float32)
    b0 = np.loadtxt(os.path.join(weight_dir, 'bias_0.txt')).astype(np.float32)

    return inputs, w0, b0

def trace_neuron(x_nerf, weight_row, bias_val, neuron_idx, use_f80=False):
    """
    Trace FMA computation for a single output neuron.
    Returns detailed trace of each step.
    """
    x_q = quantize_float_truncate(x_nerf)
    w_q = quantize_float_truncate(weight_row)
    b_q = float(quantize_float_truncate(np.array(bias_val, dtype=np.float32)))

    fma_func = fma_f80 if use_f80 else fma_f64

    acc = 0.0
    trace = []

    for i in range(len(x_q)):
        w_val = float(w_q[i])
        x_val = float(x_q[i])

        acc_before = acc
        acc = fma_func(w_val, x_val, acc)
        acc_fma = acc
        acc = float(quantize_float_truncate(np.array(acc, dtype=np.float32)))

        trace.append({
            'step': i,
            'w': w_val,
            'x': x_val,
            'acc_before': acc_before,
            'acc_fma': acc_fma,
            'acc_quant': acc,
            'quant_diff': acc_fma - acc
        })

    # Add bias
    acc_before = acc
    acc = fma_func(b_q, 1.0, acc)
    acc_fma = acc
    acc = float(quantize_float_truncate(np.array(acc, dtype=np.float32)))

    trace.append({
        'step': 'bias',
        'bias': b_q,
        'acc_before': acc_before,
        'acc_fma': acc_fma,
        'acc_quant': acc,
        'quant_diff': acc_fma - acc
    })

    return acc, trace

def main():
    print("="*70)
    print("Comparing FMA Implementations for Sample 15, Layer 0")
    print("="*70)

    inputs, w0, b0 = load_data()

    x = inputs[15]
    x_nerf = np.concatenate([x, np.sin(x), np.cos(x)])

    # Compare output 0 which has a 0.125 difference
    neuron_idx = 0
    weight_row = w0[neuron_idx]
    bias_val = b0[neuron_idx]

    # Run with float64 FMA
    result_f64, trace_f64 = trace_neuron(x_nerf, weight_row, bias_val, neuron_idx, use_f80=False)
    # Run with longdouble FMA
    result_f80, trace_f80 = trace_neuron(x_nerf, weight_row, bias_val, neuron_idx, use_f80=True)

    print(f"\nNeuron 0, Final pre-activation:")
    print(f"  Float64 FMA:    {result_f64:.10f}")
    print(f"  LongDouble FMA: {result_f80:.10f}")
    print(f"  Difference:     {abs(result_f64 - result_f80):.2e}")

    # Find steps where f64 and f80 differ
    diffs = []
    for i in range(len(trace_f64)):
        t64 = trace_f64[i]
        t80 = trace_f80[i]

        if t64['acc_quant'] != t80['acc_quant']:
            step = t64.get('step', 'bias')
            diffs.append((step, t64['acc_quant'], t80['acc_quant']))

    if diffs:
        print(f"\nSteps where f64 and f80 differ after quantization: {len(diffs)}")
        for step, v64, v80 in diffs[:5]:
            print(f"  Step {step}: f64={v64:.10f}, f80={v80:.10f}, diff={abs(v64-v80):.2e}")
    else:
        print(f"\nNo differences between f64 and f80 after quantization!")

    # Now let's see what the raw FMA results look like before quantization
    print("\n" + "="*70)
    print("Checking pre-quantization FMA values at selected steps")
    print("="*70)

    # Find steps with largest quantization loss
    quant_losses = [(i, trace_f64[i]['quant_diff']) for i in range(len(trace_f64))]
    quant_losses.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nLargest quantization losses:")
    for i, loss in quant_losses[:10]:
        t = trace_f64[i]
        step = t.get('step', i)
        print(f"  Step {step}: acc_fma={t['acc_fma']:.10f}, acc_quant={t['acc_quant']:.10f}, loss={loss:.6f}")

    # Check all neurons for differences between f64 and f80
    print("\n" + "="*70)
    print("Comparing all Layer 0 neurons")
    print("="*70)

    neuron_diffs = []
    for n in range(256):
        result_f64, _ = trace_neuron(x_nerf, w0[n], b0[n], n, use_f80=False)
        result_f80, _ = trace_neuron(x_nerf, w0[n], b0[n], n, use_f80=True)

        if result_f64 != result_f80:
            neuron_diffs.append((n, result_f64, result_f80))

    if neuron_diffs:
        print(f"\nNeurons with different results ({len(neuron_diffs)}):")
        for n, v64, v80 in neuron_diffs[:10]:
            print(f"  Neuron {n}: f64={v64:.6f}, f80={v80:.6f}")
    else:
        print("\nAll 256 neurons give same result with f64 and f80!")

    # The real question is: why does Python differ from C++?
    # Let's check if the issue might be in something other than FMA precision
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"""
Since f64 and f80 give identical results after quantization,
the difference must come from somewhere else:

1. Different order of operations?
2. Different quantization timing (C++ might quantize at different points)?
3. Different handling of special values?
4. Compiler optimizations in C++ changing computation order?

Next step: Add debug output to C++ to trace exact FMA values
and compare with Python trace.
""")

if __name__ == '__main__':
    main()
