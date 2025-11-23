"""
Debug FMA implementation to find the bug causing 32.49cm error for E8M23
"""

import torch
import numpy as np

# Simple test: matmul with FMA vs PyTorch matmul
def quantize_float(tensor, exponent_bits, mantissa_bits):
    """Early return for FP32"""
    if exponent_bits >= 8 and mantissa_bits >= 23:
        return tensor
    # ... rest of quantization code ...
    return tensor  # Placeholder

def fma_matmul_output_stationary(x, weight, bias, exponent_bits, mantissa_bits):
    """FMA matmul with per-operation quantization"""
    batch_size, in_features = x.shape
    out_features, _ = weight.shape

    # Quantize inputs and weights once
    x_q = quantize_float(x, exponent_bits, mantissa_bits)
    weight_q = quantize_float(weight, exponent_bits, mantissa_bits)

    # Output buffer
    output = torch.zeros(batch_size, out_features, device=x.device)

    # Output stationary: iterate over each output element
    for b in range(batch_size):
        for o in range(out_features):
            # Initialize accumulator for this output element
            acc = torch.tensor(0.0, device=x.device)

            # FMA loop: accumulate input_i * weight_{o,i}
            for i in range(in_features):
                # Multiply
                mul_result = x_q[b, i] * weight_q[o, i]

                # Add (FMA)
                acc = acc + mul_result

                # Quantize accumulator after each FMA
                acc = quantize_float(acc, exponent_bits, mantissa_bits)

            # Add bias
            if bias is not None:
                bias_q = quantize_float(bias[o], exponent_bits, mantissa_bits)
                acc = acc + bias_q
                acc = quantize_float(acc, exponent_bits, mantissa_bits)

            output[b, o] = acc

    return output


print("="*80)
print("DEBUG: FMA matmul vs PyTorch matmul (FP32)")
print("="*80)

# Create simple test case
torch.manual_seed(42)
x = torch.randn(2, 3)  # batch=2, in_features=3
weight = torch.randn(4, 3)  # out_features=4, in_features=3
bias = torch.randn(4)

print(f"\nInput x shape: {x.shape}")
print(f"Weight shape: {weight.shape}")
print(f"Bias shape: {bias.shape}")

# PyTorch matmul
pytorch_output = torch.matmul(x, weight.T) + bias
print(f"\nPyTorch output:\n{pytorch_output}")

# FMA matmul (E8M23 - no quantization)
fma_output = fma_matmul_output_stationary(x, weight, bias, 8, 23)
print(f"\nFMA output:\n{fma_output}")

# Compare
diff = torch.abs(pytorch_output - fma_output)
print(f"\nMax difference: {torch.max(diff).item():.10e}")
print(f"Mean difference: {torch.mean(diff).item():.10e}")

print("\nPer-element comparison:")
for b in range(2):
    for o in range(4):
        print(f"  [{b}, {o}]: PyTorch={pytorch_output[b,o]:.10f}, FMA={fma_output[b,o]:.10f}, Diff={diff[b,o]:.10e}")

# Check if they're close
if torch.allclose(pytorch_output, fma_output, atol=1e-5, rtol=1e-5):
    print("\n✓ FMA implementation matches PyTorch!")
else:
    print("\n✗ BUG: FMA implementation does NOT match PyTorch!")

print("\n" + "="*80)
