"""
Unit test for quantize_float function
"""

import torch
import numpy as np

def quantize_float(tensor, exponent_bits, mantissa_bits):
    """
    Quantize tensor to specified exponent and mantissa bits
    IEEE 754 bit manipulation

    FIXED: Properly handle scalar tensors and avoid unnecessary conversions for FP32
    """
    # Early return for FP32 (no quantization needed)
    if exponent_bits >= 8 and mantissa_bits >= 23:
        return tensor

    # Convert to proper dtype first
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    # Handle scalar tensors by adding dimension
    original_shape = tensor.shape
    is_scalar = (tensor.ndim == 0)

    if is_scalar:
        tensor = tensor.unsqueeze(0)

    # Flatten for processing
    tensor_flat = tensor.flatten()

    # Convert to numpy for bit manipulation
    # Use .astype to ensure proper dtype
    tensor_np = tensor_flat.detach().cpu().numpy().astype(np.float32)

    # View as uint32 for bit manipulation
    bits = tensor_np.view(np.uint32)

    # Quantize mantissa
    mantissa_bits_to_zero = 23 - mantissa_bits
    mantissa_mask = np.uint32(0x007FFFFF >> mantissa_bits_to_zero)
    mantissa_mask = mantissa_mask << mantissa_bits_to_zero

    # Quantize exponent
    exponent = (bits >> 23) & 0xFF
    exponent_bits_to_zero = 8 - exponent_bits
    exponent_quantized = (exponent >> exponent_bits_to_zero) << exponent_bits_to_zero

    # Reconstruct quantized bits
    sign_bit = bits & 0x80000000
    exponent_field = (exponent_quantized & 0xFF) << 23
    mantissa_field = bits & mantissa_mask
    quantized_bits = sign_bit | exponent_field | mantissa_field

    # Convert back to float32
    quantized_np = quantized_bits.view(np.float32)

    # Convert to torch tensor
    quantized_tensor = torch.from_numpy(quantized_np).to(tensor.device)

    # Restore original shape
    if is_scalar:
        quantized_tensor = quantized_tensor.squeeze(0)
    else:
        quantized_tensor = quantized_tensor.reshape(original_shape)

    return quantized_tensor


# Test 1: E8M23 should return original tensor (no quantization)
print("="*80)
print("Test 1: E8M23 (FP32) should not quantize")
print("="*80)

x = torch.tensor([1.23456789, 2.34567890, 3.45678901])
x_quant = quantize_float(x, 8, 23)
print(f"Original: {x}")
print(f"Quantized: {x_quant}")
print(f"Are they identical? {torch.allclose(x, x_quant, atol=0, rtol=0)}")
print(f"Max difference: {torch.max(torch.abs(x - x_quant))}")

# Test 2: Scalar tensor handling
print("\n" + "="*80)
print("Test 2: Scalar tensor handling")
print("="*80)

scalar = torch.tensor(3.14159)
scalar_quant = quantize_float(scalar, 8, 7)  # BF16
print(f"Original scalar: {scalar} (shape: {scalar.shape})")
print(f"Quantized scalar: {scalar_quant} (shape: {scalar_quant.shape})")
print(f"Is scalar? {scalar_quant.ndim == 0}")

# Test 3: BF16 quantization
print("\n" + "="*80)
print("Test 3: BF16 (E8M7) quantization")
print("="*80)

x = torch.tensor([1.23456789])
x_quant = quantize_float(x, 8, 7)
print(f"Original: {x} ({x.item():.10f})")
print(f"Quantized: {x_quant} ({x_quant.item():.10f})")

# Manual BF16 conversion for verification
def manual_bf16(val):
    bits = np.array([val], dtype=np.float32).view(np.uint32)[0]
    # Zero out lower 16 bits of mantissa (keep only 7 bits)
    bits = bits & 0xFFFF0000
    return np.frombuffer(bits.tobytes(), dtype=np.float32)[0]

manual_result = manual_bf16(1.23456789)
print(f"Manual BF16: {manual_result:.10f}")
print(f"Match? {np.abs(x_quant.item() - manual_result) < 1e-10}")

# Test 4: Accumulation test
print("\n" + "="*80)
print("Test 4: FMA accumulation with E8M23")
print("="*80)

a = torch.tensor(0.1)
b = torch.tensor(0.2)
acc = torch.tensor(0.0)

# No quantization (E8M23)
for i in range(10):
    prod = a * b
    acc = acc + prod
    acc_quant = quantize_float(acc, 8, 23)
    if i < 3:
        print(f"Step {i}: acc={acc.item():.10f}, acc_quant={acc_quant.item():.10f}, diff={abs(acc.item() - acc_quant.item()):.2e}")

print(f"\nFinal accumulator: {acc.item():.10f}")
print(f"Expected (10 * 0.1 * 0.2): {10 * 0.1 * 0.2:.10f}")
print(f"Difference: {abs(acc.item() - 10*0.1*0.2):.2e}")

# Test 5: Accumulation test with BF16
print("\n" + "="*80)
print("Test 5: FMA accumulation with E8M7 (BF16)")
print("="*80)

a = torch.tensor(0.1)
b = torch.tensor(0.2)
acc = torch.tensor(0.0)

for i in range(10):
    prod = a * b
    acc = acc + prod
    acc = quantize_float(acc, 8, 7)  # Quantize after each FMA
    if i < 3:
        print(f"Step {i}: acc={acc.item():.10f}")

print(f"\nFinal accumulator (BF16): {acc.item():.10f}")
print(f"Expected (no quantization): {10 * 0.1 * 0.2:.10f}")
print(f"Error due to quantization: {abs(acc.item() - 10*0.1*0.2):.2e}")

print("\n" + "="*80)
print("✓ All tests completed")
print("="*80)
