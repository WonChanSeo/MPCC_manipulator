#!/usr/bin/env python3
"""
Test to demonstrate the difference between truncation (Python) and round-to-nearest (FlexFloat)
"""

import torch
import numpy as np


def quantize_float_truncate(tensor, exponent_bits, mantissa_bits):
    """Python's bit manipulation - performs TRUNCATION"""
    if mantissa_bits == 23:
        return tensor

    device = tensor.device
    tensor = tensor.cpu()

    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    flat_tensor = tensor.view(-1)
    bits = flat_tensor.numpy().view(np.uint32)

    # Quantize mantissa (TRUNCATION)
    mantissa_bits_to_zero = 23 - mantissa_bits
    mantissa_mask = np.uint32(0x007FFFFF >> mantissa_bits_to_zero)
    mantissa_mask = mantissa_mask << mantissa_bits_to_zero

    # Quantize exponent (TRUNCATION)
    exponent = (bits >> 23) & 0xFF
    exponent_bits_to_zero = 8 - exponent_bits
    exponent_quantized = (exponent >> exponent_bits_to_zero) << exponent_bits_to_zero

    # Extract sign and reconstruct
    sign_bit = bits & 0x80000000
    exponent_field = (exponent_quantized & 0xFF) << 23
    quantized_bits = sign_bit | exponent_field | (bits & mantissa_mask)

    quantized_flat = torch.from_numpy(quantized_bits.view(np.float32))
    result = quantized_flat.view(tensor.shape)

    if squeeze_output:
        result = result.squeeze(0)

    return result.to(device)


def quantize_float_round_to_nearest(tensor, exponent_bits, mantissa_bits):
    """Simulate FlexFloat's ROUND-TO-NEAREST-EVEN behavior"""
    if mantissa_bits == 23:
        return tensor

    device = tensor.device
    tensor = tensor.cpu()

    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    flat_tensor = tensor.view(-1)
    bits = flat_tensor.numpy().view(np.uint32)

    # Extract components
    sign_bit = bits & 0x80000000
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x007FFFFF

    # Quantize exponent (truncation)
    exponent_bits_to_zero = 8 - exponent_bits
    exponent_quantized = (exponent >> exponent_bits_to_zero) << exponent_bits_to_zero

    # Quantize mantissa (ROUND-TO-NEAREST-EVEN)
    mantissa_bits_to_zero = 23 - mantissa_bits

    # Get the bits that will be discarded
    discard_mask = np.uint32((1 << mantissa_bits_to_zero) - 1)
    discarded_bits = mantissa & discard_mask

    # Round bit is the MSB of discarded bits
    round_bit_pos = mantissa_bits_to_zero - 1
    round_bit = (discarded_bits >> round_bit_pos) & 1

    # Sticky bit is OR of all bits below round bit
    sticky_mask = np.uint32((1 << round_bit_pos) - 1) if round_bit_pos > 0 else 0
    sticky_bit = 1 if (discarded_bits & sticky_mask) != 0 else 0

    # Truncate mantissa first
    mantissa_truncated = (mantissa >> mantissa_bits_to_zero) << mantissa_bits_to_zero

    # Apply rounding: round to nearest, ties to even
    should_round_up = np.zeros(bits.shape, dtype=bool)

    # If round_bit is 1 and (sticky_bit is 1 OR LSB of truncated mantissa is 1)
    lsb_of_truncated = (mantissa_truncated >> mantissa_bits_to_zero) & 1
    should_round_up = (round_bit == 1) & ((sticky_bit == 1) | (lsb_of_truncated == 1))

    # Add 1 to the truncated mantissa where needed
    increment = np.uint32(1 << mantissa_bits_to_zero)
    mantissa_rounded = np.where(should_round_up, mantissa_truncated + increment, mantissa_truncated)

    # Handle mantissa overflow (carry into exponent)
    mantissa_overflow = (mantissa_rounded & 0x00800000) != 0
    # This is already normalized, but if all mantissa bits are set and we round up,
    # we might overflow into exponent
    mantissa_rounded = mantissa_rounded & 0x007FFFFF

    # Reconstruct
    quantized_bits = sign_bit | (exponent_quantized << 23) | mantissa_rounded

    quantized_flat = torch.from_numpy(quantized_bits.view(np.float32))
    result = quantized_flat.view(tensor.shape)

    if squeeze_output:
        result = result.squeeze(0)

    return result.to(device)


def test_specific_value():
    """Test the specific value that showed difference: 2.2150976658"""
    value = 2.2150976658
    tensor = torch.tensor(value, dtype=torch.float32)

    print("="*70)
    print("QUANTIZATION MODE COMPARISON")
    print("="*70)
    print(f"\nOriginal value: {value:.10f}")
    print(f"As FP32: {tensor.item():.10f}")

    # Show bit representation
    bits = np.array([tensor.item()]).view(np.uint32)[0]
    print(f"Bit representation: 0x{bits:08X}")
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x007FFFFF
    print(f"  Sign: {sign}, Exp: {exp} (0x{exp:02X}), Mantissa: 0x{mant:06X}")

    # Quantize with both methods
    truncated = quantize_float_truncate(tensor, exponent_bits=8, mantissa_bits=7)
    rounded = quantize_float_round_to_nearest(tensor, exponent_bits=8, mantissa_bits=7)

    print(f"\n{'Method':<30} {'Result':<15} {'Difference':<15}")
    print("-"*70)
    print(f"{'Python (Truncation)':<30} {truncated.item():.10f}  {truncated.item() - value:.10f}")
    print(f"{'FlexFloat (Round-to-nearest)':<30} {rounded.item():.10f}  {rounded.item() - value:.10f}")

    print(f"\nDifference between methods: {rounded.item() - truncated.item():.10f}")
    print(f"Ratio: {rounded.item() / truncated.item():.10f}")

    # Show bit representations
    bits_truncated = np.array([truncated.item()]).view(np.uint32)[0]
    bits_rounded = np.array([rounded.item()]).view(np.uint32)[0]

    print(f"\nTruncated bits: 0x{bits_truncated:08X}")
    print(f"Rounded bits:   0x{bits_rounded:08X}")

    # Expected C++ value from debug output
    cpp_value = 2.21875
    print(f"\n{'Actual C++ output:':<30} {cpp_value:.10f}")
    print(f"{'Matches rounded?':<30} {abs(cpp_value - rounded.item()) < 1e-6}")

    return truncated, rounded


def test_more_values():
    """Test a few more values to see the pattern"""
    test_values = [
        2.2150976658,  # The problematic value
        1.0,
        1.5,
        1.25,
        1.125,
        0.1,
        -0.5493164062,  # A weight value
    ]

    print("\n" + "="*70)
    print("MULTIPLE VALUE TEST")
    print("="*70)
    print(f"\n{'Value':<20} {'Truncated':<15} {'Rounded':<15} {'Difference':<15}")
    print("-"*70)

    for value in test_values:
        tensor = torch.tensor(value, dtype=torch.float32)
        truncated = quantize_float_truncate(tensor, exponent_bits=8, mantissa_bits=7)
        rounded = quantize_float_round_to_nearest(tensor, exponent_bits=8, mantissa_bits=7)
        diff = rounded.item() - truncated.item()

        print(f"{value:<20.10f} {truncated.item():<15.10f} {rounded.item():<15.10f} {diff:<15.10f}")


if __name__ == "__main__":
    test_specific_value()
    test_more_values()
