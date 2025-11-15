#include "Constraints/EnvCollision/cuda_nn_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

// Custom precision FMA device function
// Performs FMA with configurable precision (exponent and mantissa bits)
__device__ float custom_precision_fma(float a, float b, float c, int exp_bits, int mant_bits) {
    // Perform FMA in full precision first
    float result = fmaf(a, b, c);

    // Apply custom precision rounding
    if (exp_bits == 8 && mant_bits == 23) {
        // Standard float32, no rounding needed
        return result;
    }

    // Extract float representation
    unsigned int bits = __float_as_uint(result);
    unsigned int sign = bits & 0x80000000u;
    int exponent = ((bits >> 23) & 0xFFu) - 127;
    unsigned int mantissa = bits & 0x007FFFFFu;

    // Handle special cases (zero, infinity, NaN)
    if ((bits & 0x7FFFFFFFu) == 0) return result; // Zero
    if (exponent == 128) return result; // Infinity or NaN

    // Check exponent range for custom precision
    int max_exp = (1 << (exp_bits - 1)) - 1;
    int min_exp = -(1 << (exp_bits - 1)) + 2;

    // Overflow to infinity
    if (exponent > max_exp) {
        return sign ? -INFINITY : INFINITY;
    }

    // Underflow to zero
    if (exponent < min_exp) {
        return __uint_as_float(sign);
    }

    // Round mantissa to target precision
    int bits_to_clear = 23 - mant_bits;
    if (bits_to_clear > 0) {
        unsigned int mask = (~0u) << bits_to_clear;
        unsigned int round_bit = 1u << (bits_to_clear - 1);

        // Round to nearest, ties to even
        if ((mantissa & ~mask) > round_bit ||
            ((mantissa & ~mask) == round_bit && (mantissa & (1u << bits_to_clear)))) {
            mantissa = (mantissa & mask) + (1u << bits_to_clear);

            // Check for mantissa overflow
            if (mantissa >= 0x00800000u) {
                mantissa = 0;
                exponent++;

                // Check for exponent overflow after mantissa overflow
                if (exponent > max_exp) {
                    return sign ? -INFINITY : INFINITY;
                }
            }
        } else {
            mantissa = mantissa & mask;
        }
    }

    // Reconstruct float
    unsigned int result_bits = sign | (((unsigned int)(exponent + 127) & 0xFFu) << 23) | (mantissa & 0x007FFFFFu);
    return __uint_as_float(result_bits);
}

// CUDA kernel for matrix-vector multiplication with custom precision FMA
// Computes output[i,j] = sum_k(weight[i,k] * input[k,j]) + bias[i]
// Each FMA operation is performed with custom precision and rounded immediately
__global__ void matmul_fma_kernel(
    const float* weight,     // [out_dim x in_dim]
    const float* input,      // [in_dim x batch]
    const float* bias,       // [out_dim]
    float* output,           // [out_dim x batch]
    int out_dim,
    int in_dim,
    int batch_size,
    int exp_bits,
    int mant_bits
) {
    // Each thread computes one output element output[i,j]
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // batch column

    if (i >= out_dim || j >= batch_size) return;

    // Initialize accumulator with bias (with custom precision)
    float acc = custom_precision_fma(0.0f, 0.0f, bias[i], exp_bits, mant_bits);

    // Accumulate weight[i,k] * input[k,j] for all k
    // Each multiply-add uses FMA and is rounded to custom precision
    for (int k = 0; k < in_dim; k++) {
        float w = weight[i * in_dim + k];
        float x = input[k * batch_size + j];
        // Perform FMA: acc = w * x + acc, with custom precision rounding
        acc = custom_precision_fma(w, x, acc, exp_bits, mant_bits);
    }

    // Store result
    output[i * batch_size + j] = acc;
}

// CUDA kernel for ReLU activation with custom precision
__global__ void relu_kernel(
    const float* input,
    float* output,
    int size,
    int exp_bits,
    int mant_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = input[idx];
    float result = (val > 0.0f) ? val : 0.0f;

    // Apply custom precision to result
    output[idx] = custom_precision_fma(result, 1.0f, 0.0f, exp_bits, mant_bits);
}

// CUDA kernel for converting bfloat16 to float32
__global__ void bfloat16_to_float32_kernel(
    const unsigned short* bfloat16_data,
    float* float32_data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // bfloat16 to float32: just shift left by 16 bits
    unsigned int bits = ((unsigned int)bfloat16_data[idx]) << 16;
    float32_data[idx] = __uint_as_float(bits);
}

// CUDA kernel for converting float32 to bfloat16
__global__ void float32_to_bfloat16_kernel(
    const float* float32_data,
    unsigned short* bfloat16_data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // float32 to bfloat16: round to nearest even and take upper 16 bits
    unsigned int bits = __float_as_uint(float32_data[idx]);
    unsigned int rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    bits += rounding_bias;
    bfloat16_data[idx] = (unsigned short)(bits >> 16);
}

// Host functions to launch kernels

extern "C" void launch_matmul_fma(
    const float* d_weight,
    const float* d_input,
    const float* d_bias,
    float* d_output,
    int out_dim,
    int in_dim,
    int batch_size,
    int exp_bits,
    int mant_bits,
    cudaStream_t stream
) {
    // Configure kernel launch parameters
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim(
        (batch_size + blockDim.x - 1) / blockDim.x,
        (out_dim + blockDim.y - 1) / blockDim.y
    );

    matmul_fma_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_weight, d_input, d_bias, d_output,
        out_dim, in_dim, batch_size,
        exp_bits, mant_bits
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_relu(
    const float* d_input,
    float* d_output,
    int size,
    int exp_bits,
    int mant_bits,
    cudaStream_t stream
) {
    int blockDim = 256;
    int gridDim = (size + blockDim - 1) / blockDim;

    relu_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_input, d_output, size, exp_bits, mant_bits
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_bfloat16_to_float32(
    const unsigned short* d_bfloat16,
    float* d_float32,
    int size,
    cudaStream_t stream
) {
    int blockDim = 256;
    int gridDim = (size + blockDim - 1) / blockDim;

    bfloat16_to_float32_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_bfloat16, d_float32, size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_float32_to_bfloat16(
    const float* d_float32,
    unsigned short* d_bfloat16,
    int size,
    cudaStream_t stream
) {
    int blockDim = 256;
    int gridDim = (size + blockDim - 1) / blockDim;

    float32_to_bfloat16_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_float32, d_bfloat16, size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
