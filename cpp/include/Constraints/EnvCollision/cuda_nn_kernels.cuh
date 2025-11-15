#ifndef CUDA_NN_KERNELS_CUH
#define CUDA_NN_KERNELS_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch matrix multiplication with custom precision FMA
// output[out_dim x batch] = weight[out_dim x in_dim] * input[in_dim x batch] + bias[out_dim]
// Each element is computed using element-wise FMA with custom precision rounding
void launch_matmul_fma(
    const float* d_weight,
    const float* d_input,
    const float* d_bias,
    float* d_output,
    int out_dim,
    int in_dim,
    int batch_size,
    int exp_bits,
    int mant_bits,
    cudaStream_t stream = 0
);

// Launch ReLU activation with custom precision
void launch_relu(
    const float* d_input,
    float* d_output,
    int size,
    int exp_bits,
    int mant_bits,
    cudaStream_t stream = 0
);

// Launch bfloat16 to float32 conversion
void launch_bfloat16_to_float32(
    const unsigned short* d_bfloat16,
    float* d_float32,
    int size,
    cudaStream_t stream = 0
);

// Launch float32 to bfloat16 conversion
void launch_float32_to_bfloat16(
    const float* d_float32,
    unsigned short* d_bfloat16,
    int size,
    cudaStream_t stream = 0
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_NN_KERNELS_CUH
