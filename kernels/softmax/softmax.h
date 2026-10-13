#pragma once

// y[r, :] = softmax(x[r, :]) over the last dim of a row-major rows×cols matrix, FP32.
__global__ void softmax_v0_kernel(const float* x, float* y, int rows, int cols);

const int kSoftmaxV1Threads = 256;  // 8 warps = 8 rows per block
// float4 loads: needs cols % 4 == 0 and 16-byte-aligned x and y (rows_vectorizable).
__global__ void softmax_v1_kernel(const float* x, float* y, int rows, int cols);
// Same kernel with scalar loads, for every other shape.
__global__ void softmax_v1_scalar_kernel(const float* x, float* y, int rows, int cols);

#include <cuda_runtime.h>

const int kSoftmaxV2Threads = 256;  // 8 warps = 8 rows per block
// Launches softmax_v2_kernel<k> (defined and launched in v2_online.cu): k float4s per lane cached
// in registers (rows of up to 128·k floats, one DRAM read) for k in {1, 2, 4, 8, 16};
// 0 = float4 online two-pass; -1 = scalar online two-pass.
cudaError_t softmax_v2_launch(int k, const float* x, float* y, int rows, int cols, cudaStream_t s);

// v3: one block of `threads` per row, the row in `smem` bytes of dynamic shared memory (>= cols·4).
// vec = float4 loads (rows_vectorizable). The launcher (src/api.cu) picks threads and smem and
// falls back to v2 streaming for rows that don't fit.
cudaError_t softmax_v3_launch(bool vec, int threads, size_t smem, const float* x, float* y,
                              int rows, int cols, cudaStream_t s);
