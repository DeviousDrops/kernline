#pragma once

// y[r, :] = softmax(x[r, :]) over the last dim of a row-major rows×cols matrix, FP32.
__global__ void softmax_v0_kernel(const float* x, float* y, int rows, int cols);

const int kSoftmaxV1Threads = 256;  // 8 warps = 8 rows per block
// float4 loads: needs cols % 4 == 0 and 16-byte-aligned x and y (rows_vectorizable).
__global__ void softmax_v1_kernel(const float* x, float* y, int rows, int cols);
// Same kernel with scalar loads, for every other shape.
__global__ void softmax_v1_scalar_kernel(const float* x, float* y, int rows, int cols);
