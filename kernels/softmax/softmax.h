#pragma once

// y[r, :] = softmax(x[r, :]) over the last dim of a row-major rows×cols matrix, FP32.
__global__ void softmax_v0_kernel(const float* x, float* y, int rows, int cols);
