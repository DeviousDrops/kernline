#pragma once

// y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * gamma + beta over the last dim of a row-major
// rows×cols matrix, FP32. var is the biased (population) variance, as in torch.nn.LayerNorm.
__global__ void layernorm_v0_kernel(const float* x, const float* gamma, const float* beta,
                                    float* y, int rows, int cols, float eps);

const int kLayerNormV1Threads = 256;  // 8 warps = 8 rows per block
// float4 loads: needs cols % 4 == 0 and 16-byte-aligned x, gamma, beta, y (rows_vectorizable).
__global__ void layernorm_v1_kernel(const float* x, const float* gamma, const float* beta,
                                    float* y, int rows, int cols, float eps);
// Same kernel with scalar loads, for every other shape.
__global__ void layernorm_v1_scalar_kernel(const float* x, const float* gamma, const float* beta,
                                           float* y, int rows, int cols, float eps);
