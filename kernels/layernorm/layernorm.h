#pragma once

// y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * gamma + beta over the last dim of a row-major
// rows×cols matrix, FP32. var is the biased (population) variance, as in torch.nn.LayerNorm.
__global__ void layernorm_v0_kernel(const float* x, const float* gamma, const float* beta,
                                    float* y, int rows, int cols, float eps);
