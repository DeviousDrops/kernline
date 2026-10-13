// LayerNorm v0: naive baseline. One thread per row, three passes over that row: mean, then
// variance as the mean of (x - mean)^2, then normalize. The two-pass variance avoids the
// E[x^2] - E[x]^2 cancellation, but the mean itself is a sequential FP32 sum, which drifts by
// ~0.01 on rows with a large offset (tests/test_rowwise.py::test_layernorm_survives_large_mean).
// Same access problem as softmax v0: a warp's 32 loads are `cols` floats apart. gamma/beta are
// read at the same index by every thread, so those loads at least broadcast.
#include "kernels/layernorm/layernorm.h"

__global__ void layernorm_v0_kernel(const float* x, const float* gamma, const float* beta,
                                    float* y, int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* in = x + (size_t)row * cols;
    float* out = y + (size_t)row * cols;

    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) sum += in[j];
    const float mean = sum / cols;
    float sq = 0.0f;
    for (int j = 0; j < cols; ++j) {
        const float d = in[j] - mean;
        sq = fmaf(d, d, sq);
    }
    const float rstd = rsqrtf(sq / cols + eps);
    for (int j = 0; j < cols; ++j) out[j] = fmaf((in[j] - mean) * rstd, gamma[j], beta[j]);
}
