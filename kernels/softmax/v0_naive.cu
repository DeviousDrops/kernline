// Softmax v0: naive baseline. One thread per row, three passes over that row: max, sum of
// exp(x - max), then write exp(x - max) / sum. Neighbouring threads in a warp read addresses
// `cols` floats apart, so every load is its own 32-byte sector and nothing coalesces. With few
// rows there are also too few threads to fill the GPU. v1 gives each row a warp.
#include "kernels/softmax/softmax.h"

__global__ void softmax_v0_kernel(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* in = x + (size_t)row * cols;
    float* out = y + (size_t)row * cols;

    float mx = -INFINITY;
    for (int j = 0; j < cols; ++j) mx = fmaxf(mx, in[j]);
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) sum += expf(in[j] - mx);
    const float inv = 1.0f / sum;
    for (int j = 0; j < cols; ++j) out[j] = expf(in[j] - mx) * inv;
}
