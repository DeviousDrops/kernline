// Elementwise and layout kernels for the transformer block. Consecutive threads touch consecutive
// output elements, and every input index is contiguous along the same axis, so loads and stores
// coalesce on both sides of the head split/merge.
#include "kernels/block/block_ops.h"

namespace {
__device__ __forceinline__ long long tid() { return blockIdx.x * (long long)blockDim.x + threadIdx.x; }
__device__ __forceinline__ long long stride() { return (long long)gridDim.x * blockDim.x; }
}  // namespace

__global__ void add_inplace_kernel(float* y, const float* x, long long n) {
    for (long long i = tid(); i < n; i += stride()) y[i] += x[i];
}

__global__ void gelu_inplace_kernel(float* x, long long n) {
    const float c = 0.7978845608f;  // sqrt(2/pi)
    for (long long i = tid(); i < n; i += stride()) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(c * fmaf(0.044715f * v, v * v, v)));
    }
}

// One block row per token (blockIdx.y), threads across its columns: 32-bit index math with two
// divisions per element. A flat 64-bit index needs five 64-bit divisions per element, which made
// this kernel instruction-bound at about a sixth of L2 bandwidth.
__global__ void split_heads_kernel(const float* qkv, float* q, float* k, float* v, int B, int S,
                                   int H, int d) {
    const int D = H * d, t = blockIdx.y, b = t / S, s = t - b * S;
    const float* row = qkv + (size_t)t * 3 * D;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < 3 * D; c += gridDim.x * blockDim.x) {
        const int which = c / D, hc = c - which * D, h = hc / d, e = hc - h * d;
        float* out = which == 0 ? q : which == 1 ? k : v;
        out[(((size_t)b * H + h) * S + s) * d + e] = row[c];
    }
}

__global__ void merge_heads_kernel(const float* o, float* y, int B, int S, int H, int d) {
    const int D = H * d, t = blockIdx.y, b = t / S, s = t - b * S;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < D; c += gridDim.x * blockDim.x) {
        const int h = c / d, e = c - h * d;
        y[(size_t)t * D + c] = o[(((size_t)b * H + h) * S + s) * d + e];
    }
}
