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

__global__ void split_heads_kernel(const float* qkv, float* q, float* k, float* v, int B, int S,
                                   int H, int d) {
    const long long D = (long long)H * d, n = (long long)B * S * D;
    float* out[3] = {q, k, v};
    for (long long i = tid(); i < 3 * n; i += stride()) {
        const int which = (int)(i / n);
        const long long j = i % n;  // index into [B, H, S, d]
        const int e = (int)(j % d), s = (int)(j / d % S), h = (int)(j / ((long long)d * S) % H);
        const long long b = j / (D * S);
        out[which][j] = qkv[(b * S + s) * 3 * D + which * D + (long long)h * d + e];
    }
}

__global__ void merge_heads_kernel(const float* o, float* y, int B, int S, int H, int d) {
    const long long D = (long long)H * d, n = (long long)B * S * D;
    for (long long i = tid(); i < n; i += stride()) {  // i indexes y: [B, S, H, d]
        const int e = (int)(i % d), h = (int)(i / d % H), s = (int)(i / D % S);
        const long long b = i / (D * S);
        y[i] = o[((b * H + h) * S + s) * d + e];
    }
}
