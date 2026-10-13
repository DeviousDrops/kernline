// LayerNorm v2: one warp per row, each element read from DRAM as few times as possible.
//
//  - Cached (kCache > 0, cols <= 128·kCache): the lane loads its kCache float4s into registers
//    once; mean, then variance as the mean of (x − mean)², then the output all come from
//    registers. One read and one write per element, and the exact two-pass variance for free.
//  - Streamed (kCache == 0, longer rows): Welford. One read pass keeps (n, mean, M2) per lane and
//    lanes merge with Chan's formula; a second pass normalizes. Two reads instead of v1's three,
//    without the E[x²] − E[x]² cancellation a naive one-pass variance would have.
//
// Chan merge of (n_a, mean_a, M2_a) and (n_b, mean_b, M2_b), with d = mean_b − mean_a:
//   n = n_a + n_b,  mean = mean_a + d·n_b/n,  M2 = M2_a + M2_b + d²·n_a·n_b/n.
#include "common/warp_reduce.cuh"
#include "kernels/layernorm/layernorm.h"

namespace {

struct Welford {
    float n, mean, m2;

    __device__ __forceinline__ void push(float x) {
        n += 1.0f;
        const float d = x - mean;
        mean += d / n;
        m2 = fmaf(d, x - mean, m2);
    }
};

__device__ __forceinline__ Welford merge(Welford a, Welford b) {
    const float n = a.n + b.n;
    if (n == 0.0f) return a;
    const float d = b.mean - a.mean, wb = b.n / n;
    return {n, fmaf(d, wb, a.mean), a.m2 + b.m2 + d * d * a.n * wb};
}

__device__ __forceinline__ Welford warp_merge(Welford w) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        Welford o{__shfl_xor_sync(0xffffffffu, w.n, offset),
                  __shfl_xor_sync(0xffffffffu, w.mean, offset),
                  __shfl_xor_sync(0xffffffffu, w.m2, offset)};
        w = merge(w, o);
    }
    return w;
}

__device__ __forceinline__ float4 affine(float4 v, float mean, float rstd, float4 g, float4 b) {
    return make_float4(fmaf((v.x - mean) * rstd, g.x, b.x), fmaf((v.y - mean) * rstd, g.y, b.y),
                       fmaf((v.z - mean) * rstd, g.z, b.z), fmaf((v.w - mean) * rstd, g.w, b.w));
}

template <int kCache>
__device__ __forceinline__ void row_cached(const float* in, const float* gamma, const float* beta,
                                           float* out, int cols, float eps, int lane) {
    float4 v[kCache];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kCache; ++i) {
        const int j = lane * 4 + i * 128;
        v[i] = j < cols ? *reinterpret_cast<const float4*>(in + j) : make_float4(0, 0, 0, 0);
        sum += (v[i].x + v[i].y) + (v[i].z + v[i].w);
    }
    const float mean = warp_sum(sum) / cols;
    float sq = 0.0f;
#pragma unroll
    for (int i = 0; i < kCache; ++i) {
        if (lane * 4 + i * 128 >= cols) continue;  // padding is 0, not mean: keep it out
        const float a = v[i].x - mean, b = v[i].y - mean, c = v[i].z - mean, d = v[i].w - mean;
        sq = fmaf(a, a, fmaf(b, b, fmaf(c, c, fmaf(d, d, sq))));
    }
    const float rstd = rsqrtf(warp_sum(sq) / cols + eps);
#pragma unroll
    for (int i = 0; i < kCache; ++i) {
        const int j = lane * 4 + i * 128;
        if (j < cols)
            *reinterpret_cast<float4*>(out + j) =
                affine(v[i], mean, rstd, *reinterpret_cast<const float4*>(gamma + j),
                       *reinterpret_cast<const float4*>(beta + j));
    }
}

__device__ __forceinline__ void row_streamed_vec(const float* in, const float* gamma,
                                                 const float* beta, float* out, int cols, float eps,
                                                 int lane) {
    Welford w{0.0f, 0.0f, 0.0f};
    for (int j = lane * 4; j < cols; j += 128) {
        const float4 v = *reinterpret_cast<const float4*>(in + j);
        w.push(v.x);
        w.push(v.y);
        w.push(v.z);
        w.push(v.w);
    }
    w = warp_merge(w);
    const float rstd = rsqrtf(w.m2 / cols + eps);
    for (int j = lane * 4; j < cols; j += 128)
        *reinterpret_cast<float4*>(out + j) =
            affine(*reinterpret_cast<const float4*>(in + j), w.mean, rstd,
                   *reinterpret_cast<const float4*>(gamma + j), *reinterpret_cast<const float4*>(beta + j));
}

__device__ __forceinline__ void row_streamed_scalar(const float* in, const float* gamma,
                                                    const float* beta, float* out, int cols,
                                                    float eps, int lane) {
    Welford w{0.0f, 0.0f, 0.0f};
    for (int j = lane; j < cols; j += 32) w.push(in[j]);
    w = warp_merge(w);
    const float rstd = rsqrtf(w.m2 / cols + eps);
    for (int j = lane; j < cols; j += 32) out[j] = fmaf((in[j] - w.mean) * rstd, gamma[j], beta[j]);
}

}  // namespace

// kCache > 0: float4 path, row cached in registers. kCache == 0: float4 streamed (Welford).
// kCache < 0: scalar streamed (unaligned rows or cols % 4 != 0).
template <int kCache>
__global__ void layernorm_v2_kernel(const float* x, const float* gamma, const float* beta, float* y,
                                    int rows, int cols, float eps) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;  // whole warp exits together, so the shuffles stay full-warp
    const float* in = x + (size_t)row * cols;
    float* out = y + (size_t)row * cols;
    if constexpr (kCache > 0)
        row_cached<kCache>(in, gamma, beta, out, cols, eps, lane);
    else if constexpr (kCache == 0)
        row_streamed_vec(in, gamma, beta, out, cols, eps, lane);
    else
        row_streamed_scalar(in, gamma, beta, out, cols, eps, lane);
}


// Host-side dispatch lives here, next to the template: a __global__ template may only be launched
// from the translation unit that defines it (nvcc whole-program mode).
cudaError_t layernorm_v2_launch(int k, const float* x, const float* gamma, const float* beta,
                                float* y, int rows, int cols, float eps, cudaStream_t s) {
    const int grid = (rows + kLayerNormV2Threads / 32 - 1) / (kLayerNormV2Threads / 32);
#define KL_LAUNCH(K)     layernorm_v2_kernel<K><<<grid, kLayerNormV2Threads, 0, s>>>(x, gamma, beta, y, rows, cols, eps)
    switch (k) {
        case -1: KL_LAUNCH(-1); break;
        case 0: KL_LAUNCH(0); break;
        case 1: KL_LAUNCH(1); break;
        case 2: KL_LAUNCH(2); break;
        case 4: KL_LAUNCH(4); break;
        case 8: KL_LAUNCH(8); break;
        case 16: KL_LAUNCH(16); break;
        default: return cudaErrorInvalidValue;
    }
#undef KL_LAUNCH
    return cudaGetLastError();
}
