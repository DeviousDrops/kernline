// Softmax v2: one warp per row, each element read from DRAM as few times as possible.
//
//  - Cached (kCache > 0, cols <= 128·kCache): the lane loads its kCache float4s of the row into
//    registers once, then computes max, sum and output from registers. One read and one write per
//    element, independent of whether the row would still be in L2 for a second pass.
//  - Streamed (kCache == 0, longer rows): online softmax. One read pass keeps a running max m and
//    a sum s rescaled whenever m grows (s·exp(m_old − m_new)), so max and sum come from the same
//    pass; a second pass writes exp(x − m)/s. Two reads instead of v1's three.
//
// Lanes merge their (m, s) pairs with __shfl_xor_sync: m = max(m_a, m_b),
// s = s_a·exp(m_a − m) + s_b·exp(m_b − m). A lane that saw no elements has (−inf, 0), and its
// term is defined as 0 instead of 0·exp(−inf − (−inf)) = NaN.
#include "common/warp_reduce.cuh"
#include "kernels/softmax/softmax.h"

namespace {

struct MaxSum {
    float m, s;
};

__device__ __forceinline__ float rescale(float s, float m_from, float m_to) {
    return m_from == -INFINITY ? 0.0f : s * expf(m_from - m_to);
}

__device__ __forceinline__ MaxSum merge(MaxSum a, MaxSum b) {
    const float m = fmaxf(a.m, b.m);
    return {m, rescale(a.s, a.m, m) + rescale(b.s, b.m, m)};
}

__device__ __forceinline__ MaxSum warp_merge(MaxSum v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        MaxSum o{__shfl_xor_sync(0xffffffffu, v.m, offset), __shfl_xor_sync(0xffffffffu, v.s, offset)};
        v = merge(v, o);
    }
    return v;
}

// Online update with four new values: rescale once per float4, not once per element.
__device__ __forceinline__ void online4(MaxSum& a, float4 v) {
    const float mm = fmaxf(a.m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    a.s = rescale(a.s, a.m, mm) + expf(v.x - mm) + expf(v.y - mm) + expf(v.z - mm) + expf(v.w - mm);
    a.m = mm;
}

template <int kCache>
__device__ __forceinline__ void row_cached(const float* in, float* out, int cols, int lane) {
    float4 v[kCache];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < kCache; ++i) {
        const int j = lane * 4 + i * 128;
        v[i] = j < cols ? *reinterpret_cast<const float4*>(in + j)
                        : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        mx = fmaxf(mx, fmaxf(fmaxf(v[i].x, v[i].y), fmaxf(v[i].z, v[i].w)));
    }
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kCache; ++i) {  // exp(-inf - mx) = 0 for the padding
        v[i] = make_float4(expf(v[i].x - mx), expf(v[i].y - mx), expf(v[i].z - mx), expf(v[i].w - mx));
        sum += (v[i].x + v[i].y) + (v[i].z + v[i].w);
    }
    const float inv = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < kCache; ++i) {
        const int j = lane * 4 + i * 128;
        if (j < cols)
            *reinterpret_cast<float4*>(out + j) =
                make_float4(v[i].x * inv, v[i].y * inv, v[i].z * inv, v[i].w * inv);
    }
}

__device__ __forceinline__ void row_streamed_vec(const float* in, float* out, int cols, int lane) {
    MaxSum a{-INFINITY, 0.0f};
    for (int j = lane * 4; j < cols; j += 128) online4(a, *reinterpret_cast<const float4*>(in + j));
    a = warp_merge(a);
    const float inv = 1.0f / a.s;
    for (int j = lane * 4; j < cols; j += 128) {
        const float4 v = *reinterpret_cast<const float4*>(in + j);
        *reinterpret_cast<float4*>(out + j) = make_float4(
            expf(v.x - a.m) * inv, expf(v.y - a.m) * inv, expf(v.z - a.m) * inv, expf(v.w - a.m) * inv);
    }
}

__device__ __forceinline__ void row_streamed_scalar(const float* in, float* out, int cols, int lane) {
    MaxSum a{-INFINITY, 0.0f};
    for (int j = lane; j < cols; j += 32) {
        const float mm = fmaxf(a.m, in[j]);
        a.s = rescale(a.s, a.m, mm) + expf(in[j] - mm);
        a.m = mm;
    }
    a = warp_merge(a);
    const float inv = 1.0f / a.s;
    for (int j = lane; j < cols; j += 32) out[j] = expf(in[j] - a.m) * inv;
}

}  // namespace

// kCache > 0: float4 path, row cached in registers. kCache == 0: float4 streamed. kCache < 0:
// scalar streamed (unaligned rows or cols % 4 != 0).
template <int kCache>
__global__ void softmax_v2_kernel(const float* x, float* y, int rows, int cols) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;  // whole warp exits together, so the shuffles stay full-warp
    const float* in = x + (size_t)row * cols;
    float* out = y + (size_t)row * cols;
    if constexpr (kCache > 0)
        row_cached<kCache>(in, out, cols, lane);
    else if constexpr (kCache == 0)
        row_streamed_vec(in, out, cols, lane);
    else
        row_streamed_scalar(in, out, cols, lane);
}


// Host-side dispatch lives here, next to the template: a __global__ template may only be launched
// from the translation unit that defines it (nvcc whole-program mode).
cudaError_t softmax_v2_launch(int k, const float* x, float* y, int rows, int cols, cudaStream_t s) {
    const int grid = (rows + kSoftmaxV2Threads / 32 - 1) / (kSoftmaxV2Threads / 32);
#define KL_LAUNCH(K) softmax_v2_kernel<K><<<grid, kSoftmaxV2Threads, 0, s>>>(x, y, rows, cols)
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
