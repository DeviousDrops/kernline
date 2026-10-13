// Softmax v1: one warp per row. Lanes stride along the row, so a warp's loads are contiguous:
// with float4 (kVec) a warp reads 512 consecutive bytes per instruction, four full 128-byte lines.
// Max and sum are combined across lanes with __shfl_xor_sync butterflies (common/warp_reduce.cuh),
// no shared memory or __syncthreads. A row of 1024 floats is 8 float4 loads per lane per pass.
//
// Still three passes over the row (max, sum, write), as in v0. Passes 2 and 3 re-read data the
// warp just touched; whether that hits L1/L2 or goes back to DRAM is what v2 removes.
#include "common/warp_reduce.cuh"
#include "kernels/softmax/softmax.h"

namespace {

template <bool kVec>
__device__ __forceinline__ void softmax_row(const float* in, float* out, int cols, int lane) {
    float mx = -INFINITY;
    if (kVec) {
        for (int j = lane * 4; j < cols; j += 128) {
            const float4 v = *reinterpret_cast<const float4*>(in + j);
            mx = fmaxf(mx, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
    } else {
        for (int j = lane; j < cols; j += 32) mx = fmaxf(mx, in[j]);
    }
    mx = warp_max(mx);

    float sum = 0.0f;
    if (kVec) {
        for (int j = lane * 4; j < cols; j += 128) {
            const float4 v = *reinterpret_cast<const float4*>(in + j);
            sum += expf(v.x - mx) + expf(v.y - mx) + expf(v.z - mx) + expf(v.w - mx);
        }
    } else {
        for (int j = lane; j < cols; j += 32) sum += expf(in[j] - mx);
    }
    const float inv = 1.0f / warp_sum(sum);

    if (kVec) {
        for (int j = lane * 4; j < cols; j += 128) {
            const float4 v = *reinterpret_cast<const float4*>(in + j);
            *reinterpret_cast<float4*>(out + j) =
                make_float4(expf(v.x - mx) * inv, expf(v.y - mx) * inv, expf(v.z - mx) * inv,
                            expf(v.w - mx) * inv);
        }
    } else {
        for (int j = lane; j < cols; j += 32) out[j] = expf(in[j] - mx) * inv;
    }
}

template <bool kVec>
__device__ __forceinline__ void softmax_v1_body(const float* x, float* y, int rows, int cols) {
    const int warps = blockDim.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + threadIdx.x / 32;
    if (row >= rows) return;  // whole warp exits together, so the shuffles stay full-warp
    softmax_row<kVec>(x + (size_t)row * cols, y + (size_t)row * cols, cols, lane);
}

}  // namespace

__global__ void softmax_v1_kernel(const float* x, float* y, int rows, int cols) {
    softmax_v1_body<true>(x, y, rows, cols);
}

__global__ void softmax_v1_scalar_kernel(const float* x, float* y, int rows, int cols) {
    softmax_v1_body<false>(x, y, rows, cols);
}
