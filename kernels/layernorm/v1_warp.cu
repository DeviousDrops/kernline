// LayerNorm v1: one warp per row, the same layout as softmax v1. Each lane strides along the row
// (float4 when kVec), so a warp's loads are contiguous; mean and variance are combined across
// lanes with __shfl_xor_sync butterflies. Still three passes over the row (mean, variance,
// normalize) and the two-pass variance.
//
// Accuracy: each lane sums only cols/32 values before the tree reduction, so the large-offset
// rows that broke v0's sequential sum (test_layernorm_survives_large_mean) now pass.
#include "common/warp_reduce.cuh"
#include "kernels/layernorm/layernorm.h"

namespace {

template <bool kVec>
__device__ __forceinline__ void layernorm_row(const float* in, const float* gamma,
                                              const float* beta, float* out, int cols, float eps,
                                              int lane) {
    float sum = 0.0f;
    if (kVec) {
        for (int j = lane * 4; j < cols; j += 128) {
            const float4 v = *reinterpret_cast<const float4*>(in + j);
            sum += (v.x + v.y) + (v.z + v.w);
        }
    } else {
        for (int j = lane; j < cols; j += 32) sum += in[j];
    }
    const float mean = warp_sum(sum) / cols;

    float sq = 0.0f;
    if (kVec) {
        for (int j = lane * 4; j < cols; j += 128) {
            const float4 v = *reinterpret_cast<const float4*>(in + j);
            const float a = v.x - mean, b = v.y - mean, c = v.z - mean, d = v.w - mean;
            sq = fmaf(a, a, fmaf(b, b, fmaf(c, c, fmaf(d, d, sq))));
        }
    } else {
        for (int j = lane; j < cols; j += 32) {
            const float d = in[j] - mean;
            sq = fmaf(d, d, sq);
        }
    }
    const float rstd = rsqrtf(warp_sum(sq) / cols + eps);

    if (kVec) {
        for (int j = lane * 4; j < cols; j += 128) {
            const float4 v = *reinterpret_cast<const float4*>(in + j);
            const float4 g = *reinterpret_cast<const float4*>(gamma + j);
            const float4 b = *reinterpret_cast<const float4*>(beta + j);
            *reinterpret_cast<float4*>(out + j) = make_float4(
                fmaf((v.x - mean) * rstd, g.x, b.x), fmaf((v.y - mean) * rstd, g.y, b.y),
                fmaf((v.z - mean) * rstd, g.z, b.z), fmaf((v.w - mean) * rstd, g.w, b.w));
        }
    } else {
        for (int j = lane; j < cols; j += 32) out[j] = fmaf((in[j] - mean) * rstd, gamma[j], beta[j]);
    }
}

template <bool kVec>
__device__ __forceinline__ void layernorm_v1_body(const float* x, const float* gamma,
                                                  const float* beta, float* y, int rows, int cols,
                                                  float eps) {
    const int warps = blockDim.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + threadIdx.x / 32;
    if (row >= rows) return;  // whole warp exits together, so the shuffles stay full-warp
    layernorm_row<kVec>(x + (size_t)row * cols, gamma, beta, y + (size_t)row * cols, cols, eps,
                        lane);
}

}  // namespace

__global__ void layernorm_v1_kernel(const float* x, const float* gamma, const float* beta,
                                    float* y, int rows, int cols, float eps) {
    layernorm_v1_body<true>(x, gamma, beta, y, rows, cols, eps);
}

__global__ void layernorm_v1_scalar_kernel(const float* x, const float* gamma, const float* beta,
                                           float* y, int rows, int cols, float eps) {
    layernorm_v1_body<false>(x, gamma, beta, y, rows, cols, eps);
}
