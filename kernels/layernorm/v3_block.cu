// LayerNorm v3: one block per row, the row cached in shared memory, the same structure as
// softmax v3. Each element crosses DRAM exactly once for any row that fits in shared memory
// (~24.6k floats), and the variance is the exact two-pass form over shared memory:
//
//   pass 1  global -> shared, with a running sum per thread   (the only DRAM read of x)
//   pass 2  shared: sum of (x - mean)^2
//   pass 3  shared -> global: (x - mean)·rstd·gamma + beta     (the only DRAM write)
//
// gamma and beta are read in pass 3. They are shared by every row, so after the first few rows
// they come from L2. Rows that don't fit fall back to v2's streamed Welford (src/api.cu).
#include "common/warp_reduce.cuh"
#include "kernels/layernorm/layernorm.h"

template <bool kVec>
__global__ void layernorm_v3_kernel(const float* x, const float* gamma, const float* beta, float* y,
                                    int cols, float eps) {
    extern __shared__ __align__(16) float row[];
    __shared__ float scratch[32];
    const float* in = x + (size_t)blockIdx.x * cols;
    float* out = y + (size_t)blockIdx.x * cols;
    const int t = threadIdx.x, n = blockDim.x;

    float sum = 0.0f;
    if (kVec) {
        float4* r4 = reinterpret_cast<float4*>(row);
#pragma unroll 4
        for (int j = t; j < cols / 4; j += n) {
            const float4 v = reinterpret_cast<const float4*>(in)[j];
            r4[j] = v;
            sum += (v.x + v.y) + (v.z + v.w);
        }
    } else {
        for (int j = t; j < cols; j += n) {
            row[j] = in[j];
            sum += row[j];
        }
    }
    const float mean = block_sum(sum, scratch) / cols;

    float sq = 0.0f;
    if (kVec) {
        const float4* r4 = reinterpret_cast<const float4*>(row);
        for (int j = t; j < cols / 4; j += n) {
            const float4 v = r4[j];
            const float a = v.x - mean, b = v.y - mean, c = v.z - mean, d = v.w - mean;
            sq = fmaf(a, a, fmaf(b, b, fmaf(c, c, fmaf(d, d, sq))));
        }
    } else {
        for (int j = t; j < cols; j += n) {
            const float d = row[j] - mean;
            sq = fmaf(d, d, sq);
        }
    }
    const float rstd = rsqrtf(block_sum(sq, scratch) / cols + eps);

    if (kVec) {
        const float4* r4 = reinterpret_cast<const float4*>(row);
        for (int j = t; j < cols / 4; j += n) {
            const float4 v = r4[j];
            const float4 g = reinterpret_cast<const float4*>(gamma)[j];
            const float4 b = reinterpret_cast<const float4*>(beta)[j];
            reinterpret_cast<float4*>(out)[j] = make_float4(
                fmaf((v.x - mean) * rstd, g.x, b.x), fmaf((v.y - mean) * rstd, g.y, b.y),
                fmaf((v.z - mean) * rstd, g.z, b.z), fmaf((v.w - mean) * rstd, g.w, b.w));
        }
    } else {
        for (int j = t; j < cols; j += n) out[j] = fmaf((row[j] - mean) * rstd, gamma[j], beta[j]);
    }
}

// Host-side dispatch next to the template (see softmax v3).
cudaError_t layernorm_v3_launch(bool vec, int threads, size_t smem, const float* x,
                                const float* gamma, const float* beta, float* y, int rows, int cols,
                                float eps, cudaStream_t s) {
    auto kernel = vec ? layernorm_v3_kernel<true> : layernorm_v3_kernel<false>;
    cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int)smem);
    if (err != cudaSuccess) return err;
    kernel<<<rows, threads, smem, s>>>(x, gamma, beta, y, cols, eps);
    return cudaGetLastError();
}
