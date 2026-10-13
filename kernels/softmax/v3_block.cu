// Softmax v3: one block per row, the row cached in shared memory. Each element crosses DRAM
// exactly once (read once, written once) for any row that fits in shared memory: up to ~24.6k
// floats with the 4060/4070's 99 KB per-block opt-in. That's what v2 could only do for rows of
// <= 2048 floats, held in one warp's registers.
//
//   pass 1  global -> shared, with a running max per thread   (the only DRAM read)
//   pass 2  shared: e = exp(x - max), stored back, summed
//   pass 3  shared -> global: e / sum                         (the only DRAM write)
//
// Each thread touches the same shared indices in every pass, so passes need no barriers of their
// own; the block reductions (common/warp_reduce.cuh) provide the only __syncthreads. Rows that
// don't fit fall back to v2's streamed online softmax (see softmax_v3 in src/api.cu).
#include "common/warp_reduce.cuh"
#include "kernels/softmax/softmax.h"

template <bool kVec>
__global__ void softmax_v3_kernel(const float* x, float* y, int cols) {
    extern __shared__ __align__(16) float row[];
    __shared__ float scratch[32];
    const float* in = x + (size_t)blockIdx.x * cols;
    float* out = y + (size_t)blockIdx.x * cols;
    const int t = threadIdx.x, n = blockDim.x;

    float mx = -INFINITY;
    if (kVec) {
        float4* r4 = reinterpret_cast<float4*>(row);
#pragma unroll 4
        for (int j = t; j < cols / 4; j += n) {
            const float4 v = reinterpret_cast<const float4*>(in)[j];
            r4[j] = v;
            mx = fmaxf(mx, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
    } else {
        for (int j = t; j < cols; j += n) {
            row[j] = in[j];
            mx = fmaxf(mx, row[j]);
        }
    }
    mx = block_max(mx, scratch);

    float sum = 0.0f;
    if (kVec) {
        float4* r4 = reinterpret_cast<float4*>(row);
        for (int j = t; j < cols / 4; j += n) {
            float4 v = r4[j];
            v = make_float4(expf(v.x - mx), expf(v.y - mx), expf(v.z - mx), expf(v.w - mx));
            r4[j] = v;
            sum += (v.x + v.y) + (v.z + v.w);
        }
    } else {
        for (int j = t; j < cols; j += n) {
            row[j] = expf(row[j] - mx);
            sum += row[j];
        }
    }
    const float inv = 1.0f / block_sum(sum, scratch);

    if (kVec) {
        const float4* r4 = reinterpret_cast<const float4*>(row);
        for (int j = t; j < cols / 4; j += n) {
            const float4 v = r4[j];
            reinterpret_cast<float4*>(out)[j] = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
        }
    } else {
        for (int j = t; j < cols; j += n) out[j] = row[j] * inv;
    }
}

// Host-side dispatch next to the template (a __global__ template is launched only from the
// translation unit that defines it).
cudaError_t softmax_v3_launch(bool vec, int threads, size_t smem, const float* x, float* y,
                              int rows, int cols, cudaStream_t s) {
    auto kernel = vec ? softmax_v3_kernel<true> : softmax_v3_kernel<false>;
    // Dynamic shared memory above 48 KB needs an explicit opt-in per kernel; it's cheap and
    // idempotent, so set it on every launch instead of caching per device.
    cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int)smem);
    if (err != cudaSuccess) return err;
    kernel<<<rows, threads, smem, s>>>(x, y, cols);
    return cudaGetLastError();
}
