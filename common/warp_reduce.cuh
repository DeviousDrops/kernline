#pragma once
#include <cuda_runtime.h>

// Butterfly reductions over a full warp with __shfl_xor_sync: after log2(32) = 5 steps every lane
// holds the result, so no broadcast is needed afterwards. All 32 lanes must call these.

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

// Block-wide reductions: warp butterfly, one partial per warp in `scratch` (>= 32 floats of shared
// memory), then warp 0 reduces the partials. Every thread gets the result. All threads of the
// block must call these, and `scratch` may be reused right after the call returns.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = (blockDim.x + 31) / 32;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = lane < warps ? scratch[lane] : 0.0f;
    v = warp_sum(v);  // every warp reduces the partials itself: no second broadcast barrier
    __syncthreads();  // all reads of scratch done before anyone reuses it
    return v;
}

__device__ __forceinline__ float block_max(float v, float* scratch) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = (blockDim.x + 31) / 32;
    v = warp_max(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = lane < warps ? scratch[lane] : -INFINITY;
    v = warp_max(v);
    __syncthreads();
    return v;
}

// True when every pointer is 16-byte aligned and `cols` keeps each row 16-byte aligned, i.e.
// float4 loads along a row are legal.
__host__ __device__ inline bool rows_vectorizable(int cols, const void* a, const void* b = nullptr,
                                                  const void* c = nullptr, const void* d = nullptr) {
    const auto bits = (size_t)a | (size_t)b | (size_t)c | (size_t)d;
    return cols % 4 == 0 && (bits & 15) == 0;
}
