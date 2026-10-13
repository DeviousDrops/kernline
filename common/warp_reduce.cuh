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

// True when every pointer is 16-byte aligned and `cols` keeps each row 16-byte aligned, i.e.
// float4 loads along a row are legal.
__host__ __device__ inline bool rows_vectorizable(int cols, const void* a, const void* b = nullptr,
                                                  const void* c = nullptr, const void* d = nullptr) {
    const auto bits = (size_t)a | (size_t)b | (size_t)c | (size_t)d;
    return cols % 4 == 0 && (bits & 15) == 0;
}
