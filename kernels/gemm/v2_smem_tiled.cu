// GEMM v2: shared-memory tiling. Each 32×32 block of threads computes a 32×32 tile of C. Per step
// along K it loads one 32×32 tile of A and one of B into shared memory (coalesced, each element
// read from global memory once per block), then every thread does 32 multiply-adds from shared
// memory. Global loads per FLOP drop 32× versus v1; the inner loop is now shared-memory bound.
// In the inner loop a warp (fixed ty) reads one As element (broadcast) and 32 consecutive Bs
// elements. A 1024-thread block caps occupancy at one block per SM (32 of 48 warps). Edge tiles
// are zero-padded.
#include "kernels/gemm/gemm.h"

__global__ void gemm_v2_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int T = kGemmV2Tile;
    __shared__ float As[T][T], Bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;
    const int col = blockIdx.x * T + tx;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
