// GEMM v1: coalesced. Same dot product as v0, but threadIdx.x walks the *columns* of C. With a
// 32-wide block a warp is one row of C: all 32 lanes read the same A element (one sector), read
// 32 consecutive floats of B, and write 32 consecutive floats of C. Still no reuse: every thread
// streams its full row of A and column of B, which v2 fixes with shared-memory tiles.
#include "kernels/gemm/gemm.h"

__global__ void gemm_v1_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
