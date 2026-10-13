// GEMM v0: naive baseline. One thread per element of C, full K-long dot product from DRAM.
// threadIdx.x walks the *rows* of C on purpose: neighbouring threads in a warp read A with a
// stride of K floats and write C with a stride of N floats, so nothing coalesces. v1 fixes that.
#include "kernels/gemm/gemm.h"

__global__ void gemm_v0_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
