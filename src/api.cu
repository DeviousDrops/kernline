#include "src/api.h"
#include <cublas_v2.h>
#include "kernels/gemm/gemm.h"

extern "C" int gemm_v0(const float* A, const float* B, float* C, int M, int N, int K,
                       cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    gemm_v0_kernel<<<grid, block, 0, s>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

// Baseline (D9). cuBLAS is column-major: a row-major M×N matrix is its N×M transpose, so
// C = A·B becomes Cᵀ = Bᵀ·Aᵀ, i.e. the same buffers with the operands swapped.
extern "C" int gemm_cublas(const float* A, const float* B, float* C, int M, int N, int K,
                           cudaStream_t s) {
    static cublasHandle_t handle = nullptr;
    if (!handle && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    const float one = 1.0f, zero = 0.0f;
    cublasSetStream(handle, s);
    cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K,
                                    &zero, C, N);
    return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}
