#include "src/api.h"
#include <cstdint>
#include <cublas_v2.h>
#include "common/warp_reduce.cuh"
#include "kernels/gemm/gemm.h"
#include "kernels/layernorm/layernorm.h"
#include "kernels/softmax/softmax.h"

extern "C" int gemm_v0(const float* A, const float* B, float* C, int M, int N, int K,
                       cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    gemm_v0_kernel<<<grid, block, 0, s>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

extern "C" int gemm_v1(const float* A, const float* B, float* C, int M, int N, int K,
                       cudaStream_t s) {
    dim3 block(32, 8);  // x = columns: one warp per row of C
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_v1_kernel<<<grid, block, 0, s>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

extern "C" int gemm_v2(const float* A, const float* B, float* C, int M, int N, int K,
                       cudaStream_t s) {
    dim3 block(kGemmV2Tile, kGemmV2Tile);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_v2_kernel<<<grid, block, 0, s>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

extern "C" int gemm_v3(const float* A, const float* B, float* C, int M, int N, int K,
                       cudaStream_t s) {
    dim3 grid((N + kGemmV3Tile - 1) / kGemmV3Tile, (M + kGemmV3Tile - 1) / kGemmV3Tile);
    const bool aligned = (((uintptr_t)A | (uintptr_t)B | (uintptr_t)C) & 15) == 0;
    if (aligned && K % 4 == 0 && N % 4 == 0)
        gemm_v3_kernel<<<grid, kGemmV3Threads, 0, s>>>(A, B, C, M, N, K);
    else
        gemm_v3_scalar_kernel<<<grid, kGemmV3Threads, 0, s>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

extern "C" int gemm_v4(const __half* A, const __half* B, float* C, int M, int N, int K,
                       cudaStream_t s) {
    dim3 grid((N + kGemmV4Tile - 1) / kGemmV4Tile, (M + kGemmV4Tile - 1) / kGemmV4Tile);
    const bool aligned = (((uintptr_t)A | (uintptr_t)B) & 15) == 0 && ((uintptr_t)C & 31) == 0;
    if (aligned && K % 8 == 0 && N % 8 == 0)
        gemm_v4_kernel<<<grid, kGemmV4Threads, 0, s>>>(A, B, C, M, N, K);
    else
        gemm_v4_scalar_kernel<<<grid, kGemmV4Threads, 0, s>>>(A, B, C, M, N, K);
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

// ---- softmax / layernorm: row-wise over the last dim of a rows×cols matrix

extern "C" int softmax_v0(const float* x, float* y, int rows, int cols, cudaStream_t s) {
    const int block = 128;
    softmax_v0_kernel<<<(rows + block - 1) / block, block, 0, s>>>(x, y, rows, cols);
    return cudaGetLastError();
}

extern "C" int layernorm_v0(const float* x, const float* gamma, const float* beta, float* y,
                            int rows, int cols, float eps, cudaStream_t s) {
    const int block = 128;
    layernorm_v0_kernel<<<(rows + block - 1) / block, block, 0, s>>>(x, gamma, beta, y, rows,
                                                                      cols, eps);
    return cudaGetLastError();
}

extern "C" int softmax_v1(const float* x, float* y, int rows, int cols, cudaStream_t s) {
    const int rows_per_block = kSoftmaxV1Threads / 32;
    const int grid = (rows + rows_per_block - 1) / rows_per_block;
    if (rows_vectorizable(cols, x, y))
        softmax_v1_kernel<<<grid, kSoftmaxV1Threads, 0, s>>>(x, y, rows, cols);
    else
        softmax_v1_scalar_kernel<<<grid, kSoftmaxV1Threads, 0, s>>>(x, y, rows, cols);
    return cudaGetLastError();
}

extern "C" int layernorm_v1(const float* x, const float* gamma, const float* beta, float* y,
                            int rows, int cols, float eps, cudaStream_t s) {
    const int rows_per_block = kLayerNormV1Threads / 32;
    const int grid = (rows + rows_per_block - 1) / rows_per_block;
    if (rows_vectorizable(cols, x, gamma, beta, y))
        layernorm_v1_kernel<<<grid, kLayerNormV1Threads, 0, s>>>(x, gamma, beta, y, rows, cols,
                                                                  eps);
    else
        layernorm_v1_scalar_kernel<<<grid, kLayerNormV1Threads, 0, s>>>(x, gamma, beta, y, rows,
                                                                         cols, eps);
    return cudaGetLastError();
}
