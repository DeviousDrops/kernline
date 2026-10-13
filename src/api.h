#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Stable C ABI of libkernels.so (docs/DESIGN.md, "Frozen interfaces").
// Row-major device pointers. Every launcher returns a cudaError_t cast to int.
#ifdef __cplusplus
extern "C" {
#endif

// C (M×N) = A (M×K) · B (K×N)
int gemm_v0(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
int gemm_v1(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
int gemm_v2(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
int gemm_v3(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
// FP16 A and B, FP32 C (tensor cores, FP32 accumulation).
int gemm_v4(const __half* A, const __half* B, float* C, int M, int N, int K, cudaStream_t s);
int gemm_cublas(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);

// y = softmax(x) along the last dim; x, y are rows×cols
int softmax_v0(const float* x, float* y, int rows, int cols, cudaStream_t s);
int softmax_v1(const float* x, float* y, int rows, int cols, cudaStream_t s);
int softmax_v2(const float* x, float* y, int rows, int cols, cudaStream_t s);

// y = LayerNorm(x) along the last dim with affine gamma, beta (length cols)
int layernorm_v0(const float* x, const float* gamma, const float* beta, float* y, int rows,
                 int cols, float eps, cudaStream_t s);
int layernorm_v1(const float* x, const float* gamma, const float* beta, float* y, int rows,
                 int cols, float eps, cudaStream_t s);
int layernorm_v2(const float* x, const float* gamma, const float* beta, float* y, int rows,
                 int cols, float eps, cudaStream_t s);

#ifdef __cplusplus
}
#endif
