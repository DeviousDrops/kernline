#pragma once
#include <cuda_runtime.h>

// Stable C ABI of libkernels.so (docs/DESIGN.md, "Frozen interfaces").
// Row-major device pointers. Every launcher returns a cudaError_t cast to int.
#ifdef __cplusplus
extern "C" {
#endif

// C (M×N) = A (M×K) · B (K×N)
int gemm_v0(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
int gemm_v1(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
int gemm_cublas(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);

#ifdef __cplusplus
}
#endif
