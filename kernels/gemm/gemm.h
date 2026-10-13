#pragma once
#include <cuda_fp16.h>

// C (M×N) = A (M×K) · B (K×N), row-major, FP32.
__global__ void gemm_v0_kernel(const float* A, const float* B, float* C, int M, int N, int K);
__global__ void gemm_v1_kernel(const float* A, const float* B, float* C, int M, int N, int K);

const int kGemmV2Tile = 32;  // block is kGemmV2Tile × kGemmV2Tile threads
__global__ void gemm_v2_kernel(const float* A, const float* B, float* C, int M, int N, int K);

const int kGemmV3Tile = 128;     // tile of C per block
const int kGemmV3Threads = 256;  // 1D block; each thread owns an 8×8 sub-tile
// float4 loads: needs K % 4 == 0, N % 4 == 0, and 16-byte-aligned A, B, C (and, when batched,
// strides that are multiples of 4). gridDim.z = batch count; matrix z of the batch starts
// z·stride floats past A, B, C. Strides default to 0, so a single GEMM launches as before.
__global__ void gemm_v3_kernel(const float* A, const float* B, float* C, int M, int N, int K,
                               long long strideA = 0, long long strideB = 0, long long strideC = 0);
// Same kernel with scalar global loads, for every other shape.
__global__ void gemm_v3_scalar_kernel(const float* A, const float* B, float* C, int M, int N,
                                      int K, long long strideA = 0, long long strideB = 0,
                                      long long strideC = 0);

// v4: A and B are FP16, C is FP32.
const int kGemmV4Tile = 128;     // tile of C per block
const int kGemmV4Threads = 256;  // 8 warps, each owns a 32×64 sub-tile
// 16-byte loads: needs K % 8 == 0, N % 8 == 0, 16-byte-aligned A and B, 32-byte-aligned C.
__global__ void gemm_v4_kernel(const __half* A, const __half* B, float* C, int M, int N, int K);
// Same kernel with scalar global loads and staged stores, for every other shape.
__global__ void gemm_v4_scalar_kernel(const __half* A, const __half* B, float* C, int M, int N,
                                      int K);
