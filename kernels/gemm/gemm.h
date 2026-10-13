#pragma once

// C (M×N) = A (M×K) · B (K×N), row-major, FP32.
__global__ void gemm_v0_kernel(const float* A, const float* B, float* C, int M, int N, int K);
__global__ void gemm_v1_kernel(const float* A, const float* B, float* C, int M, int N, int K);

const int kGemmV2Tile = 32;  // block is kGemmV2Tile × kGemmV2Tile threads
__global__ void gemm_v2_kernel(const float* A, const float* B, float* C, int M, int N, int K);
