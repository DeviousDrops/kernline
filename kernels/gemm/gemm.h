#pragma once

// C (M×N) = A (M×K) · B (K×N), row-major, FP32.
__global__ void gemm_v0_kernel(const float* A, const float* B, float* C, int M, int N, int K);
