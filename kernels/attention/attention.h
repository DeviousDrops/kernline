#pragma once

// Helpers for unfused attention (src/api.cu composes them with the GEMM and softmax launchers).
// Tensors are [heads, rows, cols] contiguous, row-major per head; `heads` is batch × heads.

// out[h] = in[h]ᵀ · scale for each head: in is rows×cols, out is cols×rows. 32×32 tiles through
// shared memory (padded to 33 columns against bank conflicts), so reads and writes both coalesce.
__global__ void transpose_scale_kernel(const float* in, float* out, int rows, int cols, float scale);
const int kTransposeTile = 32;  // block is 32×8 threads, each moving 4 elements

// x[i] *= scale for i < n. Attention v0's separate pass over the N×N scores.
__global__ void scale_kernel(float* x, long long n, float scale);
