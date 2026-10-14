#pragma once

// Elementwise and layout kernels the transformer block (block/block.cu, D4) needs between the
// GEMM, LayerNorm and attention launches. All are memory-bound grid-stride loops.

// y[i] += x[i]: the residual connection.
__global__ void add_inplace_kernel(float* y, const float* x, long long n);

// x[i] = GELU(x[i]), tanh approximation (GPT-2's gelu_new).
__global__ void gelu_inplace_kernel(float* x, long long n);

// qkv is [B·S, 3·D] with D = H·d (columns: all of Q, then K, then V; head h at columns h·d..);
// q, k, v are [B·H, S, d], the layout attention_v* takes. Launch with gridDim.y = B·S tokens.
__global__ void split_heads_kernel(const float* qkv, float* q, float* k, float* v, int B, int S,
                                   int H, int d);

// o is [B·H, S, d]; y is [B·S, D]: the inverse of split_heads for one tensor. gridDim.y = B·S.
__global__ void merge_heads_kernel(const float* o, float* y, int B, int S, int H, int d);
