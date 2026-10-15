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
// `batch` GEMMs in one launch; matrix z is A + z·strideA, B + z·strideB, C + z·strideC (floats).
int gemm_v3_batched(const float* A, const float* B, float* C, int M, int N, int K, int batch,
                    long long strideA, long long strideB, long long strideC, cudaStream_t s);
// FP16 A and B, FP32 C (tensor cores, FP32 accumulation).
int gemm_v4(const __half* A, const __half* B, float* C, int M, int N, int K, cudaStream_t s);
int gemm_cublas(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);

// y = softmax(x) along the last dim; x, y are rows×cols
int softmax_v0(const float* x, float* y, int rows, int cols, cudaStream_t s);
int softmax_v1(const float* x, float* y, int rows, int cols, cudaStream_t s);
int softmax_v2(const float* x, float* y, int rows, int cols, cudaStream_t s);
int softmax_v3(const float* x, float* y, int rows, int cols, cudaStream_t s);

// y = LayerNorm(x) along the last dim with affine gamma, beta (length cols)
int layernorm_v0(const float* x, const float* gamma, const float* beta, float* y, int rows,
                 int cols, float eps, cudaStream_t s);
int layernorm_v1(const float* x, const float* gamma, const float* beta, float* y, int rows,
                 int cols, float eps, cudaStream_t s);
int layernorm_v2(const float* x, const float* gamma, const float* beta, float* y, int rows,
                 int cols, float eps, cudaStream_t s);
int layernorm_v3(const float* x, const float* gamma, const float* beta, float* y, int rows,
                 int cols, float eps, cudaStream_t s);

// O = softmax(Q·Kᵀ / sqrt(d)) · V, unfused. Q, K, V, O are [heads, N, d] (heads = batch × heads);
// `work` must hold attention_workspace_bytes(heads, N, d) bytes: the N×N scores per head and Kᵀ.
size_t attention_workspace_bytes(int heads, int N, int d);
int attention_v0(const float* Q, const float* K, const float* V, float* O, float* work, int heads,
                 int N, int d, cudaStream_t s);
int attention_v1(const float* Q, const float* K, const float* V, float* O, float* work, int heads,
                 int N, int d, cudaStream_t s);
int attention_v2(const float* Q, const float* K, const float* V, float* O, float* work, int heads,
                 int N, int d, cudaStream_t s);

// GPT-2-style transformer block over the [B·S, D] residual stream x (updated in place), H heads,
// FP32, no biases. Weights are row-major [in, out]. `work` must hold
// transformer_block_workspace_bytes(B, S, D, H). transformer_block runs every step;
// transformer_block_step runs one (0 .. transformer_block_num_steps() - 1, named by
// transformer_block_step_name), which is how block/block.cu times them separately.
size_t transformer_block_workspace_bytes(int B, int S, int D, int H);
int transformer_block_num_steps(void);
const char* transformer_block_step_name(int step);
int transformer_block_step(int step, float* x, const float* ln1_g, const float* ln1_b,
                           const float* Wqkv, const float* Wo, const float* ln2_g,
                           const float* ln2_b, const float* W1, const float* W2, float* work, int B,
                           int S, int D, int H, cudaStream_t s);
int transformer_block(float* x, const float* ln1_g, const float* ln1_b, const float* Wqkv,
                      const float* Wo, const float* ln2_g, const float* ln2_b, const float* W1,
                      const float* W2, float* work, int B, int S, int D, int H, cudaStream_t s);

#ifdef __cplusplus
}
#endif
