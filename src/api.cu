#include "src/api.h"
#include <algorithm>
#include <cstdint>
#include <cublas_v2.h>
#include <cmath>
#include "common/warp_reduce.cuh"
#include "kernels/attention/attention.h"
#include "kernels/block/block_ops.h"
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

// `batch` independent GEMMs in one launch (gridDim.z = batch): matrix z is A + z·strideA etc.
// One launch fills the GPU even when each matrix is a handful of 128×128 tiles.
extern "C" int gemm_v3_batched(const float* A, const float* B, float* C, int M, int N, int K,
                               int batch, long long strideA, long long strideB, long long strideC,
                               cudaStream_t s) {
    dim3 grid((N + kGemmV3Tile - 1) / kGemmV3Tile, (M + kGemmV3Tile - 1) / kGemmV3Tile, batch);
    const bool aligned = (((uintptr_t)A | (uintptr_t)B | (uintptr_t)C) & 15) == 0 &&
                         ((strideA | strideB | strideC) & 3) == 0;
    if (aligned && K % 4 == 0 && N % 4 == 0)
        gemm_v3_kernel<<<grid, kGemmV3Threads, 0, s>>>(A, B, C, M, N, K, strideA, strideB, strideC);
    else
        gemm_v3_scalar_kernel<<<grid, kGemmV3Threads, 0, s>>>(A, B, C, M, N, K, strideA, strideB,
                                                              strideC);
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

// v2 register-cache size: the smallest kCache with 128·kCache >= cols, or 0 (stream) past 2048.
// 16 float4 = 64 registers of row data per lane is the largest that keeps 3 blocks per SM.
static int v2_cache(int cols) {
    for (int k : {1, 2, 4, 8, 16})
        if (cols <= 128 * k) return k;
    return 0;
}

extern "C" int softmax_v2(const float* x, float* y, int rows, int cols, cudaStream_t s) {
    const int k = rows_vectorizable(cols, x, y) ? v2_cache(cols) : -1;
    return softmax_v2_launch(k, x, y, rows, cols, s);
}

extern "C" int layernorm_v2(const float* x, const float* gamma, const float* beta, float* y,
                            int rows, int cols, float eps, cudaStream_t s) {
    const int k = rows_vectorizable(cols, x, gamma, beta, y) ? v2_cache(cols) : -1;
    return layernorm_v2_launch(k, x, gamma, beta, y, rows, cols, eps, s);
}

// v3 launch shape for a row of `cols` floats: about 4 float4 per thread, 128..1024 threads, and
// the row's bytes of dynamic shared memory. Returns false if the row does not fit in the
// per-block opt-in shared memory (minus the 128 bytes of static reduction scratch).
static bool v3_shape(int cols, int* threads, size_t* smem) {
    static int optin = -1;
    if (optin < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    }
    *smem = (size_t)cols * sizeof(float);
    if (*smem + 32 * sizeof(float) > (size_t)optin) return false;
    const int want = ((cols + 15) / 16 + 31) / 32 * 32;
    *threads = want < 128 ? 128 : want > 1024 ? 1024 : want;
    return true;
}

extern "C" int softmax_v3(const float* x, float* y, int rows, int cols, cudaStream_t s) {
    const bool vec = rows_vectorizable(cols, x, y);
    int threads;
    size_t smem;
    if (!v3_shape(cols, &threads, &smem))
        return softmax_v2_launch(vec ? 0 : -1, x, y, rows, cols, s);
    return softmax_v3_launch(vec, threads, smem, x, y, rows, cols, s);
}

extern "C" int layernorm_v3(const float* x, const float* gamma, const float* beta, float* y,
                            int rows, int cols, float eps, cudaStream_t s) {
    const bool vec = rows_vectorizable(cols, x, gamma, beta, y);
    int threads;
    size_t smem;
    if (!v3_shape(cols, &threads, &smem))
        return layernorm_v2_launch(vec ? 0 : -1, x, gamma, beta, y, rows, cols, eps, s);
    return layernorm_v3_launch(vec, threads, smem, x, gamma, beta, y, rows, cols, eps, s);
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

// ---- attention: O = softmax(Q·Kᵀ / sqrt(d)) · V per head, unfused (D10)
//
// Q, K, V, O are [heads, N, d]; `work` holds the scores S [heads, N, N] followed by Kᵀ
// [heads, d, N] (attention_workspace_bytes). The N×N scores go through DRAM between kernels,
// which is the cost this unfused design exists to measure. GEMMs run once per head.

extern "C" size_t attention_workspace_bytes(int heads, int N, int d) {
    return sizeof(float) * ((size_t)heads * N * N + (size_t)heads * d * N);
}

static cudaError_t transpose_heads(const float* in, float* out, int heads, int rows, int cols,
                                   float scale, cudaStream_t s) {
    dim3 block(kTransposeTile, 8);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile, heads);
    transpose_scale_kernel<<<grid, block, 0, s>>>(in, out, rows, cols, scale);
    return cudaGetLastError();
}

typedef int (*GemmLauncher)(const float*, const float*, float*, int, int, int, cudaStream_t);

// C_h = A_h · B_h for every head h: the GEMMs on either side of softmax.
static int per_head_gemm(GemmLauncher gemm, const float* A, const float* B, float* C, int heads,
                         int M, int N, int K, cudaStream_t s) {
    for (int h = 0; h < heads; ++h) {
        const int err = gemm(A + (size_t)h * M * K, B + (size_t)h * K * N, C + (size_t)h * M * N,
                             M, N, K, s);
        if (err) return err;
    }
    return cudaSuccess;
}

// v0: everything naive and separate. Kᵀ, S = Q·Kᵀ (gemm_v0), a full extra pass scaling S by
// 1/sqrt(d), softmax_v0 in place, O = P·V (gemm_v0).
extern "C" int attention_v0(const float* Q, const float* K, const float* V, float* O, float* work,
                            int heads, int N, int d, cudaStream_t s) {
    float* S = work;
    float* Kt = work + (size_t)heads * N * N;
    int err = transpose_heads(K, Kt, heads, N, d, 1.0f, s);
    if (!err) err = per_head_gemm(gemm_v0, Q, Kt, S, heads, N, N, d, s);
    if (!err) {
        scale_kernel<<<1024, 256, 0, s>>>(S, (long long)heads * N * N, 1.0f / std::sqrt((float)d));
        err = cudaGetLastError();
    }
    if (!err) err = softmax_v0(S, S, heads * N, N, s);
    if (!err) err = per_head_gemm(gemm_v0, S, V, O, heads, N, d, N, s);
    return err;
}

// v1: the best kernels, and the scale folded away. Kᵀ is written pre-multiplied by 1/sqrt(d), so
// S = Q·(K/sqrt(d))ᵀ comes out of gemm_v3 already scaled, with no extra pass over the N×N
// scores. Softmax runs in place with the variant that reads each score once: v2 (row cached in
// one warp's registers) up to N = 2048, v3 (row in one block's shared memory) above.
extern "C" int attention_v1(const float* Q, const float* K, const float* V, float* O, float* work,
                            int heads, int N, int d, cudaStream_t s) {
    float* S = work;
    float* Kt = work + (size_t)heads * N * N;
    int err = transpose_heads(K, Kt, heads, N, d, 1.0f / std::sqrt((float)d), s);
    if (!err) err = per_head_gemm(gemm_v3, Q, Kt, S, heads, N, N, d, s);
    if (!err)
        err = N <= 2048 ? softmax_v2(S, S, heads * N, N, s) : softmax_v3(S, S, heads * N, N, s);
    if (!err) err = per_head_gemm(gemm_v3, S, V, O, heads, N, d, N, s);
    return err;
}

// v2: v1 with each GEMM batched over heads into one launch (gemm_v3_batched). At short N a head's
// Q·Kᵀ is only (N/128)² tiles, e.g. 4 blocks at N = 256 on a 24/36-SM GPU, and v1 runs the heads
// one after another; batching puts every head's tiles in one grid. 4 launches per call instead
// of 2·heads + 2.
extern "C" int attention_v2(const float* Q, const float* K, const float* V, float* O, float* work,
                            int heads, int N, int d, cudaStream_t s) {
    float* S = work;
    float* Kt = work + (size_t)heads * N * N;
    const long long nd = (long long)N * d, nn = (long long)N * N;
    int err = transpose_heads(K, Kt, heads, N, d, 1.0f / std::sqrt((float)d), s);
    if (!err) err = gemm_v3_batched(Q, Kt, S, N, N, d, heads, nd, nd, nn, s);
    if (!err)
        err = N <= 2048 ? softmax_v2(S, S, heads * N, N, s) : softmax_v3(S, S, heads * N, N, s);
    if (!err) err = gemm_v3_batched(S, V, O, N, d, N, heads, nn, nd, nd, s);
    return err;
}

// ---- transformer block (D4): GPT-2-style pre-LN block, FP32, no biases
//
//   h = LN1(x); qkv = h·Wqkv; q,k,v = split(qkv); o = attention(q,k,v); a = merge(o)
//   x += a·Wo; h = LN2(x); f = GELU(h·W1); x += f·W2
//
// x is the [B·S, D] residual stream, updated in place. Weights are row-major [in, out]: Wqkv is
// D×3D, Wo D×D, W1 D×4D, W2 4D×D. Each step uses the fastest variant at transformer shapes:
// gemm_v3, layernorm_v2, attention_v2. `work` holds transformer_block_workspace_bytes().

namespace {

const char* const kBlockSteps[] = {"ln1",  "qkv", "split", "attn", "merge", "proj",
                                   "res1", "ln2", "fc1",   "gelu", "fc2",   "res2"};
const int kNumBlockSteps = sizeof(kBlockSteps) / sizeof(kBlockSteps[0]);

// Elementwise launches: enough blocks to cover the GPU a few times over, grid-stride beyond that.
int elementwise_grid(long long n) { return (int)std::min<long long>((n + 255) / 256, 4096); }

}  // namespace

extern "C" int transformer_block_num_steps() { return kNumBlockSteps; }

extern "C" const char* transformer_block_step_name(int step) {
    return step >= 0 && step < kNumBlockSteps ? kBlockSteps[step] : nullptr;
}

extern "C" size_t transformer_block_workspace_bytes(int B, int S, int D, int H) {
    const size_t T = (size_t)B * S;
    // h, qkv (3), q, k, v, o, a, p (6), f (4): 14·T·D floats, plus attention's scores and Kᵀ.
    return sizeof(float) * 14 * T * D + attention_workspace_bytes(B * H, S, D / H);
}

extern "C" int transformer_block_step(int step, float* x, const float* ln1_g, const float* ln1_b,
                                      const float* Wqkv, const float* Wo, const float* ln2_g,
                                      const float* ln2_b, const float* W1, const float* W2,
                                      float* work, int B, int S, int D, int H, cudaStream_t s) {
    const int T = B * S, F = 4 * D, d = D / H;
    const long long TD = (long long)T * D;
    float* h = work;
    float* qkv = h + TD;
    float* q = qkv + 3 * TD;
    float* k = q + TD;
    float* v = k + TD;
    float* o = v + TD;
    float* a = o + TD;
    float* p = a + TD;
    float* f = p + TD;
    float* attn = f + 4 * TD;
    const float eps = 1e-5f;
    switch (step) {
        case 0: return layernorm_v2(x, ln1_g, ln1_b, h, T, D, eps, s);
        case 1: return gemm_v3(h, Wqkv, qkv, T, 3 * D, D, s);
        case 2:
            split_heads_kernel<<<dim3((3 * D + 255) / 256, T), 256, 0, s>>>(qkv, q, k, v, B, S, H, d);
            return cudaGetLastError();
        case 3: return attention_v2(q, k, v, o, attn, B * H, S, d, s);
        case 4:
            merge_heads_kernel<<<dim3((D + 255) / 256, T), 256, 0, s>>>(o, a, B, S, H, d);
            return cudaGetLastError();
        case 5: return gemm_v3(a, Wo, p, T, D, D, s);
        case 6:
            add_inplace_kernel<<<elementwise_grid(TD), 256, 0, s>>>(x, p, TD);
            return cudaGetLastError();
        case 7: return layernorm_v2(x, ln2_g, ln2_b, h, T, D, eps, s);
        case 8: return gemm_v3(h, W1, f, T, F, D, s);
        case 9:
            gelu_inplace_kernel<<<elementwise_grid((long long)T * F), 256, 0, s>>>(f, (long long)T * F);
            return cudaGetLastError();
        case 10: return gemm_v3(f, W2, p, T, D, F, s);
        case 11:
            add_inplace_kernel<<<elementwise_grid(TD), 256, 0, s>>>(x, p, TD);
            return cudaGetLastError();
        default: return cudaErrorInvalidValue;
    }
}

extern "C" int transformer_block(float* x, const float* ln1_g, const float* ln1_b,
                                 const float* Wqkv, const float* Wo, const float* ln2_g,
                                 const float* ln2_b, const float* W1, const float* W2, float* work,
                                 int B, int S, int D, int H, cudaStream_t s) {
    for (int step = 0; step < kNumBlockSteps; ++step) {
        const int err = transformer_block_step(step, x, ln1_g, ln1_b, Wqkv, Wo, ln2_g, ln2_b, W1,
                                               W2, work, B, S, D, H, s);
        if (err) return err;
    }
    return cudaSuccess;
}
