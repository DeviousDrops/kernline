// bench <kernel> <variant> <dims...>  ->  one CSV row in the frozen schema (docs/DESIGN.md).
// bench --header prints the header line. dram_bytes and achieved_occ are left empty here:
// they come from Nsight Compute, not from a timing run. Clocks are the median NVML reading
// taken while each timed kernel is in flight.
//
//   bench gemm      <variant> <M> <N> <K>
//   bench softmax   <variant> <rows> <cols>
//   bench layernorm <variant> <rows> <cols>
//   bench attention <variant> <heads> <N> <d>
#include <cstdlib>
#include <cstring>
#include <string>
#include "common/bench_util.h"
#include "src/api.h"

typedef int (*GemmFn)(const float*, const float*, float*, int, int, int, cudaStream_t);
// v4 takes FP16 A and B: same buffers, filled as halves (see device_random).
static int gemm_v4_fp16(const float* A, const float* B, float* C, int M, int N, int K,
                        cudaStream_t s) {
    return gemm_v4((const __half*)A, (const __half*)B, C, M, N, K, s);
}
static const struct { const char* variant; GemmFn fn; bool fp16_in = false; } kGemm[] = {
    {"v0", gemm_v0},
    {"v1", gemm_v1},
    {"v2", gemm_v2},
    {"v3", gemm_v3},
    {"v4", gemm_v4_fp16, true},
    {"cublas", gemm_cublas},
};

typedef int (*SoftmaxFn)(const float*, float*, int, int, cudaStream_t);
static const struct { const char* variant; SoftmaxFn fn; } kSoftmax[] = {
    {"v0", softmax_v0},
    {"v1", softmax_v1},
    {"v2", softmax_v2},
    {"v3", softmax_v3},
};

typedef int (*LayerNormFn)(const float*, const float*, const float*, float*, int, int, float,
                           cudaStream_t);
static const struct { const char* variant; LayerNormFn fn; } kLayerNorm[] = {
    {"v0", layernorm_v0},
    {"v1", layernorm_v1},
    {"v2", layernorm_v2},
    {"v3", layernorm_v3},
};

typedef int (*AttentionFn)(const float*, const float*, const float*, float*, float*, int, int, int,
                           cudaStream_t);
static const struct { const char* variant; AttentionFn fn; } kAttention[] = {
    {"v0", attention_v0},
    {"v1", attention_v1},
    {"v2", attention_v2},
};

template <class Table>
static auto find(const Table& table, const char* variant) -> decltype(table[0].fn) {
    for (const auto& e : table)
        if (!std::strcmp(variant, e.variant)) return e.fn;
    return nullptr;
}

static int usage() {
    std::fprintf(stderr, "usage: bench --header\n");
    std::fprintf(stderr, "       bench gemm <variant> <M> <N> <K>        variants:");
    for (const auto& e : kGemm) std::fprintf(stderr, " %s", e.variant);
    std::fprintf(stderr, "\n       bench softmax <variant> <rows> <cols>   variants:");
    for (const auto& e : kSoftmax) std::fprintf(stderr, " %s", e.variant);
    std::fprintf(stderr, "\n       bench layernorm <variant> <rows> <cols> variants:");
    for (const auto& e : kLayerNorm) std::fprintf(stderr, " %s", e.variant);
    std::fprintf(stderr, "\n       bench attention <variant> <heads> <N> <d> variants:");
    for (const auto& e : kAttention) std::fprintf(stderr, " %s", e.variant);
    std::fprintf(stderr, "\n");
    return 2;
}

static int bench_gemm(const char* variant, int M, int N, int K) {
    GemmFn fn = nullptr;
    bool fp16_in = false;
    for (const auto& e : kGemm)
        if (!std::strcmp(variant, e.variant)) fn = e.fn, fp16_in = e.fp16_in;
    if (!fn || M <= 0 || N <= 0 || K <= 0) return usage();
    const size_t m = M, n = N, k = K;
    float* A = device_random(m * k, fp16_in);
    float* B = device_random(k * n, fp16_in);
    float* C = device_random(m * n);
    Timing t = time_launch([&] { return fn(A, B, C, M, N, K, 0); });
    const double flop = 2.0 * m * n * k;  // one mul + one add per term
    // read A and B once, write C once
    const double bytes = (fp16_in ? 2.0 : 4.0) * (m * k + k * n) + 4.0 * (m * n);
    print_row(t, "gemm", variant, std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K),
              flop, bytes);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}

// Row-wise kernels over a rows×cols matrix (softmax and LayerNorm along the last dim).
// bytes_theory is the floor for any implementation: read x once, write y once (+ gamma/beta).
// flop counts per element are a convention, stated so the roofline is reproducible:
//   softmax:   max-compare, subtract, exp, add, multiply                    = 5
//   layernorm: add (mean), subtract + FMA (variance), subtract, multiply, FMA (gamma/beta) = 8
static int bench_rowwise(const char* kernel, const char* variant, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return usage();
    const size_t elems = (size_t)rows * cols;
    float* x = device_random(elems);
    float* y = device_random(elems);
    Timing t;
    double flop, bytes;
    if (!std::strcmp(kernel, "softmax")) {
        SoftmaxFn fn = find(kSoftmax, variant);
        if (!fn) return usage();
        t = time_launch([&] { return fn(x, y, rows, cols, 0); });
        flop = 5.0 * elems;
        bytes = 8.0 * elems;
    } else {
        LayerNormFn fn = find(kLayerNorm, variant);
        if (!fn) return usage();
        float* gamma = device_random(cols);
        float* beta = device_random(cols);
        t = time_launch([&] { return fn(x, gamma, beta, y, rows, cols, 1e-5f, 0); });
        flop = 8.0 * elems;
        bytes = 8.0 * elems + 8.0 * cols;
        cudaFree(gamma);
        cudaFree(beta);
    }
    print_row(t, kernel, variant, std::to_string(rows) + "x" + std::to_string(cols), flop, bytes);
    cudaFree(x);
    cudaFree(y);
    return 0;
}

// Unfused attention over `heads` (= batch × heads) heads of N×d. dims are written headsxNxd.
//   flop:  2·N²·d (Q·Kᵀ) + 5·N² (softmax, as above) + 2·N²·d (P·V) per head
//   bytes: the floor for an *unfused* implementation, whatever its kernels: read Q, K, V and
//          write O once (4·N·d floats), and move the N×N scores four times: Q·Kᵀ writes S,
//          softmax reads S and writes P, P·V reads P. So arithmetic intensity is bounded by
//          ~(4·N²·d) / (16·N²) = d/4 FLOP/byte, independent of N: 16 for GPT-2's d = 64.
static int bench_attention(const char* variant, int heads, int N, int d) {
    AttentionFn fn = find(kAttention, variant);
    if (!fn || heads <= 0 || N <= 0 || d <= 0) return usage();
    const size_t qkv = (size_t)heads * N * d;
    float* Q = device_random(qkv);
    float* K = device_random(qkv);
    float* V = device_random(qkv);
    float* O = device_random(qkv);
    float* work;
    CUDA_CHECK(cudaMalloc(&work, attention_workspace_bytes(heads, N, d)));
    Timing t = time_launch([&] { return fn(Q, K, V, O, work, heads, N, d, 0); });
    const double nn = (double)heads * N * N;
    const double flop = 4.0 * nn * d + 5.0 * nn;
    const double bytes = 4.0 * (4.0 * qkv + 4.0 * nn);
    print_row(t, "attention", variant,
              std::to_string(heads) + "x" + std::to_string(N) + "x" + std::to_string(d), flop, bytes);
    for (float* p : {Q, K, V, O, work}) cudaFree(p);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--header")) {
        std::puts("gpu,sm_clk_mhz,mem_clk_mhz,kernel,variant,dims,time_ms,flop,bytes_theory,"
                  "dram_bytes,achieved_occ,notes");
        return 0;
    }
    if (argc == 6 && !std::strcmp(argv[1], "gemm"))
        return bench_gemm(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));
    if (argc == 5 && (!std::strcmp(argv[1], "softmax") || !std::strcmp(argv[1], "layernorm")))
        return bench_rowwise(argv[1], argv[2], std::atoi(argv[3]), std::atoi(argv[4]));
    if (argc == 6 && !std::strcmp(argv[1], "attention"))
        return bench_attention(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));
    return usage();
}
