// bench <kernel> <variant> <dims...>  ->  one CSV row in the frozen schema (docs/DESIGN.md).
// bench --header prints the header line. dram_bytes and achieved_occ are left empty here:
// they come from Nsight Compute, not from a timing run. Clocks are the median NVML reading
// taken while each timed kernel is in flight.
//
//   bench gemm      <variant> <M> <N> <K>
//   bench softmax   <variant> <rows> <cols>
//   bench layernorm <variant> <rows> <cols>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "common/cuda_check.h"
#include "common/gpu_info.h"
#include "common/timer.h"
#include "src/api.h"

static const int kWarmup = 3;
static const float kWarmupMs = 500.0f;
static const float kBatchMs = 5.0f;  // minimum length of one timed run
static const int kRuns = 21;         // median of >= 20

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
};

typedef int (*LayerNormFn)(const float*, const float*, const float*, float*, int, int, float,
                           cudaStream_t);
static const struct { const char* variant; LayerNormFn fn; } kLayerNorm[] = {
    {"v0", layernorm_v0},
    {"v1", layernorm_v1},
};

template <class Table>
static auto find(const Table& table, const char* variant) -> decltype(table[0].fn) {
    for (const auto& e : table)
        if (!std::strcmp(variant, e.variant)) return e.fn;
    return nullptr;
}

// n random values in [-0.5, 0.5) on the device, stored as FP32 or (fp16) as halves.
static float* device_random(size_t n, bool fp16 = false) {
    std::vector<float> host(n);
    for (float& x : host) x = (float)std::rand() / RAND_MAX - 0.5f;
    std::vector<__half> half;
    if (fp16)
        for (float x : host) half.push_back(__float2half(x));
    const void* src = fp16 ? (const void*)half.data() : host.data();
    const size_t bytes = n * (fp16 ? sizeof(__half) : sizeof(float));
    float* dev;
    CUDA_CHECK(cudaMalloc(&dev, bytes));
    CUDA_CHECK(cudaMemcpy(dev, src, bytes, cudaMemcpyHostToDevice));
    return dev;
}

struct Timing {
    float median_ms;
    GpuInfo gpu;
};

// Median time of one launch. launch() must enqueue exactly one call of the launcher on stream 0
// and return its cudaError_t.
static Timing time_launch(const std::function<int()>& launch) {
    GpuTimer timer;
    ClockSampler clocks;
    // Warm up for at least kWarmup launches *and* kWarmupMs: an idle laptop GPU needs a few
    // hundred ms to reach its boost clock, and NVML (notably under WSL2) reports the change late.
    float warm_ms = 0.0f;
    int warm_launches = 0;
    for (; warm_launches < kWarmup || warm_ms < kWarmupMs; ++warm_launches) {
        timer.start();
        CUDA_CHECK(launch());
        warm_ms += timer.stop_ms();
    }
    // Each timed run is a batch of back-to-back launches lasting >= kBatchMs. Timing a sub-ms
    // kernel alone leaves the GPU idle between syncs (clocks sag) and puts launch/WSL jitter
    // into the measurement; the batch keeps the GPU busy and amortizes both.
    const int batch = std::max(1, (int)std::ceil(kBatchMs / (warm_ms / warm_launches)));
    std::vector<float> ms(kRuns);
    for (int i = 0; i < kRuns; ++i) {
        timer.start();
        for (int b = 0; b < batch; ++b) CUDA_CHECK(launch());
        clocks.sample();  // kernels still in flight: reads the clock they run at
        ms[i] = timer.stop_ms() / batch;
    }
    std::sort(ms.begin(), ms.end());
    return {ms[kRuns / 2], gpu_info(clocks)};
}

static void print_row(const Timing& t, const char* kernel, const char* variant,
                      const std::string& dims, double flop, double bytes) {
    std::printf("%s,%d,%d,%s,%s,%s,%.4f,%.0f,%.0f,,,\n", t.gpu.name.c_str(), t.gpu.sm_clk_mhz,
                t.gpu.mem_clk_mhz, kernel, variant, dims.c_str(), t.median_ms, flop, bytes);
}

static int usage() {
    std::fprintf(stderr, "usage: bench --header\n");
    std::fprintf(stderr, "       bench gemm <variant> <M> <N> <K>        variants:");
    for (const auto& e : kGemm) std::fprintf(stderr, " %s", e.variant);
    std::fprintf(stderr, "\n       bench softmax <variant> <rows> <cols>   variants:");
    for (const auto& e : kSoftmax) std::fprintf(stderr, " %s", e.variant);
    std::fprintf(stderr, "\n       bench layernorm <variant> <rows> <cols> variants:");
    for (const auto& e : kLayerNorm) std::fprintf(stderr, " %s", e.variant);
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
    return usage();
}
