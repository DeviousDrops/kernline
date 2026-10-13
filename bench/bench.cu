// bench <kernel> <variant> <dims...>  ->  one CSV row in the frozen schema (docs/DESIGN.md).
// bench --header prints the header line. dram_bytes and achieved_occ are left empty here:
// they come from Nsight Compute, not from a timing run. Clocks are the median NVML reading
// taken while each timed kernel is in flight.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "common/cuda_check.h"
#include "common/gpu_info.h"
#include "common/timer.h"
#include "src/api.h"

static const int kWarmup = 3;
static const float kWarmupMs = 500.0f;
static const float kBatchMs = 5.0f;  // minimum length of one timed run
static const int kRuns = 21;  // median of >= 20

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

static int usage() {
    std::fprintf(stderr, "usage: bench --header | bench gemm <variant> <M> <N> <K>\n  gemm variants:");
    for (const auto& e : kGemm) std::fprintf(stderr, " %s", e.variant);
    std::fprintf(stderr, "\n");
    return 2;
}

int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--header")) {
        std::puts("gpu,sm_clk_mhz,mem_clk_mhz,kernel,variant,dims,time_ms,flop,bytes_theory,"
                  "dram_bytes,achieved_occ,notes");
        return 0;
    }
    if (argc != 6 || std::strcmp(argv[1], "gemm")) return usage();

    GemmFn fn = nullptr;
    bool fp16_in = false;
    for (const auto& e : kGemm)
        if (!std::strcmp(argv[2], e.variant)) fn = e.fn, fp16_in = e.fp16_in;
    const int M = std::atoi(argv[3]), N = std::atoi(argv[4]), K = std::atoi(argv[5]);
    if (!fn || M <= 0 || N <= 0 || K <= 0) return usage();

    const size_t m = M, n = N, k = K;
    float* A = device_random(m * k, fp16_in);
    float* B = device_random(k * n, fp16_in);
    float* C = device_random(m * n);

    GpuTimer timer;
    ClockSampler clocks;
    // Warm up for at least kWarmup launches *and* kWarmupMs: an idle laptop GPU needs a few
    // hundred ms to reach its boost clock, and NVML (notably under WSL2) reports the change late.
    float warm_ms = 0.0f;
    int warm_launches = 0;
    for (; warm_launches < kWarmup || warm_ms < kWarmupMs; ++warm_launches) {
        timer.start();
        CUDA_CHECK(fn(A, B, C, M, N, K, 0));
        warm_ms += timer.stop_ms();
    }
    // Each timed run is a batch of back-to-back launches lasting >= kBatchMs. Timing a sub-ms
    // kernel alone leaves the GPU idle between syncs (clocks sag) and puts launch/WSL jitter
    // into the measurement; the batch keeps the GPU busy and amortizes both.
    const int batch = std::max(1, (int)std::ceil(kBatchMs / (warm_ms / warm_launches)));
    std::vector<float> ms(kRuns);
    for (int i = 0; i < kRuns; ++i) {
        timer.start();
        for (int b = 0; b < batch; ++b) CUDA_CHECK(fn(A, B, C, M, N, K, 0));
        clocks.sample();  // kernels still in flight: reads the clock they run at
        ms[i] = timer.stop_ms() / batch;
    }
    std::sort(ms.begin(), ms.end());

    const GpuInfo gpu = gpu_info(clocks);
    const double flop = 2.0 * m * n * k;                          // one mul + one add per term
    // read A and B once, write C once
    const double bytes = (fp16_in ? 2.0 : 4.0) * (m * k + k * n) + 4.0 * (m * n);
    std::printf("%s,%d,%d,gemm,%s,%dx%dx%d,%.4f,%.0f,%.0f,,,\n", gpu.name.c_str(),
                gpu.sm_clk_mhz, gpu.mem_clk_mhz, argv[2], M, N, K, ms[kRuns / 2], flop, bytes);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
