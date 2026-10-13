// bench <kernel> <variant> <dims...>  ->  one CSV row in the frozen schema (docs/DESIGN.md).
// bench --header prints the header line. dram_bytes and achieved_occ are left empty here:
// they come from Nsight Compute, not from a timing run.
#include <algorithm>
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
static const int kRuns = 21;  // median of >= 20

typedef int (*GemmFn)(const float*, const float*, float*, int, int, int, cudaStream_t);
static const struct { const char* variant; GemmFn fn; } kGemm[] = {
    {"v0", gemm_v0},
    {"cublas", gemm_cublas},
};

static float* device_random(size_t n) {
    std::vector<float> host(n);
    for (float& x : host) x = (float)std::rand() / RAND_MAX - 0.5f;
    float* dev;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
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
    for (const auto& e : kGemm)
        if (!std::strcmp(argv[2], e.variant)) fn = e.fn;
    const int M = std::atoi(argv[3]), N = std::atoi(argv[4]), K = std::atoi(argv[5]);
    if (!fn || M <= 0 || N <= 0 || K <= 0) return usage();

    const size_t m = M, n = N, k = K;
    float* A = device_random(m * k);
    float* B = device_random(k * n);
    float* C = device_random(m * n);

    GpuTimer timer;
    std::vector<float> ms(kRuns);
    for (int i = -kWarmup; i < kRuns; ++i) {
        timer.start();
        CUDA_CHECK(fn(A, B, C, M, N, K, 0));
        float t = timer.stop_ms();
        if (i >= 0) ms[i] = t;
    }
    std::sort(ms.begin(), ms.end());

    const GpuInfo gpu = gpu_info();
    const double flop = 2.0 * m * n * k;                          // one mul + one add per term
    const double bytes = 4.0 * (m * k + k * n + m * n);           // read A and B once, write C once
    std::printf("%s,%d,%d,gemm,%s,%dx%dx%d,%.4f,%.0f,%.0f,,,\n", gpu.name.c_str(),
                gpu.sm_clk_mhz, gpu.mem_clk_mhz, argv[2], M, N, K, ms[kRuns / 2], flop, bytes);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
