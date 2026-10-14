#pragma once
// Timing method shared by bench/bench.cu and block/block.cu (docs/DESIGN.md, "Timing method"),
// and the CSV row both print.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>
#include "common/cuda_check.h"
#include "common/gpu_info.h"
#include "common/timer.h"

inline const int kWarmup = 3;
inline const float kWarmupMs = 500.0f;
inline const float kBatchMs = 5.0f;  // minimum length of one timed run
inline const int kRuns = 21;         // median of >= 20
inline const char* kProfileRange = "kl_profile";  // see time_launch and tools/profile.sh

// n random values in [-0.5, 0.5) on the device, stored as FP32 or (fp16) as halves.
inline float* device_random(size_t n, bool fp16 = false) {
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
inline Timing time_launch(const std::function<int()>& launch) {
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
    // One complete call inside an NVTX range, after the warmup (caches and clocks are hot).
    // tools/profile.sh profiles exactly the kernels in this range, however many one call launches
    // (attention launches 2·heads + 2 kernels per call).
    nvtxRangePushA(kProfileRange);
    CUDA_CHECK(launch());
    nvtxRangePop();
    CUDA_CHECK(cudaDeviceSynchronize());
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

inline void print_row(const Timing& t, const char* kernel, const char* variant,
                      const std::string& dims, double flop, double bytes) {
    std::printf("%s,%d,%d,%s,%s,%s,%.4f,%.0f,%.0f,,,\n", t.gpu.name.c_str(), t.gpu.sm_clk_mhz,
                t.gpu.mem_clk_mhz, kernel, variant, dims.c_str(), t.median_ms, flop, bytes);
}
