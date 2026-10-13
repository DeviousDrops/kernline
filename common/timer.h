#pragma once
#include "common/cuda_check.h"

// GPU-side wall time between start() and stop_ms(), measured with cudaEvents on one stream.
struct GpuTimer {
    cudaEvent_t t0, t1;
    cudaStream_t stream;

    explicit GpuTimer(cudaStream_t s = 0) : stream(s) {
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
    }
    ~GpuTimer() {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    void start() { CUDA_CHECK(cudaEventRecord(t0, stream)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        return ms;
    }
};
