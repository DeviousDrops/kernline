#pragma once
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#include <nvml.h>
#include "common/cuda_check.h"

// GPU identity for the CSV schema (D7): slug such as "rtx4070_laptop" plus clocks.
struct GpuInfo {
    std::string name;
    int sm_clk_mhz = 0;
    int mem_clk_mhz = 0;
};

// "NVIDIA GeForce RTX 4070 Laptop GPU" -> "rtx4070_laptop"
inline std::string gpu_slug(std::string s) {
    for (const char* drop : {"NVIDIA ", "GeForce ", " GPU"}) {
        size_t at = s.find(drop);
        if (at != std::string::npos) s.erase(at, std::char_traits<char>::length(drop));
    }
    size_t rtx = s.find("RTX ");
    if (rtx != std::string::npos) s.erase(rtx + 3, 1);
    for (char& c : s) c = c == ' ' ? '_' : (char)std::tolower((unsigned char)c);
    return s;
}

// Samples SM/memory clocks through NVML. Call sample() while a kernel is in flight (after the
// launch, before synchronizing): laptop GPUs drop to idle clocks within milliseconds, so a query
// after the runs reads the idle state, not the clock the kernel ran at. Clocks stay 0 when NVML
// is unavailable.
class ClockSampler {
  public:
    explicit ClockSampler(int device = 0) {
        char bus_id[32];
        ok_ = cudaDeviceGetPCIBusId(bus_id, sizeof bus_id, device) == cudaSuccess &&
              nvmlInit_v2() == NVML_SUCCESS &&
              nvmlDeviceGetHandleByPciBusId_v2(bus_id, &dev_) == NVML_SUCCESS;
    }
    ~ClockSampler() {
        if (ok_) nvmlShutdown();
    }
    void sample() {
        unsigned sm = 0, mem = 0;
        if (ok_ && nvmlDeviceGetClockInfo(dev_, NVML_CLOCK_SM, &sm) == NVML_SUCCESS &&
            nvmlDeviceGetClockInfo(dev_, NVML_CLOCK_MEM, &mem) == NVML_SUCCESS) {
            sm_.push_back(sm);
            mem_.push_back(mem);
        }
    }
    // True once the memory clock is at its maximum (or NVML is unavailable). After an idle gap a
    // laptop GPU can sit at a lower memory clock for over a second while the SM clock is already
    // boosted, which makes memory-bound kernels up to ~35% slower.
    bool mem_at_max() const {
        unsigned cur = 0, max = 0;
        if (!ok_ || nvmlDeviceGetClockInfo(dev_, NVML_CLOCK_MEM, &cur) != NVML_SUCCESS ||
            nvmlDeviceGetMaxClockInfo(dev_, NVML_CLOCK_MEM, &max) != NVML_SUCCESS)
            return true;
        return cur >= max;
    }
    int median_sm() const { return median(sm_); }
    int median_mem() const { return median(mem_); }

  private:
    static int median(std::vector<unsigned> v) {
        if (v.empty()) return 0;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return (int)v[v.size() / 2];
    }
    bool ok_ = false;
    nvmlDevice_t dev_{};
    std::vector<unsigned> sm_, mem_;
};

inline GpuInfo gpu_info(const ClockSampler& clocks, int device = 0) {
    GpuInfo info;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    info.name = gpu_slug(prop.name);
    info.sm_clk_mhz = clocks.median_sm();
    info.mem_clk_mhz = clocks.median_mem();
    return info;
}
