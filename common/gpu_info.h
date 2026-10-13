#pragma once
#include <cctype>
#include <cstdio>
#include <string>
#include "common/cuda_check.h"

// GPU identity for the CSV schema (D7): slug such as "rtx4070_laptop" plus current clocks.
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

// Clocks come from nvidia-smi and stay 0 when it cannot report them. Call this right after
// the timed runs so the clocks reflect the loaded state.
inline GpuInfo gpu_info(int device = 0) {
    GpuInfo info;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    info.name = gpu_slug(prop.name);

    std::string cmd = "nvidia-smi -i " + std::to_string(device) +
                      " --query-gpu=clocks.current.sm,clocks.current.memory"
                      " --format=csv,noheader,nounits 2>/dev/null";
    if (FILE* p = popen(cmd.c_str(), "r")) {
        if (std::fscanf(p, "%d , %d", &info.sm_clk_mhz, &info.mem_clk_mhz) != 2)
            info.sm_clk_mhz = info.mem_clk_mhz = 0;
        pclose(p);
    }
    return info;
}
