#pragma once
// Generators: one per kernel launch, replaying that kernel's global memory instructions from
// its index math (kernels/*/). Addresses are bytes; strides are in floats, as in src/api.cu.
#include <cstdint>
#include <memory>
#include "sim/sim.h"

namespace sim {

// gemm_v3's tile. The real kernel is the default; others exist only in the simulator.
// threads = (BM / TM) * (BN / 8): each thread owns TM rows x 8 columns of C.
struct V3Tile {
    int BM = 128, BN = 128, BK = 8, TM = 8;
};

std::unique_ptr<Generator> gemm_v0(uint64_t A, uint64_t B, uint64_t C, int M, int N, int K);
std::unique_ptr<Generator> gemm_v2(uint64_t A, uint64_t B, uint64_t C, int M, int N, int K,
                                   int T = 32);
std::unique_ptr<Generator> gemm_v3(uint64_t A, uint64_t B, uint64_t C, int M, int N, int K,
                                   V3Tile t = {}, int batch = 1, int64_t strideA = 0,
                                   int64_t strideB = 0, int64_t strideC = 0);

}  // namespace sim
