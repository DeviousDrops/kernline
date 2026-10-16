#pragma once
// Generators: one per kernel launch, replaying that kernel's global memory instructions from
// its index math (kernels/*/). Addresses are bytes; strides are in floats, as in src/api.cu.
#include <cstdint>
#include <memory>
#include <string>
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

// kernels/attention/helpers.cu: out[h] = in[h]^T * scale, in is rows x cols per head.
std::unique_ptr<Generator> transpose_scale(uint64_t in, uint64_t out, int heads, int rows, int cols);
// x[i] *= s for i < n: grid 1024 x 256, grid-stride.
std::unique_ptr<Generator> scale(uint64_t x, int64_t n);
// kernels/softmax: v0 (thread per row), v2 row cached in registers (k float4 per lane, float4
// path), v3 row in shared memory (float4 path).
std::unique_ptr<Generator> softmax_v0(uint64_t x, uint64_t y, int64_t rows, int64_t cols);
std::unique_ptr<Generator> softmax_v2_cached(uint64_t x, uint64_t y, int64_t rows, int64_t cols, int k);
std::unique_ptr<Generator> softmax_v3(uint64_t x, uint64_t y, int64_t rows, int64_t cols);
// src/api.cu attention_v0/v1/v2: buffers Q, K, V, O, work allocated in the bench's order, and
// the launches of one call in launch order.
Launches attention(const std::string& variant, int heads, int N, int d, Buffers& buf);

}  // namespace sim
