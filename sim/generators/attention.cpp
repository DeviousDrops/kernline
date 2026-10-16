// Unfused attention calls (src/api.cu attention_v0/v1/v2) as launch lists. Buffers follow
// bench/bench.cu: Q, K, V, O ([heads, N, d] each), then work = S [heads, N, N] + Kt [heads, d, N].
#include <stdexcept>
#include "sim/generators/generators.h"

namespace sim {

// src/api.cu: softmax_v2 (register cache of k float4 per lane) up to N = 2048, v3 above.
static std::unique_ptr<Generator> attention_softmax(uint64_t S, int64_t rows, int64_t N) {
    if (N > 2048) return softmax_v3(S, S, rows, N);
    int k = 1;
    while (N > 128 * k) k *= 2;
    return softmax_v2_cached(S, S, rows, N, k);
}

Launches attention(const std::string& variant, int heads, int N, int d, Buffers& buf) {
    const int64_t nd = int64_t(N) * d, nn = int64_t(N) * N, h_ = heads;
    const uint64_t Q = buf.alloc("Q", 4ull * h_ * nd), K = buf.alloc("K", 4ull * h_ * nd),
                   V = buf.alloc("V", 4ull * h_ * nd), O = buf.alloc("O", 4ull * h_ * nd);
    const uint64_t work = buf.alloc("work", 4ull * (h_ * nn + h_ * nd));
    const uint64_t S = work, Kt = work + 4ull * h_ * nn;
    auto at = [](uint64_t base, int64_t floats) { return base + 4ull * uint64_t(floats); };

    Launches out;
    out.push_back(transpose_scale(K, Kt, heads, N, d));
    if (variant == "v0") {
        for (int h = 0; h < heads; ++h)
            out.push_back(gemm_v0(at(Q, h * nd), at(Kt, h * nd), at(S, h * nn), N, N, d));
        out.push_back(scale(S, h_ * nn));
        out.push_back(softmax_v0(S, S, h_ * N, N));
        for (int h = 0; h < heads; ++h)
            out.push_back(gemm_v0(at(S, h * nn), at(V, h * nd), at(O, h * nd), N, d, N));
    } else if (variant == "v1") {
        for (int h = 0; h < heads; ++h)
            out.push_back(gemm_v3(at(Q, h * nd), at(Kt, h * nd), at(S, h * nn), N, N, d));
        out.push_back(attention_softmax(S, h_ * N, N));
        for (int h = 0; h < heads; ++h)
            out.push_back(gemm_v3(at(S, h * nn), at(V, h * nd), at(O, h * nd), N, d, N));
    } else if (variant == "v2") {
        out.push_back(gemm_v3(Q, Kt, S, N, N, d, V3Tile{}, heads, nd, nd, nn));
        out.push_back(attention_softmax(S, h_ * N, N));
        out.push_back(gemm_v3(S, V, O, N, d, N, V3Tile{}, heads, nn, nd, nd));
    } else {
        throw std::invalid_argument("attention variants: v0 v1 v2");
    }
    return out;
}

}  // namespace sim
