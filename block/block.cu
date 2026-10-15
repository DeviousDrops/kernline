// block <B> <S> [D=768] [H=12]: time a GPT-2-style transformer block (docs/DESIGN.md, D4).
//
// Prints one CSV row (frozen schema) per step, each timed on its own with the bench method
// (common/bench_util.h), then a "total" row for all steps back to back. kernel is "block",
// variant the step name, dims BxSxDxH. The model's block prediction (python -m model block) is
// the sum of its per-step predictions, compared against both the steps and the total.
//
// flop / bytes_theory per step, T = B·S tokens, F = 4·D:
//   ln:     8·T·D flop (as bench layernorm);    read x + write y + gamma/beta
//   GEMMs:  2·M·N·K;                            read A and B once, write C once
//   attn:   as bench attention over B·H heads of S×d
//   split/merge/res/gelu: read and write each element once (res also reads p)
#include <cstdlib>
#include <cstring>
#include <string>
#include "common/bench_util.h"
#include "src/api.h"

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: block <B> <S> [D=768] [H=12]\n");
        return 2;
    }
    const int B = std::atoi(argv[1]), S = std::atoi(argv[2]);
    const int D = argc > 3 ? std::atoi(argv[3]) : 768, H = argc > 4 ? std::atoi(argv[4]) : 12;
    if (B <= 0 || S <= 0 || D <= 0 || H <= 0 || D % H) {
        std::fprintf(stderr, "B, S, D, H must be positive and H must divide D\n");
        return 2;
    }
    const size_t T = (size_t)B * S, F = 4 * (size_t)D, d = D / H;
    float* x = device_random(T * D);
    float* ln1_g = device_random(D);
    float* ln1_b = device_random(D);
    float* Wqkv = device_random((size_t)D * 3 * D);
    float* Wo = device_random((size_t)D * D);
    float* ln2_g = device_random(D);
    float* ln2_b = device_random(D);
    float* W1 = device_random((size_t)D * F);
    float* W2 = device_random(F * D);
    float* work;
    CUDA_CHECK(cudaMalloc(&work, transformer_block_workspace_bytes(B, S, D, H)));

    const double TD = (double)T * D, TF = (double)T * F, nn = (double)B * H * S * S;
    // {flop, bytes} per step, in transformer_block_step order.
    const double cost[][2] = {
        {8 * TD, 8 * TD + 8.0 * D},                                      // ln1
        {2 * TD * 3 * D, 4 * (TD + 3.0 * D * D + 3 * TD)},              // qkv
        {0, 4 * 6 * TD},                                                 // split
        {4 * nn * d + 5 * nn, 4 * (4 * TD + 4 * nn)},                    // attn
        {0, 4 * 2 * TD},                                                 // merge
        {2 * TD * D, 4 * (2 * TD + (double)D * D)},                      // proj
        {TD, 4 * 3 * TD},                                                // res1
        {8 * TD, 8 * TD + 8.0 * D},                                      // ln2
        {2 * TD * F, 4 * (TD + (double)D * F + TF)},                     // fc1
        {8 * TF, 4 * 2 * TF},                                            // gelu
        {2 * TF * D, 4 * (TF + (double)F * D + TD)},                     // fc2
        {TD, 4 * 3 * TD},                                                // res2
    };
    const int steps = transformer_block_num_steps();
    const std::string dims = std::to_string(B) + "x" + std::to_string(S) + "x" +
                             std::to_string(D) + "x" + std::to_string(H);
    double flop = 0, bytes = 0;
    for (int i = 0; i < steps; ++i) {
        Timing t = time_launch([&] {
            return transformer_block_step(i, x, ln1_g, ln1_b, Wqkv, Wo, ln2_g, ln2_b, W1, W2, work,
                                          B, S, D, H, 0);
        });
        print_row(t, "block", transformer_block_step_name(i), dims, cost[i][0], cost[i][1]);
        flop += cost[i][0];
        bytes += cost[i][1];
    }
    Timing t = time_launch([&] {
        return transformer_block(x, ln1_g, ln1_b, Wqkv, Wo, ln2_g, ln2_b, W1, W2, work, B, S, D, H, 0);
    });
    print_row(t, "block", "total", dims, flop, bytes);

    for (float* p : {x, ln1_g, ln1_b, Wqkv, Wo, ln2_g, ln2_b, W1, W2, work}) cudaFree(p);
    return 0;
}
