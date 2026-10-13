# KernLine — Part B: memory side + cache simulator (RTX 4060 Laptop)

**Read `docs/DESIGN.md` first.**

## Your role
You own softmax, LayerNorm, attention (unfused → FlashAttention-style), the shared test/profiling tooling, and the cache simulator. Part A (RTX 4070) owns GEMM and the performance model. Your 4060 Laptop measurements are the model's held-out test set (D1), so never hand-tune the model against them.

## Weekly tasks
| Week | Tasks |
|---|---|
| 1 | Day-1 environment setup. Repo + docs + README. `tools/smoke/` go/no-go. `tests/` ctypes+pytest harness, `tools/profile.sh`, `tools/ceilings/` (stream copy, FMA, smem BW, mma, pointer-chase latency) → `specs/rtx4060_laptop.json`. `analysis/roofline.py` skeleton. |
| 2 | softmax v0 (row per thread, 3 passes), LayerNorm v0 (naive two-pass mean/var), attention v0 unfused (A's GEMM v0 → softmax v0 → GEMM, materializes N×N). Tests vs torch. |
| 3 | softmax v1 / LN v1: row per warp/block, `__shfl_xor_sync` reductions, float4 loads. |
| 4 | softmax v2 online single-pass (running max + rescaled sum), LN v2 Welford single-pass. Target ≥ 80% of measured DRAM BW. |
| 5 | attention v1 FlashAttention-style: Q tile in smem, stream K/V tiles, online softmax, O accumulated in registers, never write N×N. Compare vs `F.scaled_dot_product_attention`. Show the DRAM-bytes drop in ncu. |
| 6 | Model descriptors for softmax/LN/attention. Run the full bench + ncu suite on the 4060 Laptop, including A's GEMMs (D8). |
| 7 | Hand the held-out results to A. Write the "why attention is memory-bound" analysis (arithmetic intensity vs sequence length, crossover point on each GPU). |
| 8 | `sim/` (C++): address-replay trace generator (D5) + set-associative L1 per SM (128 B lines, 32 B sectors, LRU) + shared L2 (32 MB on 4060 Laptop, 36 MB on 4070). Hit rate vs tile size for GEMM v0/v2/v3 and attention v0/v1. Cross-check with ncu hit rates and report the gap. |

## Definition of done per step
Correctness test passes → CSV row in `results/rtx4060_laptop/` → ncu report → one-sentence roofline explanation in the PR.
