# Why unfused attention is memory-bound

Measured on both laptops: the RTX 4060 Laptop (24 SMs, FP32 14.1 TFLOP/s, DRAM 249 GB/s, ridge point 57 FLOP/byte) and the RTX 4070 Laptop (36 SMs, FP32 20.9 TFLOP/s, DRAM 249 GB/s, ridge 84). GPT-2 small shapes: 12 heads, head size d = 64, sequence length N = 256 … 4096, FP32, non-causal. Every row at memory clock 8001 MHz. Data: `results/<gpu>/attention.csv` from `tools/suite.sh`, per-kernel phases from the ncu reports via `analysis/attention_breakdown.py`.

## The argument in one line

Unfused attention writes the N×N score matrix to DRAM and reads it back. Per head that is 4·N² floats of traffic (Q·Kᵀ writes S, softmax reads S and writes P, P·V reads P) against 4·N²·d FLOPs of GEMM, so its arithmetic intensity is at most **d/4 FLOP/byte, independent of N**. For d = 64 that is 16 FLOP/byte, below the 4060's ridge of 57 and the 4070's of 84. Making N larger does not help: the FLOPs and the N² traffic grow together. Unfused attention stays memory-bound for any d below 4 × ridge (228 on the 4060, 336 on the 4070).

The roofline confirms it: on both GPUs every implementation sits under the DRAM slope at AI ≈ 16 (`analysis/plots/attention.png`), and ncu measures v2 moving the unfused floor of DRAM bytes (0.97–1.00× `bytes_theory`) at N ≥ 2048. The cross-GPU check below shows it more directly: with 1.5× the SMs, the 4070 runs v2 at the same speed as the 4060 for N ≥ 1024.

## Variants

| | how | kernel launches per call |
|---|---|---|
| v0 | Kᵀ, `gemm_v0` per head, separate pass scaling S by 1/√d, `softmax_v0`, `gemm_v0` per head | 2·heads + 3 |
| v1 | Kᵀ written pre-scaled by 1/√d (no pass over S), `gemm_v3` per head, softmax v2 (N ≤ 2048) or v3 | 2·heads + 2 |
| v2 | v1 with each GEMM batched over all heads in one launch (`gemm_v3_batched`, gridDim.z = heads) | 4 |
| torch | `F.scaled_dot_product_attention`, math backend (unfused) | ~6 |

## Results (median ms, bench at boost clocks)

| N | v0 | v1 | v2 | torch | v2 vs torch | DRAM floor | v2 % of floor | v2 on the 4070 | 4060 / 4070 |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 1.155 | 0.490 | **0.062** | 0.086 | 1.37× | 0.063 | L2-resident | 0.052 | 1.20 |
| 512 | 4.332 | 0.856 | **0.187** | 0.250 | 1.34× | 0.227 | L2-resident | 0.156 | 1.20 |
| 1024 | 16.89 | 2.397 | **1.176** | 1.745 | 1.48× | 0.859 | 73% | 1.151 | 1.02 |
| 2048 | 71.19 | 6.266 | **5.218** | 7.113 | 1.36× | 3.337 | 64% | 5.410 | 0.96 |
| 4096 | 274.2 | 22.08 | **20.85** | 28.76 | 1.38× | 13.14 | 63% | 20.95 | 1.00 |

v0–torch are the 4060. "DRAM floor" is `bytes_theory` (the unfused minimum: Q, K, V, O once plus four passes over the scores) at the measured 249 GB/s. At N ≤ 512 the 12 heads' scores (3–13 MB) fit in the 32 MB L2, so the passes over S never reach DRAM and v2 beats the "floor". At N = 1024 (50 MB of scores) part of S still hits in L2, and v2 moves 0.82× the floor.

**Why each step moved:**
- **v0 → v1 (2.4–12×):** the GEMMs. `gemm_v0` is uncoalesced, and the separate scale pass costs one more read and write of S (2% of v0's time; it matters more in torch, below).
- **v1 → v2 (7.9× at N = 256, 1.06× at N = 4096):** occupancy. At N = 256, one head's Q·Kᵀ is 4 tiles of 128×128, so `gemm_v3` per head leaves 20 of 24 SMs idle. P·V (N×64 output) is worse: 2 tiles per head. Batching the heads into one grid fills the GPU. At N = 4096 each head already has 1024 tiles, so batching only removes launch gaps.
- **v2 vs torch (1.34–1.48×):** traffic. ncu measures torch moving 1.86× the unfused floor at N = 4096 (6.09 GB vs 3.27 GB) because the math backend runs separate elementwise kernels (scale, copies; 38% of its kernel time). v2 folds the scale into the Kᵀ prep and moves 1.00×.

## Where v2's time goes (N = 4096, kernel time at ncu's base clocks)

| phase | 4060 time (share) | DRAM traffic | at the DRAM roof | 4060 achieved | 4070 time | 4070 speedup | limited by |
|---|---|---|---|---|---|---|---|
| Kᵀ prep | 0.06 ms (0.3%) | Q-sized | – | – | 0.06 ms | 1.0× | – |
| Q·Kᵀ | 5.14 ms (23%) | writes S: 0.82 GB | 3.3 ms | 159 GB/s, 64% | 4.39 ms | 1.17× | storing S: only 8 k-steps (K = d = 64) per tile, register-limited occupancy (33%) leaves few stores in flight |
| softmax | 7.03 ms (32%) | reads S, writes P: 1.59 GB | 6.4 ms | 227 GB/s, **91%** | 7.34 ms | 0.96× | DRAM bandwidth (softmax v3, one read per score) |
| P·V | 9.97 ms (45%) | reads P: 0.84 GB | 3.4 ms | 85 GB/s, 34% | 11.26 ms | **0.89×** | **how P is read**: `gemm_v3` loads a 128×8 tile of A per k-step, so each k-step reads 32 bytes from each of 128 rows of P that are 16 KB apart |

So v2 runs at 63% of the unfused roofline, and the softmax phase is already at the roof. The remaining gap is in the GEMMs, not in attention's structure.

**P·V is DRAM-bound, by its access pattern.** It reads P once (0.84 GB measured, against 0.81 GB in theory), but at only ~85 GB/s: P is the A operand, and every k-step of `gemm_v3` pulls one 32-byte sector from each of 128 separate rows of P, 4·N bytes apart. Q·Kᵀ uses the same tile on Q, whose rows are only 256 bytes apart, and doesn't have this problem. The 4070 confirms it. It has more blocks in flight (384 blocks at 2 per SM take 5.3 waves on 36 SMs vs 8 on 24) and 1.5× the compute, yet P·V is 11% *slower* there, at 77 GB/s. Its top stall moves to `long_scoreboard` (waiting on global loads, 5.6 warp-cycles per issued instruction vs 1.9 on the 4060). The zero columns in each 128×128 tile (the output is N×64) do waste half the FMAs, but they aren't the limit: the GPU with 1.5× the compute doesn't go faster. The lever is reading P in longer contiguous runs, such as a deeper k-step or a tile shape matched to d = 64 for this GEMM. That belongs to `gemm_v3` (`a/gemm`), so it's noted here rather than worked around.

An earlier version of this write-up called P·V compute-limited ("wasted compute") and predicted the 4070 would run it up to 1.4× faster. The 4070 data in #10 showed otherwise, and the per-kernel ncu numbers above, from both GPUs, agree.

## The 4070 as a held-out check (D1/D8)

Predictions made from the 4060 data alone, before the 4070 suite ran, against what the 4070 measured (N = 4096 phases at base clocks; whole call from the bench):

| | predicted | measured on the 4070 | |
|---|---|---|---|
| softmax phase | same time (DRAM-bound at 91%) | 0.96× (7.34 vs 7.03 ms) | right |
| Q·Kᵀ phase | within ~15% (store-bound) | 1.17× faster | about right |
| P·V phase | up to ~1.4× faster (compute-limited) | 0.89×, slower | **wrong**: DRAM-bound, see above |
| whole call, N ≥ 1024 | 1.1–1.25× faster | 0.96–1.02× | wrong, because of P·V |

The miss runs in one direction: unfused attention v2 is *more* memory-bound than claimed, in all three phases, not less. Only at N ≤ 512, where the scores stay in L2, does the 4070's extra compute show (1.20×). That is the pattern DESIGN.md (D1) expects of memory-bound kernels: about the same speed on two GPUs with the same memory system, whatever the SM count.

In the full transformer block (`block/block.cu`, #10), attention v2 is 53–55% of the step time at sequence length 2048 on the 4060 (41.8 ms of the 8×2048 block's steps).

## What would change the conclusion

Only removing the N² traffic: computing softmax inside the GEMM tiles so the scores never reach DRAM (fused / FlashAttention-style). Then intensity grows with N and attention becomes compute-bound. That is deliberately out of scope here (D10); this project measures the unfused cost.
