# Why unfused attention is memory-bound

Measured on the RTX 4060 Laptop (24 SMs, FP32 14.1 TFLOP/s, DRAM 249 GB/s, ridge point 57 FLOP/byte), GPT-2 small shapes: 12 heads, head size d = 64, sequence length N = 256 … 4096, FP32, non-causal. Every row at memory clock 8001 MHz. Data: `results/rtx4060_laptop/attention.csv`, per-kernel phases from the ncu reports via `analysis/attention_breakdown.py`.

## The argument in one line

Unfused attention writes the N×N score matrix to DRAM and reads it back. Per head that is 4·N² floats of traffic (Q·Kᵀ writes S, softmax reads S and writes P, P·V reads P) against 4·N²·d FLOPs of GEMM, so its arithmetic intensity is at most **d/4 FLOP/byte, independent of N**. For d = 64 that is 16 FLOP/byte, below the 4060's ridge of 57 and the 4070's of 84. Making N larger does not help: the FLOPs and the N² traffic grow together. Unfused attention stays memory-bound for any d below 4 × ridge (228 on the 4060, 336 on the 4070).

The roofline confirms it: every implementation sits under the DRAM slope at AI ≈ 16 (`analysis/plots/attention.png`), and ncu measures v2 moving exactly the unfused floor of DRAM bytes (0.99–1.00× `bytes_theory`) at every N ≥ 1024.

## Variants

| | how | kernel launches per call |
|---|---|---|
| v0 | Kᵀ, `gemm_v0` per head, separate pass scaling S by 1/√d, `softmax_v0`, `gemm_v0` per head | 2·heads + 3 |
| v1 | Kᵀ written pre-scaled by 1/√d (no pass over S), `gemm_v3` per head, softmax v2 (N ≤ 2048) or v3 | 2·heads + 2 |
| v2 | v1 with each GEMM batched over all heads in one launch (`gemm_v3_batched`, gridDim.z = heads) | 4 |
| torch | `F.scaled_dot_product_attention`, math backend (unfused) | ~6 |

## Results (median ms, bench at boost clocks)

| N | v0 | v1 | v2 | torch | v2 vs torch | DRAM floor | v2 % of floor |
|---|---|---|---|---|---|---|---|
| 256 | 1.153 | 0.565 | **0.062** | 0.086 | 1.39× | 0.063 | L2-resident |
| 512 | 4.330 | 0.846 | **0.174** | 0.268 | 1.54× | 0.227 | L2-resident |
| 1024 | 17.00 | 2.408 | **1.262** | 1.746 | 1.38× | 0.859 | 68% |
| 2048 | 71.46 | 6.237 | **5.216** | 7.022 | 1.35× | 3.335 | 64% |
| 4096 | 274.3 | 21.73 | **20.32** | 28.81 | 1.42× | 13.14 | 65% |

"DRAM floor" is `bytes_theory` (the unfused minimum: Q, K, V, O once plus four passes over the scores) at the measured 249 GB/s. At N ≤ 512 the 12 heads' scores (3–13 MB) fit in the 32 MB L2, so the passes over S never reach DRAM and v2 beats the "floor".

**Why each step moved:**
- **v0 → v1 (4.8–13×):** the GEMMs. `gemm_v0` is uncoalesced, and the separate scale pass costs one more read and write of S (2% of v0's time; it matters more in torch, below).
- **v1 → v2 (9× at N = 256, 1.07× at N = 4096):** occupancy. At N = 256, one head's Q·Kᵀ is 4 tiles of 128×128, so `gemm_v3` per head leaves 20 of 24 SMs idle. P·V (N×64 output) is worse: 2 tiles per head. Batching the heads into one grid fills the GPU. At N = 4096 each head already has 1024 tiles, so batching only removes launch gaps.
- **v2 vs torch (1.35–1.54×):** traffic. ncu measures torch moving 1.86× the unfused floor at N = 4096 (6.09 GB vs 3.27 GB) because the math backend runs separate elementwise kernels (scale, copies; 37% of its kernel time). v2 folds the scale into the Kᵀ prep and moves 1.00×.

## Where v2's time goes (N = 4096, kernel time at ncu's base clocks)

| phase | share | DRAM traffic | at the DRAM roof it would take | achieved | limited by |
|---|---|---|---|---|---|
| Kᵀ prep | 0.3% | Q-sized | – | – | – |
| Q·Kᵀ | 23% (5.1 ms) | writes S: 0.81 GB | 3.2 ms | 63% of DRAM BW | storing S: only 8 k-steps (K = d = 64) per tile, register-limited occupancy (33%) leaves few stores in flight |
| softmax | 32% (7.1 ms) | reads S, writes P: 1.61 GB | 6.5 ms | **91% of DRAM BW** | DRAM bandwidth (softmax v3, one read per score) |
| P·V | 45% (9.9 ms) | reads P: 0.81 GB | 3.2 ms | 33% of DRAM BW | **wasted compute**: the output is N×64 but `gemm_v3`'s tile is 128×128, so half of every tile computes zeros over a K = 4096 loop |

So v2 runs at 65% of the unfused roofline, and the softmax phase is already at the roof. The remaining gap is in the GEMMs' shapes, not in attention's structure. The biggest lever is a GEMM tile matched to d = 64 (e.g. 128×64) for P·V, which would roughly halve its time. That belongs to `gemm_v3` (`a/gemm`), so it's noted here rather than worked around.

## Predictions for the 4070 Laptop (held-out check, D1/D8)

Same DRAM (249 GB/s), 1.5× the SMs.
- **Softmax phase:** same time as on the 4060 (DRAM-bound at 91%).
- **Q·Kᵀ phase:** close to the 4060, within ~15% (store-bound).
- **P·V phase:** up to ~1.4× faster (it's limited by wasted compute, and compute scales with SMs × clock).
- **Whole call at N ≥ 1024:** therefore ~1.1–1.25× faster than the 4060, much less than the ~1.4× GEMM v3 gains. If the 4070 shows ~1.4×, the "softmax and Q·Kᵀ are memory-bound" reading above is wrong.

## What would change the conclusion

Only removing the N² traffic: computing softmax inside the GEMM tiles so the scores never reach DRAM (fused / FlashAttention-style). Then intensity grows with N and attention becomes compute-bound. That is deliberately out of scope here (D10); this project measures the unfused cost.
