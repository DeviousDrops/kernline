# KernLine — Part B: memory side + cache simulator (RTX 4060 Laptop)

**Read `docs/DESIGN.md` first.**

## Your role
You own softmax, LayerNorm, attention (naive unfused → optimized unfused), the shared test/profiling tooling, and the cache simulator. Part A (RTX 4070 Laptop) owns GEMM and the performance model. Your 4060 Laptop measurements are the model's held-out test set (D1), so never hand-tune the model against them.

## Weekly tasks
| Week | Branch | Tasks |
|---|---|---|
| 1 | `docs/setup`, `b/tooling` | Day-1 environment setup. `tools/smoke/` go/no-go. `tests/` ctypes+pytest harness, `tools/profile.sh`, `tools/ceilings/` (stream copy, FMA, smem BW, mma, pointer-chase latency) → `specs/rtx4060_laptop.json`, `analysis/roofline.py` skeleton. Open the PR. Review A's `a/scaffold`. Both merged = Gate 1. |
| 2 | `b/reductions`, `b/attention` | softmax v0 (row per thread, 3 passes), LayerNorm v0 (naive two-pass mean/var). On `b/attention`: attention v0 unfused (A's GEMM v0 → softmax v0 → GEMM, materializes N×N). Tests vs torch. |
| 3 | `b/reductions` | softmax v1 / LN v1: row per warp/block, `__shfl_xor_sync` reductions, float4 loads. |
| 4 | `b/reductions` | softmax v2 online single-pass (running max + rescaled sum), LN v2 Welford single-pass. Target ≥ 80% of measured DRAM BW. Open the PR. Review A's `a/gemm`. |
| 5 | `b/attention` | Rebase on `main` once `a/gemm` and `b/reductions` are merged. Attention v1 optimized unfused: batched (B×H) tiled QKᵀ using A's best GEMM with the 1/√d scale fused into its epilogue → softmax v2 in place on S → tiled P·V. It still materializes N×N. Compare vs the PyTorch math path (`sdpa_kernel(SDPBackend.MATH)`). Sequence-length sweep + "why attention is memory-bound" analysis (arithmetic intensity vs N, ncu DRAM bytes, crossover on each GPU). Open the PR. Merged with `a/gemm` and `b/reductions` = Gate 2. |
| 6 | `b/model-inputs` | Model descriptors for softmax/LN/attention. Full bench + ncu suite on the 4060 Laptop, including A's GEMMs (D8). This is the held-out set, so never tune the model against it. Open the PR. |
| 7–8 | `b/sim` | `sim/` (C++): address-replay trace generator (D5) + set-associative L1 per SM (128 B lines, 32 B sectors, LRU) + shared L2 (32 MB on both GPUs). Hit rate vs tile size for GEMM v0/v2/v3 and attention v0/v1. Cross-check with ncu hit rates and report the gap. Open the PR. Review A's `a/model`. Then add your sections to the `results` PR. |

## Branches and PRs
Follow *Workflow* in `DESIGN.md`. Your default branches are `docs/setup`, `b/tooling`, `b/reductions`, `b/attention`, `b/model-inputs`, and `b/sim`, plus your commits on `results`. Aim for at least ~4 real commits per PR, and send plan changes as small `docs/<topic>` PRs. A reviews and merges yours. You review A's (`a/scaffold`, `a/gemm`, `a/model`, `results`, and any `docs/*`) per *Reviewing a PR*.

## Definition of done per variant (one commit each)
Correctness test passes → CSV row in `results/rtx4060_laptop/` → ncu report → one-sentence roofline explanation, collected into the PR's table.
