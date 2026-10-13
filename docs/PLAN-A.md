# KernLine — Part A: compute side + performance model (RTX 4070 Laptop)

Self-contained plan for Part A. **Read `docs/DESIGN.md` first.** It holds the design decisions (D1–D10), the environment setup, the repo layout, and the frozen interfaces.

## Your role
You own GEMM (naive → tensor cores) and the analytical performance model. Part B (RTX 4060 Laptop) owns softmax, LayerNorm, attention, and the cache simulator, and has already created the repo, docs, test harness, and ceiling microbenchmarks. The model is **tuned on your 4070 Laptop only**. B's 4060 Laptop data is the held-out test (D1).

## Day 1
1. Follow *Environment* in `DESIGN.md` on your machine. Clone the repo into `~/kernline`.
2. Run `tools/smoke/run.sh` and confirm `ncu` reads counters.
3. Record `nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem --format=csv` in `specs/rtx4070_laptop.json`. Then run `tools/ceilings/` once B has pushed it.

## Weekly tasks
| Week | Tasks |
|---|---|
| 1 | Build scaffold: `CMakeLists.txt`, `common/` (cuda_check, cudaEvent timer, gpu_info → CSV), `src/api.cu`, `bench/bench.cu` emitting the frozen CSV schema. |
| 2 | GEMM v0 naive (one thread per C element, uncoalesced on purpose). Tests vs `torch.matmul`. cuBLAS baseline in bench. |
| 3 | GEMM v1 coalesced (thread→column mapping), v2 shared-memory tiling (32×32). Expect ncu to show DRAM bytes dropping, then the kernel becoming shared-memory bound. |
| 4 | GEMM v3 register tiling (2D thread tile, e.g. 8×8, BM=BN=128, BK=8, float4 loads, transposed/padded A tile against bank conflicts). Target ≥ 60% of measured FP32 peak. |
| 5 | GEMM v4 WMMA FP16-in/FP32-acc (stretch). `block/block.cu` skeleton wiring your GEMMs with B's kernels (D4). |
| 6 | `model/`: spec loader, occupancy calculator (regs, smem, warps, blocks/SM. Check against ncu "theoretical occupancy"), waves/tail effect, tiers T0/T1/T2 (D2). GEMM descriptors. CLI `python -m model predict` / `validate`. |
| 7 | Tune on 4070 Laptop data only. Freeze, then predict the 4060 Laptop from B's results. Error tables per tier × kernel × GPU, scatter plots, written failure analysis (latency at small sizes, bank conflicts, L2 reuse, clock variance, launch overhead). Block-level prediction vs measured. |
| 8 | Support B's simulator: document GEMM v0/v2/v3 index math for address replay (D5). README assembly with B. |

## Definition of done per step
Correctness test passes (including sizes not divisible by the tile) → CSV row in `results/rtx4070_laptop/` → ncu report → one-sentence roofline explanation in the PR.

## Verification
- `cmake -B build -G Ninja && cmake --build build`
- `pytest tests/ -k gemm`
- `tools/profile.sh gemm v3 4096 4096 4096` gives an ncu report with non-zero counters
- `python -m model validate --gpu rtx4060_laptop` prints the held-out error table, and no 4060 data was used for tuning
