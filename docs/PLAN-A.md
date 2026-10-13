# KernLine — Part A: compute side + performance model (RTX 4070 Laptop)

Self-contained plan for Part A. **Read `docs/DESIGN.md` first.** It holds the design decisions (D1–D10), the environment setup, the repo layout, and the frozen interfaces.

## Your role
You own GEMM (naive → tensor cores) and the analytical performance model. Part B (RTX 4060 Laptop) owns softmax, LayerNorm, attention, and the cache simulator, and has already created the repo, docs, test harness, and ceiling microbenchmarks. The model is **tuned on your 4070 Laptop only**. B's 4060 Laptop data is the held-out test (D1).

## Day 1
1. Follow *Environment* in `DESIGN.md` on your machine. Clone the repo into `~/kernline`.
2. Run `tools/smoke/run.sh` and confirm `ncu` reads counters.
3. Record `nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem --format=csv` in `specs/rtx4070_laptop.json`. Then run `tools/ceilings/` once B has pushed it.

## Weekly tasks
| Week | Branch | Tasks |
|---|---|---|
| 1 | `a/scaffold` | `CMakeLists.txt`, `common/` (cuda_check, cudaEvent timer, gpu_info → CSV), `src/api.cu`, `bench/bench.cu` emitting the frozen CSV schema, **GEMM v0** naive (one thread per C element, uncoalesced on purpose) + test vs `torch.matmul` + cuBLAS baseline. Open the PR. B needs it merged (Gate 1) to start attention. |
| 2–3 | `a/gemm` | GEMM v1 coalesced (thread→column mapping), v2 shared-memory tiling (32×32). Expect ncu to show DRAM bytes dropping, then the kernel becoming shared-memory bound. Review B's `b/tooling`. |
| 4 | `a/gemm` | GEMM v3 register tiling (2D thread tile, e.g. 8×8, BM=BN=128, BK=8, float4 loads, transposed/padded A tile against bank conflicts). Target ≥ 60% of measured FP32 peak. Open the PR once v3 is solid, because B's attention v1 waits on it. |
| 5 | `a/gemm` → `a/model` | GEMM v4 WMMA FP16-in/FP32-acc (stretch: a follow-up commit or small PR, don't block on it). Start `a/model`: spec loader, occupancy calculator (regs, smem, warps, blocks/SM; check against ncu "theoretical occupancy"). |
| 6 | `a/model` | Waves/tail effect, tiers T0/T1/T2 (D2), GEMM descriptors, CLI `python -m model predict` / `validate`. `block/block.cu` transformer-block runner (D4). Full bench suite on the 4070 Laptop incl. B's kernels (D8). |
| 7 | `a/model` | Tune on 4070 Laptop data only. Freeze, then predict the 4060 Laptop from B's `b/model-inputs` data. Error tables per tier × kernel × GPU, scatter plots, written failure analysis (latency at small sizes, bank conflicts, L2 reuse, clock variance, launch overhead). Block-level prediction vs measured. Open the PR (Gate 3). |
| 8 | `results` | Review B's `b/sim` (especially that its GEMM address replay matches your kernels' index math). Open the `results` PR assembling the README with B. |

## Branches and PRs
Follow *Workflow* in `DESIGN.md`. Your default branches are `a/scaffold`, `a/gemm`, `a/model`, and `results`. Aim for at least ~4 real commits per PR, and send plan changes as small `docs/<topic>` PRs. B reviews and merges yours. You review B's (`b/tooling`, `b/reductions`, `b/attention`, `b/model-inputs`, `b/sim`, and any `docs/*`) per *Reviewing a PR*. Respect the gates: don't open optimization PRs before Gate 1, and don't open `a/model` before Gate 2 and `b/model-inputs`.

## Definition of done per variant (one commit each)
Correctness test passes (including sizes not divisible by the tile) → CSV row in `results/rtx4070_laptop/` → ncu report → one-sentence roofline explanation, collected into the PR's table.

## Verification
- `cmake -B build -G Ninja && cmake --build build`
- `pytest tests/ -k gemm`
- `tools/profile.sh gemm v3 4096 4096 4096` gives an ncu report with non-zero counters
- `python -m model validate --gpu rtx4060_laptop` prints the held-out error table, and no 4060 data was used for tuning
