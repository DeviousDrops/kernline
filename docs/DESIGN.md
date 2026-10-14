# KernLine — Design

Shared contract for both contributors. Read this before `PLAN-A.md` / `PLAN-B.md`.

## Goal
Hand-written CUDA kernels for a transformer block (GEMM, softmax, LayerNorm, attention), optimized step by step with Nsight Compute evidence and roofline plots. Then an analytical performance model that predicts their runtime before running them, validated on two GPUs. Stretch goal: a trace-driven cache simulator.

## Team and hardware
| Part | Owner | GPU | Focus |
|---|---|---|---|
| A | Arjun | RTX 4070 Laptop (AD106, `sm_89`, 36 SM, 8 GB, 128-bit GDDR6 ~256 GB/s, 32 MB L2) | GEMM + performance model |
| B | Kartik | RTX 4060 Laptop (AD107, `sm_89`, 24 SM, 8 GB, 128-bit GDDR6 ~256 GB/s, 32 MB L2) | softmax, LayerNorm, attention + cache simulator |

Both GPUs are Ada laptop parts with the **same memory system** (bus width, nominal bandwidth, L2 size) and **different SM counts** (36 vs 24). That makes this a near-controlled experiment: compute-bound kernels should scale with SMs × clock (~1.5×), while memory-bound kernels should run at about the same speed on both. Laptop power limits (TGP) differ per machine, so clocks must be measured, not assumed (D3, D7). The second GPU is the model's held-out test set.

## Design decisions
| # | Decision | Why |
|---|---|---|
| D1 | **Two-GPU held-out validation.** Build and tune the model with 4070 Laptop data only, then predict the 4060 Laptop with **no refitting**. | Turns "model error" into "does the model generalize across SKUs?", which is what architecture teams actually do. Same memory system + different SM count makes it a sharp test: the model must predict ~1.5× for compute-bound and ~1× for memory-bound kernels. |
| D2 | **Tiered model.** T0 pure roofline → T1 + occupancy & wave quantization (tail effect on 36 vs 24 SMs) → T2 + latency hiding via Little's law (bytes in flight needed = BW × mem latency, vs bytes in flight the occupancy provides) + launch overhead. Report error per tier. | Shows *which* effect explains *which* error. |
| D3 | **Measured ceilings, not datasheet.** `tools/ceilings/` microbenchmarks (stream copy, FMA loop, smem bandwidth, mma loop, pointer-chase latency) per GPU, stored as `specs/<gpu>.json`. | Laptop clocks vary. Pointer-chase latency feeds T2. |
| D4 | **End-to-end transformer block:** LN → QKV GEMM → attention → out-proj GEMM → residual → LN → MLP GEMMs, GPT-2-small shapes (d=768, heads=12, seq 512–2048, batch 1–8). Compare measured block time vs the sum of model predictions vs PyTorch eager. | Shows where time goes in a real layer. |
| D5 | **Cache sim uses CPU address replay**, not GPU instrumentation. Re-execute each kernel's index math on the CPU to emit (SM, warp, addr, r/w) traces. Validate against ncu `l1tex__t_sector_hit_rate` / `lts__t_sector_hit_rate`. | GPU tracing tools are unreliable on WSL2. Replay is deterministic, and the ncu cross-check keeps it honest. |
| D6 | **Kernel launchers behind a stable C ABI** (`extern "C"` in `src/api.cu` → `libkernels.so`). Tests use ctypes + torch `data_ptr()`. Bench is a C++ CLI emitting CSV. | No pybind/torch-extension build coupling. |
| D7 | **Every result row carries GPU name + clocks.** ncu runs pass `--clock-control base` explicitly (ncu 2026.x defaults to `boost`). | Results from two machines merge without ambiguity. |
| D8 | **Cross-running:** each person runs the *full* bench + ncu suite on their own GPU, including the partner's kernels. | Every plot has two GPUs, and the SM-scaling vs flat-bandwidth contrast shows up directly. |
| D9 | **Baselines:** cuBLAS / `torch.matmul`, `F.softmax`, `F.layer_norm`, `F.scaled_dot_product_attention` forced to the math (unfused) backend. Report "% of baseline" next to "% of peak". | "% of peak" alone flatters naive memory-bound kernels. |
| D10 | **Dtype:** FP32 for all main variants. FP16-in/FP32-acc only for WMMA GEMM. Non-causal attention first. Attention stays **unfused** (no FlashAttention): QKᵀ → softmax → ·V reuses the GEMM and softmax kernels. | Simple tolerances, apples-to-apples comparisons. |

## Environment (each machine)
1. **Windows:** update the NVIDIA driver to the latest (≥ 580 for CUDA 13, which Arch ships). NVIDIA Control Panel → Desktop → *Enable Developer Settings*, then Developer → *Manage GPU Performance Counters* → **Allow access to the GPU performance counters to all users**. Run `wsl --shutdown`.
2. **Arch WSL** (never install a Linux NVIDIA driver inside WSL): `bash tools/setup_arch.sh && source ~/.bashrc`. This installs `cuda` (nvcc in `/opt/cuda/bin`), `nsight-compute` (a separate Arch package; `ncu` in `/opt/nsight-compute`), cmake, ninja, python, and uv, and generates the locale. From Git Bash on Windows, prefix `wsl` calls with `MSYS_NO_PATHCONV=1`, or `/root/...` paths get mangled.
3. **Python:** `uv venv && uv pip install torch numpy pandas matplotlib pytest`.
4. Keep the repo in the Linux filesystem (`~/kernline`), **not** `/mnt/c` or `/mnt/d`, because 9P makes builds slow.
5. **Go/no-go:** `tools/smoke/run.sh` builds a vector-add and runs it under `ncu`. If `ERR_NVGPUCTRPERM` appears, recheck step 1.

## Repo layout (the contract)
```
CMakeLists.txt        CUDA_ARCHITECTURES 89, -lineinfo, builds libkernels.so + bench + block
common/               cuda_check.h, timer.h, gpu_info.h (name, clocks → CSV)
src/api.cu            extern "C" launchers, one per kernel+variant
kernels/{gemm,softmax,layernorm,attention}/vN_*.cu
bench/bench.cu        bench <kernel> <variant> <dims> → CSV row (warmup, median of ≥20)
block/block.cu        end-to-end transformer block runner (D4)
tests/                pytest via ctypes vs torch, incl. sizes not divisible by tile
tools/smoke/          day-1 go/no-go check
tools/profile.sh      ncu wrapper → results/ncu/<gpu>/*.ncu-repz + metrics CSV
tools/suite.sh        full bench + ncu suite (D8): every kernel, variant and model shape
tools/ceilings/       microbenchmarks → specs/<gpu>.json
analysis/roofline.py  plots from results/*.csv, one panel per GPU
model/                Python performance model (Part A); write-up in analysis/model.md
sim/                  C++ cache simulator (Part B)
specs/<gpu>.json      measured ceilings + static specs
results/<gpu>/*.csv   committed measurements
```

## Frozen interfaces
- **Launcher signature style:** `extern "C" int gemm_v2(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);` returns a `cudaError_t` cast to int. Row-major, device pointers.
- **Memory:** benchmarks, tests, and microbenchmarks use `cudaMalloc` + explicit `cudaMemcpy`, never `cudaMallocManaged`. Under WSL2, managed pages migrate on first touch *inside* the kernel, so timings and ncu DRAM bytes measure page migration, not the kernel.
- **ncu reports:** Nsight Compute 2026.x writes compressed `.ncu-repz`. Commit them under `results/ncu/<gpu>/<kernel>_<variant>_<dims>.ncu-repz`. `tools/profile.sh` writes them, fills `dram_bytes`/`achieved_occ` in `results/<gpu>/<kernel>.csv`, and upserts the full metric row (hit rates, bank conflicts, regs, smem, top stall reason) into `results/ncu/<gpu>/metrics.csv`. Profiling runs at `--clock-control base` so counters are comparable across the two laptops; `time_ms` comes from the bench at boost clocks.
- **Timing method (`bench`):** warm up for ≥ 3 launches and ≥ 500 ms, and until NVML reports the maximum memory clock (at most 5 s; after an idle gap the memory clock can lag the SM clock by over a second), then 21 timed runs. Each run is a batch of back-to-back launches lasting ≥ 5 ms, divided by the batch size, and `time_ms` is the median. `sm_clk_mhz`/`mem_clk_mhz` are the median NVML readings taken while the timed kernels are in flight.
- **Specs file (`specs/<gpu>.json`):** hand-written keys (`name`, `chip`, `driver`, `os`, ...), plus `device` (SM count, regs/smem/threads/blocks per SM, L2 size: everything the occupancy model needs) and `ceilings` (`fp32_gflops`, `fp16_tc_fp32acc_gflops`, `dram_read_gbs`, `dram_copy_gbs`, `l2_read_gbs`, `smem_read_gbs`, `latency_ns`/`latency_cycles` for L1/L2/DRAM, and the clocks they were measured at). `tools/ceilings/run.sh` rewrites only `device` and `ceilings`.
- **CSV schema:** `gpu,sm_clk_mhz,mem_clk_mhz,kernel,variant,dims,time_ms,flop,bytes_theory,dram_bytes,achieved_occ,notes`. `dims` is `x`-joined, e.g. `4096x4096x4096`.
- **ncu metrics:** `gpu__time_duration.sum`, `dram__bytes_read.sum`, `dram__bytes_write.sum`, `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__warps_active.avg.pct_of_peak_sustained_active`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`, `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`, `smsp__average_warps_issue_stalled_*_per_issue_active` (warp-cycles stalled per issued instruction, the "Warp State" chart; `metrics.csv` stores the largest as `top_stall` / `top_stall_per_issue`). Not `smsp__average_warp_latency_issue_stalled_*`: that is total cycles per warp, so it grows with kernel length and favours long stalls over frequent ones.
- **Model kernel descriptor** (`model/kernels/<kernel>.py`): `flops(dims)`, `bytes(dims, variant)`, `launch_cfg(dims, variant)`, `resources(variant)` (regs/smem from `nvcc --resource-usage`).

## Workflow
Nobody pushes to `main`. Work happens on **branches cut around real units of work**: one coherent, reviewable chunk that someone else can build, test, and reason about as a whole. That means neither one branch per kernel variant nor one per calendar phase. That gives roughly ten PRs over the project.

### Planned branches
| # | Branch | Owner | Contents (each bullet ≈ one or more commits) | Weeks | Start after | Merge before |
|---|---|---|---|---|---|---|
| 1 | `docs/setup` | B | workflow, PR template, `.gitignore` | 1 | — | anything else |
| 2 | `a/scaffold` | A | CMake · `common/` · C ABI in `src/api.cu` · bench CLI + CSV · GEMM v0 + cuBLAS baseline + test | 1 | #1 | **Gate 1** |
| 3 | `b/tooling` | B | ctypes/pytest harness · `tools/profile.sh` · ceiling microbenchmarks · `specs/rtx4060_laptop.json` · `analysis/roofline.py` | 1 | #1 | **Gate 1** |
| 4 | `a/gemm` | A | GEMM v1 coalesced · v2 smem tiling · v3 register tiling · v4 WMMA (stretch) · results + ncu per variant | 2–5 | Gate 1 | **Gate 2** |
| 5 | `b/reductions` | B | softmax v0 · v1 · v2, LayerNorm v0 · v1 · v2 · results + ncu per variant | 2–4 | Gate 1 | **Gate 2** |
| 6 | `b/attention` | B | attention v0 (unfused, GEMM v0) · v1 (best GEMM + fused scale + softmax v2) · sequence-length sweep · memory-bound analysis | 2–5 | Gate 1 (rebase on #4, #5 before opening the PR) | **Gate 2** |
| 7 | `b/model-inputs` | B | model descriptors for softmax/LN/attention · full bench + ncu suite on the 4060 Laptop incl. A's GEMMs (held-out data) | 6 | Gate 2 | **Gate 3** |
| 8 | `a/model` | A | spec loader · occupancy calc · T0/T1/T2 tiers · GEMM descriptors · 4070 tuning + full suite · transformer-block runner · held-out validation | 5–7 | Gate 1 (open PR after #7) | **Gate 3** |
| 9 | `b/sim` | B | address-replay generator · L1 · L2 · tiling experiments · ncu cross-check | 7–8 | Gate 2 | final PR |
| 10 | `results` | A opens, both commit | README tables, plots, "who did what" | 8 | Gate 3 (+ #9 if done) | — |

**Gates (what must be merged before moving on):**
- **Gate 1** (#2 + #3): the build system, launcher ABI, CSV schema, and test harness exist on `main`. No kernel-optimization branch opens a PR before this.
- **Gate 2** (#4 + #5 + #6): every kernel variant is final. After this, kernels only change for bug fixes, so the model and the simulator measure something stable.
- **Gate 3** (#7 + #8): model validated on both GPUs. The `results` PR is written against this.

### Rules
- **At least ~4 meaningful commits before opening a PR** (e.g. one per kernel variant, plus results/tests). This is a guideline, not a quota. Never pad with trivial commits, and if a branch is getting huge (> ~1500 changed lines excluding results CSVs), split it.
- **Plan changes are welcome as their own small PR** (`docs/<topic>`, e.g. `docs/drop-wmma`; repo automation uses `ci/<topic>`). Batch related doc edits into one PR instead of opening several, and fold a doc change into the feature branch when it only concerns that feature. These are exempt from the 4-commit guideline.
- **Branches above are the default, not a contract.** Rename, merge, or split them when the work changes shape. Record that in a `docs/` PR, or note it in the affected PR's description.
- Cut each branch from an up-to-date `main` (`git switch main && git pull && git switch -c b/reductions`). Before opening the PR, `git rebase main`.
- Optional: open a **draft PR** early (`gh pr create --draft`) on long branches (#4, #6, #8), so the partner can comment along the way.
- **The other person reviews and merges.** Authors never merge their own PRs. That review is how both learn both halves.
- Merge with a **merge commit** (`gh pr merge --merge`), not squash, so each kernel variant stays a separate commit on `main` that the README can link to.
- Kernel PRs fill in `.github/pull_request_template.md`. Each variant needs a correctness pass, a CSV row from your GPU, an ncu report, and one sentence: *"moved from X to Y because <bound> — ncu showed <metric>"*.
- Commits, PR descriptions, and review comments contain no AI/tool attribution lines (no `Co-Authored-By` trailers, no "generated with" footers).

### Enforcement
- `main` is protected. Changes need a PR with **1 approving review** from someone other than the author, all review conversations resolved, and the `no-attribution` + `branch-name` checks passing. Direct pushes, force pushes, and deletion are blocked, for the repo owner too.
- `.github/workflows/pr-checks.yml` runs on every PR. `no-attribution` rejects co-author trailers and "generated with [...]" footers in commit messages or the PR description. `branch-name` requires `a/*`, `b/*`, `docs/*`, `ci/*`, or `results`.
- Branch *count* and the ~4-commit guideline are judgment calls. The reviewer checks them via the PR template's checklist.
- If an attribution line slips into a commit, fix it on the branch (`git commit --amend` for the latest commit, or `git rebase -i main` and reword) and `git push --force-with-lease`. Force-pushing your own feature branch is fine. Only `main` is locked.

### Reviewing a PR
1. Check out the branch: `gh pr checkout <n>`.
2. Build and run the tests **on your own GPU**: `cmake --build build && pytest tests/`. A PR that passes only on the author's GPU is not done.
3. Re-run the PR's benchmark row on your GPU and check that the numbers and the roofline sentence are plausible (the ncu metric actually supports the claim).
4. Check the frozen interfaces (launcher signature, CSV schema, metric list) are unchanged, or changed together with `docs/DESIGN.md`.
5. Decide:
   - **Good:** `gh pr review <n> --approve`, then `gh pr merge <n> --merge --delete-branch`.
   - **Needs changes:** `gh pr review <n> --request-changes -b "<what and why>"`, plus inline comments on specific lines (`gh api repos/{owner}/{repo}/pulls/<n>/comments`). The author pushes fixes to the same branch and re-requests review.
   - **Small question, not blocking:** `gh pr review <n> --comment -b "..."`.
