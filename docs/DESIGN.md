# KernLine — Design

Shared contract for both contributors. Read this before `PLAN-A.md` / `PLAN-B.md`.

## Goal
Hand-written CUDA kernels for a transformer block (GEMM, softmax, LayerNorm, attention), optimized step by step with Nsight Compute evidence and roofline plots. Then an analytical performance model that predicts their runtime before running them, validated on two GPUs. Stretch goal: a trace-driven cache simulator.

## Team and hardware
| Part | Owner | GPU | Focus |
|---|---|---|---|
| A | Partner | RTX 4070 Laptop (AD106, `sm_89`, 36 SM, 8 GB, 128-bit GDDR6 ~256 GB/s, 32 MB L2) | GEMM + performance model |
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
| D7 | **Every result row carries GPU name + clocks.** ncu runs keep its default `--clock-control base`. | Results from two machines merge without ambiguity. |
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
tools/profile.sh      ncu wrapper → results/ncu/<gpu>/*.ncu-rep + metrics CSV
tools/ceilings/       microbenchmarks → specs/<gpu>.json
analysis/roofline.py  plots from results/*.csv, one panel per GPU
model/                Python performance model (Part A)
sim/                  C++ cache simulator (Part B)
specs/<gpu>.json      measured ceilings + static specs
results/<gpu>/*.csv   committed measurements
```

## Frozen interfaces
- **Launcher signature style:** `extern "C" int gemm_v2(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);` returns a `cudaError_t` cast to int. Row-major, device pointers.
- **CSV schema:** `gpu,sm_clk_mhz,mem_clk_mhz,kernel,variant,dims,time_ms,flop,bytes_theory,dram_bytes,achieved_occ,notes`. `dims` is `x`-joined, e.g. `4096x4096x4096`.
- **ncu metrics:** `gpu__time_duration.sum`, `dram__bytes_read.sum`, `dram__bytes_write.sum`, `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__warps_active.avg.pct_of_peak_sustained_active`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`, `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`, `smsp__average_warp_latency_issue_stalled_*`.
- **Model kernel descriptor** (`model/kernels/<kernel>.py`): `flops(dims)`, `bytes(dims, variant)`, `launch_cfg(dims, variant)`, `resources(variant)` (regs/smem from `nvcc --resource-usage`).

## Workflow
- **One branch and one PR per person per phase.** Nobody pushes to `main`. Commit each kernel variant separately inside the phase branch, but open a single PR when the phase is done.

  | Phase | Weeks | Part A branch | Part B branch |
  |---|---|---|---|
  | 1 Baseline: scaffold, tooling, v0 kernels | 1–2 | `a/phase1-baseline` | `b/phase1-baseline` |
  | 2 Optimize: all variants + roofline evidence | 3–5 | `a/phase2-optimize` | `b/phase2-optimize` |
  | 3 Model + transformer block | 6–7 | `a/phase3-model` | `b/phase3-model` |
  | 4 Cache simulator | 8 | `a/phase4-sim` (index math only) | `b/phase4-sim` |

- Cut each phase branch from an up-to-date `main` when the phase starts (`git switch main && git pull && git switch -c b/phase2-optimize`).
- Optional: open the PR as a **draft** early (`gh pr create --draft`), so the partner can comment while you work. Mark it ready when the phase is done.
- **Cross-dependencies:** B's attention needs A's GEMM (v0 in phase 1, best variant in phase 2). A merges their phase PR first, or at least the GEMM commits. B then runs `git rebase main` and finishes attention.
- **The other person reviews and merges.** Authors never merge their own PRs. That review is how both learn both halves.
- Each phase PR fills in `.github/pull_request_template.md`. For every variant it needs a correctness pass, a CSV row from your GPU, an ncu report, and one sentence: *"moved from X to Y because <bound> — ncu showed <metric>"*.
- Commits, PR descriptions, and review comments contain no AI/tool attribution lines (no `Co-Authored-By` trailers, no "generated with" footers).
- Weekly sync: merge both GPUs' results and regenerate the plots.

### Reviewing a PR
1. Check out the branch: `gh pr checkout <n>`.
2. Build and run the tests **on your own GPU**: `cmake --build build && pytest tests/`. A PR that passes only on the author's GPU is not done.
3. Re-run the PR's benchmark row on your GPU and check that the numbers and the roofline sentence are plausible (the ncu metric actually supports the claim).
4. Check the frozen interfaces (launcher signature, CSV schema, metric list) are unchanged, or changed together with `docs/DESIGN.md`.
5. Decide:
   - **Good:** `gh pr review <n> --approve`, then `gh pr merge <n> --squash --delete-branch`.
   - **Needs changes:** `gh pr review <n> --request-changes -b "<what and why>"`, plus inline comments on specific lines (`gh api repos/{owner}/{repo}/pulls/<n>/comments`). The author pushes fixes to the same branch and re-requests review.
   - **Small question, not blocking:** `gh pr review <n> --comment -b "..."`.
