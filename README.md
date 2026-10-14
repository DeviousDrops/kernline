# KernLine

**Optimize a transformer block on the GPU, then predict its performance before running it.**

KernLine is hand-written CUDA C++ for the four kernels behind a transformer layer: GEMM, softmax, LayerNorm, and attention. Each kernel is optimized step by step, and every step is backed by Nsight Compute evidence and placed on a roofline chart. An analytical performance model (roofline + occupancy + latency hiding) then predicts the kernels' runtime. It is tuned on one GPU and tested on a second GPU it has never seen. A trace-driven cache simulator shows how tiling changes L1/L2 hit rates.

## Team & hardware
| Part | Focus | GPU |
|---|---|---|
| A | GEMM (naive → register tiling → WMMA), performance model | RTX 4070 Laptop (Ada, `sm_89`, 36 SMs, 8 GB) |
| B | softmax, LayerNorm, attention (naive → optimized unfused), cache simulator, tooling | RTX 4060 Laptop (Ada, `sm_89`, 24 SMs, 8 GB) |

Both are Ada laptop GPUs with the same memory system (128-bit GDDR6, ~256 GB/s, 32 MB L2) but different SM counts (36 vs 24): compute-bound kernels should scale with SMs, memory-bound ones should not. That makes the second GPU a clean held-out test for the model. Details are in [`docs/DESIGN.md`](docs/DESIGN.md). Per-person plans: [`docs/PLAN-A.md`](docs/PLAN-A.md) and [`docs/PLAN-B.md`](docs/PLAN-B.md).

## Phases
1. **Naive baseline.** Plain CUDA kernels, checked against PyTorch, profiled for FLOP/s, bandwidth, and occupancy.
2. **Optimize with evidence.**
   - GEMM: coalescing → shared-memory tiling → register tiling → tensor cores (WMMA).
   - Softmax/LayerNorm: warp-shuffle reductions and single-pass algorithms (online softmax, Welford).
   - Attention: QKᵀ → softmax → ·V built from the optimized GEMM and softmax kernels, with the scale fused into the QKᵀ epilogue. The N×N score matrix is still materialized, and measuring that traffic is the point.
   - Each step goes on the roofline with a one-sentence explanation of why it moved.
3. **Analytical performance model.** Tiered (roofline → + occupancy & waves → + latency hiding). It is tuned on the 4070 Laptop, predicts the 4060 Laptop, and reports error per tier.
4. **End-to-end transformer block.** GPT-2-small shapes: measured time vs the sum of predictions vs PyTorch.
5. **Cache simulator (stretch).** CPU address replay of kernel access patterns into an L1/L2 model, cross-checked against ncu hit rates.

## Setup (WSL2 + Arch Linux)
**On Windows:**
1. Update the NVIDIA driver to the latest version (≥ 580, required by CUDA 13).
2. NVIDIA Control Panel → Desktop → *Enable Developer Settings*, then Developer → *Manage GPU Performance Counters* → **Allow access to all users**. Without this, `ncu` fails with `ERR_NVGPUCTRPERM`.
3. `wsl --shutdown`, then reopen Arch.

**In Arch WSL** (do not install a Linux NVIDIA driver; WSL provides `libcuda`):
```sh
pacman -Sy --noconfirm git
git clone <repo-url> ~/kernline && cd ~/kernline   # keep it on the Linux FS, not /mnt/*
bash tools/setup_arch.sh && source ~/.bashrc        # cuda, nsight-compute, cmake, ninja, python, uv + PATH
uv venv && source .venv/bin/activate
uv pip install torch numpy pandas matplotlib pytest

tools/smoke/run.sh   # go/no-go: builds, runs, and profiles a vector-add
```

## Build, test, profile
```sh
cmake -B build -G Ninja && cmake --build build      # libkernels.so, bench, block, ceilings
.venv/bin/python -m pytest tests/                    # every variant vs PyTorch + tooling + model tests
tools/ceilings/run.sh                                # measure this GPU's roofs → specs/<gpu>.json
./build/bench gemm v0 4096 4096 4096                 # CSV row: time, FLOP, bytes, clocks
tools/profile.sh gemm v0 4096 4096 4096              # ncu report + results/<gpu>/gemm.csv row
./build/block 2 1024                                 # transformer block, per step and total (D4)
tools/suite.sh                                       # everything above, every kernel and shape (D8)
.venv/bin/python analysis/roofline.py                # analysis/plots/<kernel>.png from results/
.venv/bin/python -m model validate --gpu rtx4060_laptop   # held-out model errors (analysis/model.md)
```
Run benchmarks and ceilings on an idle, plugged-in laptop: clocks follow the power limit, and every row records the clock it ran at.

## Repo layout
```
common/          CUDA helpers (error checks, timers, GPU info)
kernels/         gemm/ softmax/ layernorm/ attention/, one file per optimization step
src/api.cu       extern "C" launchers → libkernels.so (used by tests, bench, block)
bench/           benchmark CLI → CSV
block/           end-to-end transformer block runner
tests/           pytest correctness vs PyTorch (via ctypes)
tools/           smoke test, ncu wrapper, ceiling microbenchmarks
analysis/        roofline plots
model/           analytical performance model (Python)
sim/             trace-driven cache simulator (C++)
specs/           per-GPU measured ceilings
results/         committed measurements, per GPU
docs/            design + per-person plans
```

## Results

### Measured ceilings
*TBD: FP32 / FP16-TC peak, DRAM and shared-memory bandwidth, memory latency for both GPUs.*

### GEMM
*TBD: before/after table (time, GFLOP/s, % of peak, % of cuBLAS, occupancy, limiter) for v0 → v4.*

### Softmax & LayerNorm
*TBD: before/after table (time, GB/s, % of DRAM peak, % of PyTorch) for v0 → v2.*

### Attention
*TBD: naive vs optimized unfused vs PyTorch math path, DRAM bytes, sequence-length sweep, and why attention is memory-bound.*

### Roofline plots
*TBD: one chart per kernel family and GPU, each step annotated with why it moved.*

### Transformer block
*TBD: per-kernel time breakdown at GPT-2-small shapes vs PyTorch.*

### Performance model accuracy
*TBD: predicted vs measured per tier (T0/T1/T2), held-out GPU error, and where the model is wrong.*

### Cache simulator
*TBD: L1/L2 hit rate vs tiling, simulated vs ncu.*

## Who did what
*TBD*
