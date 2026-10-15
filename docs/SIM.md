# KernLine — Cache simulator design (`sim/`, Part B)

Design for branch #9 `b/sim` (PLAN-B weeks 7–8, D5). Read `docs/DESIGN.md` first.

## Purpose
**Explanatory, not predictive.** The simulator shows *why* tiling and launch order move the L1 and L2 hit rates of our kernels, and how close an address-level model gets to Nsight Compute. Its numbers don't feed the performance model (`model/`, Part A). The gap to ncu is reported and explained, not tuned away: roughly ±10 points on a hit rate is acceptable when the write-up can say where the rest comes from.

## Scope
Kernels replayed, each with every sub-kernel it launches:
- GEMM v0, v2, v3 (and `gemm_v3_batched`).
- Attention v0, v1, v2: `transpose_scale_kernel`, `scale_kernel` (v0), softmax v0 / v2⟨k⟩ / v3 as `src/api.cu` picks them, and the GEMMs. v2 touches the same addresses as v1 in a different block order, so v1 vs v2 isolates the effect of scheduling on L2.

Out of scope: shared memory and registers (not traced), GEMM v1/v4, LayerNorm, timing, and the scalar-load fallbacks of `gemm_v3` and softmax v3. Every validation shape takes the float4 paths, and the generators reject shapes that wouldn't.

## Architecture
One C++ binary, `build/cachesim`. Each trace is generated and consumed in the same process and never stored: GEMM v0 at 4096³ alone is ~4×10⁹ warp-level loads.

```
sim/
  sim.h, buffers.cpp, coalesce.cpp   WarpOp, Generator, Buffers, coalesce()
  cache.h/.cpp       sectored set-associative LRU cache (every L1 and the L2)
  memory.h/.cpp      L1 per SM + L2 + counters (MemorySystem)
  schedule.cpp       run(): blocks -> SMs, resident-block slots, round-robin warps and SMs
  launch_config.h/.cpp  resident blocks and carveout per launch (ncu-measured, or the fallback rule)
  generators/        gemm.cpp, rowwise.cpp (transpose, scale, softmax), attention.cpp
  main.cpp           build/cachesim <kernel> <variant> <dims> [options] -> CSV
  test.cpp           build/cachesim_test
  cases.txt          the validation cases
tools/sim_ncu.sh, sim_ncu.py   ncu reference counters + launch configs (the one GPU step)
analysis/cachesim.py   runs the sim, compares with ncu launch by launch, tables + plots
analysis/cachesim.md   write-up
tests/test_cachesim.py pytest over build/cachesim; build/cachesim_test for cache + coalescer
```

### Generator interface
A generator describes one kernel launch: grid, block size, resident blocks per SM, `ops_per_warp(block)`, and `op(block, warp, i)`, the i-th global memory instruction of that warp in program order. A `WarpOp` holds 32 lane addresses, an active mask, bytes per lane (4 or 16), and load or store. `op()` is closed-form index math with no stored state, so memory stays flat at any size, runs are deterministic, and a test can ask for one specific op and check it against the kernel source by hand.

Generators take tile parameters (v2: T; v3: BM, BN, BK, TM), so sweeps can include tiles the real kernels don't build. Only the real configuration is compared with ncu; the others are marked `simulated_only`.

### Launch order and concurrency ("waves, round-robin")
- Blocks go to SMs in `blockIdx` order (x fastest, then y, then z), up to the resident-blocks limit per SM. A finished block's slot takes the next block.
- Within an SM, resident warps issue one memory instruction each in turn. SMs take turns at the L2 the same way.
- A block finishes when its warps have issued all their ops. Time is counted in issued instructions, not cycles.
- Resident blocks per SM are the minimum of ncu's `launch__occupancy_limit_{blocks,registers,shared_mem,warps}` for that kernel, from its committed report. Simulated-only tiles use the occupancy rule (threads, registers, shared memory; limits from `specs/<gpu>.json`). The SM count (24 or 36) selects the GPU.

Composite calls (attention) are a list of launches in the real launch order: per-head GEMMs for v0/v1, one batched grid for v2. Caches start cold for every launch, as under ncu's default `--cache-control all`, which flushes before each profiled kernel.

## Cache model
One `Cache` class: 128 B lines, 32 B sectors, set-associative, LRU. A miss fills only the requested sectors. Associativity and the set-index function are parameters. Their defaults below are assumptions (NVIDIA doesn't publish them), named in the write-up and checked for sensitivity.

**L1, one per SM.**
- Size = 128 KB − the shared-memory carveout the driver chose for that kernel, read from the committed ncu report (`launch__shared_mem_config_size`). The driver doesn't pick the smallest carveout that fits. GEMM v3 needs 2 × 9.1 KB but runs with a 64 KB carveout (64 KB of L1) on both GPUs. GEMM v0 and v2 get 16 KB (112 KB of L1), `transpose_scale_kernel` 64 KB, softmax v3 100 KB (28 KB of L1). Simulated-only tiles keep their variant's real carveout, or the next size in {0, 8, 16, 32, 64, 100} KB if their shared memory doesn't fit; such rows say so.
- Default 4-way.
- Global loads allocate. Global stores write through to L2 without allocating, and invalidate a cached copy of their sectors.

**L2, one, shared by all SMs.**
- 32 MB, default 16-way, write-back, write-allocate per sector (a store fills its sector without reading DRAM).
- The set index is an XOR hash of the line address, standing in for the hardware's physical-address hash. Plain modulo would create conflicts the hardware doesn't have, e.g. P·V's rows of P, 16 KB apart.

**Addresses.** Buffers are placed as the bench allocates them: one `cudaMalloc` each, in allocation order, 2 MB-aligned. The cache model only sees addresses relative to these bases.

**DRAM bytes** = (L2 read-miss sectors + dirty sectors evicted during the launch) × 32 B. Dirty data still in L2 at the end of the launch isn't counted, matching ncu (e.g. `dram_write_bytes = 0` for GEMM v3 at 1024³).

## Comparison with ncu
**Reference counters (`tools/sim_ncu.sh`).** The committed reports have hit-rate percentages but not the absolute, per-operation counters. So the validation cases are profiled once more on the 4060, through the bench's `kl_profile` range at base clocks, collecting:
- L1: `l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st}.sum`, `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum`, `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}_lookup_hit.sum`.
- L2: `lts__t_sectors_srcunit_tex_op_{read,write}.sum`, `lts__t_sectors_srcunit_tex_op_{read,write}_lookup_hit.sum`.
- `dram__bytes_{read,write}.sum`.

They're stored as `results/sim/ncu_<gpu>.csv`, one row per launch (kernel name in launch order). Because the counters are split by operation, hit rates are compared like for like, with no guessing about how ncu's combined `*_hit_rate.pct` metrics count stores. The 4070 reference is optional: the script runs unchanged on Arjun's GPU.

**Generator check (exact).** The number of global requests and sectors depends only on the index math, not on any cache assumption. For every validated launch, the simulator's load and store request and sector counts must equal ncu's **exactly**. This runs before any hit rate is compared, so a wrong generator can't be mistaken for a cache effect.

**Hit rates and DRAM bytes** are then compared launch by launch: L1 load hit rate, L2 read and write hit rate, DRAM read and write bytes. The gap is reported per kernel with its likely cause (associativity, hash, replacement policy, scheduling order, prefetch).

## Experiments
1. **Validation** at the profiled shapes, 4060 (and 4070 if its reference exists):
   - GEMM v0 at 1024³ and 2048³ (4096³ is too slow to simulate).
   - GEMM v2 and v3 at 1024³, 2048³ and 4096³.
   - Attention v1 and v2 at 12×N×64 for N = 256 … 4096, and v0 for N ≤ 2048 (its `gemm_v0` launches at N = 4096 are ~1.6×10⁹ warp loads).
2. **Tile sweeps** at 2048³: v2 T ∈ {8, 16, 32}; v3 BM = BN ∈ {64, 128} × BK ∈ {8, 16, 32} (TM = 8; a 32×32 tile would be 16 threads, under one warp). L1 and L2 hit rates and DRAM bytes vs tile, on 24 and 36 SMs.
3. **Scheduling:** attention v1 vs v2, same addresses in a different block order. Does batching the heads change L2 reuse of Kᵀ and V?
4. **P·V:** sectors per request and L2 behaviour of the P loads, the simulator's side of why P·V reads DRAM at ~85 GB/s (`analysis/attention.md`).

## Outputs
- `results/sim/<gpu>.csv`, one row per (kernel, variant, dims, launch): the simulator's raw counters (requests, sectors, hit sectors, DRAM bytes) next to ncu's (`ncu_*`), plus load counters for the first input operand (`a_*`, e.g. P in P·V). Hit rates are derived from the counters by `analysis/cachesim.py report`. Tile sweeps go to `results/sim/sweep.csv`.
- `analysis/cachesim.md` and `analysis/plots/cachesim_*.png`.

## Testing
- `build/cachesim_test` (C++, plain asserts, run from pytest): LRU order and eviction on a tiny cache, sector fill and write-allocate, dirty eviction counting, store invalidation in L1, the carveout rule, coalescing (32 consecutive floats → 4 sectors in 1 line; 32 floats K apart → 32 sectors).
- `tests/test_cachesim.py`: `--dump` addresses for hand-picked ops of each generator (e.g. v3 block (1, 2), warp 3, op 10) against the index math in `kernels/`; the exact request and sector match against `results/sim/ncu_rtx4060_laptop.csv` for every validated launch; the resident blocks and carveout used per launch against the ncu reports.
- Tests need no GPU. Only `tools/sim_ncu.sh` does.

## Commits on `b/sim`
1. This design doc.
2. Cache, coalescer and `cachesim_test`.
3. Scheduler, GEMM generators, `tools/sim_ncu.sh` with the 4060 reference, and the exact count check.
4. Attention and softmax generators.
5. Validation run and `analysis/cachesim.py`.
6. Tile sweeps, scheduling and P·V experiments, and `analysis/cachesim.md`.
