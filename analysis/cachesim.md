# What the cache simulator shows

The cache simulator (`sim/`, design in `docs/SIM.md`) replays GEMM v0, v2 and v3 and unfused attention v0, v1 and v2 address by address. That covers the 22 calls in `sim/cases.txt`, 266 kernel launches on the RTX 4060 Laptop (24 SMs). Blocks run in waves on the SMs, warps issue round-robin, and every load and store goes through a sectored L1 per SM and one shared L2. Its global request and sector counts equal Nsight Compute's exactly on every launch. Its L2 read hit rate is within 2.8 points of ncu's on the 18 calls outside attention v0, and its L2 read misses are within 4% of ncu's for every kernel. Where a hit rate differs by more than 5 points, the simulator and the hardware either catch the same reuse in different caches (L1 vs L2), or ncu counts something else (L2 write hits).

Data: `results/sim/rtx4060_laptop.csv` (simulator and ncu counters per launch), `results/sim/sweep.csv`, `results/sim/rtx4070_laptop.csv`. Tables: `analysis/cachesim.py report`.

## Validation against ncu

| call | launches | counts exact | L1 ld hit sim / ncu (%) | L2 rd hit sim / ncu (%) | L2 wr hit sim / ncu (%) | DRAM rd MB sim / ncu (approx.) | DRAM wr MB sim / ncu (approx.) |
|---|---|---|---|---|---|---|---|
| gemm v0 1024x1024x1024 | 1 | yes | 87.6 / 97.1 | 99.6 / 98.5 | 75.0 / 99.9 | 8.4 / 16.9 | 0.0 / 4.1 |
| gemm v0 2048x2048x2048 | 1 | yes | 87.8 / 97.0 | 99.8 / 99.2 | 75.0 / n/a* | 33.6 / 94.7 | 8.4 / 10.5 |
| gemm v2 1024x1024x1024 | 1 | yes | 4.4 / 0.0 | 96.7 / 96.9 | 0.0 / 100.0 | 8.4 / 8.4 | 0.0 / 0.0 |
| gemm v2 2048x2048x2048 | 1 | yes | 0.0 / 0.0 | 98.4 / 98.4 | 0.0 / 100.0 | 33.6 / 51.1 | 8.4 / 10.3 |
| gemm v2 4096x4096x4096 | 1 | yes | 0.0 / 0.0 | 50.3 / 49.7 | 0.0 / n/a* | 8533.9 / 8760.5 | 66.8 / 81.5 |
| gemm v3 1024x1024x1024 | 1 | yes | 18.8 / 7.4 | 84.6 / 86.9 | 0.0 / 100.0 | 8.4 / 8.4 | 0.0 / 0.0 |
| gemm v3 2048x2048x2048 | 1 | yes | 0.0 / 4.3 | 93.8 / 96.0 | 0.0 / 100.0 | 33.6 / 33.6 | 7.3 / 9.1 |
| gemm v3 4096x4096x4096 | 1 | yes | 0.0 / 0.4 | 64.3 / 63.2 | 0.0 / 100.0 | 1533.0 / 1575.2 | 65.0 / 65.6 |
| attention v0 12x256x64 | 27 | yes | 93.5 / 97.1 | 89.2 / 75.9 | 89.5 / 99.7 | 12.6 / 12.8 | 0.0 / 0.0 |
| attention v0 12x512x64 | 27 | yes | 91.5 / 97.1 | 92.8 / 78.6 | 90.8 / n/a* | 44.0 / 44.3 | 0.0 / 1.3 |
| attention v0 12x1024x64 | 27 | yes | 89.3 / 97.0 | 91.4 / 70.8 | 91.5 / n/a* | 264.2 / 281.6 | 33.6 / 70.2 |
| attention v0 12x2048x64 | 27 | yes | 89.5 / 96.3 | 91.4 / 75.4 | 91.9 / 100.0 | 1031.8 / 1099.0 | 335.5 / 408.6 |
| attention v1 12x256x64 | 26 | yes | 0.0 / 0.0 | 20.0 / 20.0 | 40.0 / 100.0 | 9.4 / 9.8 | 0.0 / 0.0 |
| attention v1 12x512x64 | 26 | yes | 0.0 / 0.0 | 31.0 / 31.0 | 44.4 / 100.0 | 31.5 / 32.6 | 0.0 / 0.5 |
| attention v1 12x1024x64 | 26 | yes | 5.3 / 4.8 | 33.3 / 33.6 | 47.1 / 100.0 | 113.2 / 113.6 | 16.8 / 34.5 |
| attention v1 12x2048x64 | 26 | yes | 0.0 / 1.6 | 39.8 / 38.5 | 48.5 / 97.8 | 427.8 / 437.3 | 167.8 / 213.8 |
| attention v1 12x4096x64 | 26 | yes | 7.0 / 2.9 | 36.8 / 39.6 | 49.2 / 99.8 | 1664.1 / 1662.4 | 1194.0 / 1433.5 |
| attention v2 12x256x64 | 4 | yes | 0.0 / 0.0 | 20.0 / 20.0 | 40.0 / 100.0 | 9.4 / 9.5 | 0.0 / 0.0 |
| attention v2 12x512x64 | 4 | yes | 0.0 / 1.8 | 31.0 / 29.8 | 44.4 / 100.0 | 31.5 / 36.0 | 0.0 / 0.0 |
| attention v2 12x1024x64 | 4 | yes | 4.7 / 0.9 | 33.7 / 36.2 | 47.1 / 100.0 | 113.2 / 115.6 | 37.4 / 73.5 |
| attention v2 12x2048x64 | 4 | yes | 0.0 / 0.6 | 39.8 / 39.1 | 48.5 / 100.0 | 427.8 / 428.2 | 342.2 / 386.6 |
| attention v2 12x4096x64 | 4 | yes | 5.8 / 2.5 | 37.6 / 39.5 | 49.2 / 99.4 | 1664.9 / 1675.9 | 1556.0 / 1606.3 |

\* ncu collected these counters over several replay passes; for this call a hit count exceeds its total, so the measured rate isn't meaningful.

Global load and store requests and sectors match ncu exactly on all 266 launches, with no conversion factor. They depend only on the index math, so the generators are right, and every gap below comes from the cache model or from ncu's counters.

**ncu's replay passes.** ncu collected the counters over 4 replay passes, so a hit count and its total can come from different runs. In 46 of gemm_v0's 98 launches a hit count exceeds its total. All 46 are on L2 writes, and one of them also on L2 reads. One gemm_v2 launch does the same on L2 writes. The tables mark a call `n/a*` when its summed hit count exceeds its summed total. `cachesim_validation.png` leaves such a call out of that panel. No call is affected for L1 or L2 read rates. A rate whose hit count is below its total can still mix passes, so the kernels with free-running blocks (GEMM v0, v2) carry this caveat throughout.

- **L2 read hit rate:** within 2.8 points on all 18 calls outside attention v0 (worst: attention v1 at N = 4096, 36.8% vs 39.6%). Attention v0 is 13–21 points high (91.4% vs 70.8% at N = 1024).
- **L1 load hit rate:** within 4.4 points on 16 calls. GEMM v0 is 9.2–9.5 points low, attention v0 3.6–7.7 points low, and GEMM v3 at 1024³ 11.4 points high (18.8% vs 7.4%).
- **L2 write hit rate:** 0–92% simulated, 97.8–100% measured. That is a definition difference, below.
- **DRAM reads:** within 5% on 16 calls. The other 6 are in the DRAM table below.

`analysis/plots/cachesim_validation.png` plots the per-call L1 and L2 read rates against ncu. Every point sits on the diagonal except the three labelled groups: GEMM v0 and attention v0 below it on L1, GEMM v3 at 1024³ above it on L1, and attention v0 above it on L2.

## Where the gap comes from

Per kernel, every launch of a kernel in the 22 calls summed:

| kernel | launches | L1 ld hit sim / ncu (%) | L2 rd hit sim / ncu (%) | L2 rd misses sim / ncu (M sectors) | L2 wr hit sim / sim if full-sector writes hit / ncu (%) | launches where ncu L2 wr hits = store sectors |
|---|---|---|---|---|---|---|
| gemm_v0_kernel | 98 | 90.5 / 97.3 | 98.8 / 96.0 | 10.77 / 10.36 | 75.0 / 75.0 / n/a* | 0 of 98 |
| gemm_v2_kernel | 3 | 0.1 / 0.0 | 56.2 / 55.6 | 268.00 / 271.86 | 0.0 / 100.0 / n/a* | 2 of 3 |
| gemm_v3_kernel | 133 | 3.8 / 2.2 | 60.8 / 61.0 | 121.06 / 122.25 | 0.0 / 100.0 / 99.4 | 133 of 133 |
| transpose_scale_kernel | 14 | 0.0 / 0.0 | 0.0 / 0.0 | 1.89 / 1.89 | 0.0 / 100.0 / 100.0 | 14 of 14 |
| scale_kernel | 4 | 0.0 / 0.0 | 0.0 / 0.0 | 8.36 / 8.36 | 100.0 / 100.0 / 100.0 | 4 of 4 |
| softmax_v0_kernel | 4 | 0.0 / 73.3 | 88.0 / 54.8 | 24.08 / 24.17 | 100.0 / 100.0 / 100.0 | 4 of 4 |
| softmax_v2_kernel<2> | 2 | 0.0 / 0.0 | 0.0 / 0.0 | 0.20 / 0.20 | 100.0 / 100.0 / 100.0 | 2 of 2 |
| softmax_v2_kernel<4> | 2 | 0.0 / 0.0 | 0.0 / 0.0 | 0.79 / 0.79 | 100.0 / 100.0 / 100.0 | 2 of 2 |
| softmax_v2_kernel<8> | 2 | 0.0 / 0.0 | 0.0 / 0.0 | 3.15 / 3.15 | 100.0 / 100.0 / 100.0 | 2 of 2 |
| softmax_v2_kernel<16> | 2 | 0.0 / 0.0 | 0.0 / 0.0 | 12.58 / 12.80 | 100.0 / 100.0 / 97.7 | 2 of 2 |
| softmax_v3_kernel<1> | 2 | 0.0 / 0.0 | 0.0 / 0.0 | 50.33 / 50.33 | 100.0 / 100.0 / 100.0 | 2 of 2 |

L2 read misses agree within 4% for every kernel, even where the hit rates are 30 points apart. So the simulator sends the same sectors to DRAM as the hardware does. What differs is where it catches the reuse.

- **GEMM v0, L1 (87.6% vs 97.1%): L1 associativity and set index.** ncu's numbers here are self-consistent: every launch has fewer L1 hits than sectors. At 1024³ the measured 97.06% is exactly the rate if each block fetches each of its A and B sectors once. Per k-step a block issues 136 sectors (8 warps × (16 rows of A + 1 sector of B)), and 4 of them are new, so 132/136 = 97.06% hit. The simulator loses about 10 points to conflict misses. A warp's 16 rows of A are 4 KB apart, i.e. 32 lines. In a 112 KB, 4-way L1 with set = line mod 224, those rows fall into only 7 sets, 28 ways in total. The 6 resident blocks keep 96 rows of A live. At 8 ways that is still 56 slots (87.7%). At 16 ways it is 112 slots, and the L1 rate rises to 97.8%, 0.8 points above ncu (Assumptions table). The real L1 probably hashes its set index or is more associative. The defaults stay. In attention v0, the per-head P·V GEMMs read P, whose rows are 4N bytes apart, so the same conflicts appear as N grows. Together with softmax v0 (next bullet), the whole-call L1 gap grows from 3.6 points at N = 256 to 7.7 at N = 1024.
- **softmax v0, L1 (0.0% vs 73.3%): scheduling order.** Each thread walks one row, so a warp's load touches 32 rows (32 sectors per request). A sector serves 8 consecutive loads from its lane, so the best possible L1 rate is 7/8. ncu measures exactly 87.5% at N ≤ 512, falling to 69.5% at N = 2048 (per launch, `results/sim/rtx4060_laptop.csv`). The simulator issues one instruction per warp per turn across 48 resident warps (12 blocks of 4). Between two loads of the same sector, the other 47 warps bring in 1,504 lines, twice the 768 lines of the 96 KB L1. So every reuse misses. Raising the L1 to 32 ways leaves it at 0.0%, which rules out associativity. The hardware evidently runs one warp's consecutive loads closer together, for example because the compiler unrolled the loop and the scheduler favours one warp. The model doesn't do that. Not tested further.
- **Attention v0, L2 read (about 91% vs 71–79%): the two L1 gaps above, seen from L2.** The reuse the simulator misses in L1 reaches L2 and hits there. softmax v0's L2 read rate is 88.0% simulated vs 54.8% measured, with 24.08 M vs 24.17 M misses. DRAM reads agree within 7% (264.2 vs 281.6 MB at N = 1024).
- **GEMM v3 at 1024³, L1 (18.8% vs 7.4%): block placement and lockstep issue.** The grid is 8 × 8 tiles with 2 resident blocks per SM. Block b and block b + 24 share an SM, and 24 = 3 grid rows, so they read the same column panel of B. The simulator advances both blocks in lockstep, so the second one's B loads hit in L1. Hardware blocks drift apart and aren't placed strictly by b mod 24. At 2048³ the error goes the other way: block b + 24 shares no panel in the simulator (0.0%), while the hardware gets 4.3%. On 36 SMs the simulator gives 0.0% at 1024³ too.
- **L2 write hits (all kernels): a definition difference, not a cache effect.** ncu reports 97.7–100% write hits for every kernel with consistent counters. That includes gemm_v3 and transpose_scale_kernel, whose outputs (C, Kᵀ) no kernel has touched since ncu flushed the caches. In all 133 gemm_v3 launches, all 14 transpose launches and every softmax launch, ncu's write-hit count equals the launch's store sectors exactly. That holds for 167 of the 168 launches outside gemm_v0; the exception is the gemm_v2 launch marked `n/a*`. Where ncu's rate is below 100% (gemm_v3 99.4%, softmax_v2⟨16⟩ 97.7%), the hit count still equals the store sectors. The misses there are write sectors beyond the kernel's own stores, from a source we didn't identify. So ncu counts a full-sector write to an absent sector as a hit, which fits write-allocate without a DRAM fill. The simulator counts the same write as a miss (it still allocates without reading DRAM). Under the alternative definition (a full-sector write hits), the simulator gives 100.0% for every kernel except gemm_v0. The data can't settle partial-sector writes. Only gemm_v0 (8 B per sector) and softmax v0 (4 B per sector, overwriting scores it has just read, a hit in both) make them, and gemm_v0's write totals differ from its exact store sectors in all 98 launches. The default model is unchanged.
- **DRAM reads: mostly in ncu's own counters.** In the 6 calls where the simulator's DRAM reads and ncu's differ by more than 5%, ncu's DRAM bytes also differ from ncu's own L2 read misses × 32 B. The simulator agrees with the latter to within 7%:

| call | sim DRAM rd MB | ncu dram__bytes_read MB | ncu L2 read misses × 32 B (MB) |
|---|---|---|---|
| gemm v0 1024x1024x1024 | 8.4 | 16.9 | 8.3 |
| gemm v0 2048x2048x2048 | 33.6 | 94.7 | 35.9 |
| gemm v2 2048x2048x2048 | 33.6 | 51.1 | 33.6 |
| gemm v3 2048x2048x2048 | 33.6 | 33.6 | 20.1 |
| attention v0 12x1024x64 | 264.2 | 281.6 | 253.6 |
| attention v0 12x2048x64 | 1031.8 | 1099.0 | 1029.8 |
| attention v2 12x512x64 | 31.5 | 36.0 | 31.4 |

  The extra DRAM reads in GEMM v0 (8.6 MB at 1024³, 58.8 MB at 2048³) are unexplained. Candidates are fills for its partial-sector stores and pass-to-pass variation; neither was tested. GEMM v3 at 2048³ goes the other way. A cold L2 must fetch all of A and B (33.6 MB), which caps the read hit rate at 93.75%; the simulator sits exactly there. ncu's L2 reports only 20.1 MB of misses (96.0% hit) while its DRAM reads are 33.6 MB. That is unexplained too.
- **DRAM writes:** the simulator's are never above ncu's. They are lower in 16 of 22 calls, for example 37.4 vs 73.5 MB for attention v2 at N = 1024, and equal (both 0) in the other 6. The simulator writes dirty sectors back only when they are evicted. The hardware probably writes some back earlier. Not tested.

## Tiling moves the hit rates

GEMM 2048³ at every tile, simulated (`results/sim/sweep.csv`). Rows marked "no" are tiles the real kernels don't build. Resident blocks and L1 size are the same on both SM counts.

| tile | real kernel | resident blocks / SM | L1 KB | loads (M sectors) | L1 ld hit 24 / 36 SMs (%) | L1 → L2 reads MB 24 / 36 SMs | L2 rd hit 24 / 36 SMs (%) | DRAM rd MB 24 / 36 SMs |
|---|---|---|---|---|---|---|---|---|
| v2 8 | no | 16 | 96 | 268.4 | 43.8 / 40.6 | 4831 / 5099 | 99.3 / 99.3 | 33.6 / 33.6 |
| v2 16 | no | 4 | 112 | 134.2 | 31.8 / 27.2 | 2927 / 3128 | 98.9 / 98.9 | 33.6 / 33.6 |
| v2 32 | yes | 1 | 112 | 67.1 | 0.0 / 0.0 | 2147 / 2147 | 98.4 / 98.4 | 33.6 / 33.6 |
| v3 64x64x8x8 | no | 8 | 64 | 33.6 | 23.4 / 0.0 | 822 / 1074 | 95.9 / 96.9 | 33.6 / 33.6 |
| v3 64x64x16x8 | no | 8 | 28 | 33.6 | 35.9 / 0.0 | 688 / 1074 | 95.1 / 96.9 | 33.6 / 33.6 |
| v3 64x64x32x8 | no | 5 | 28 | 33.6 | 19.5 / 0.0 | 864 / 1074 | 96.1 / 96.9 | 33.6 / 33.6 |
| v3 128x128x8x8 | yes | 2 | 64 | 16.8 | 0.0 / 0.0 | 537 / 537 | 93.8 / 93.8 | 33.6 / 33.6 |
| v3 128x128x16x8 | no | 2 | 64 | 16.8 | 0.0 / 0.0 | 537 / 537 | 93.8 / 93.8 | 33.6 / 33.6 |
| v3 128x128x32x8 | no | 2 | 28 | 16.8 | 0.0 / 0.0 | 537 / 537 | 93.8 / 93.8 | 33.6 / 33.6 |

- **DRAM reads don't change.** Every tile on both SM counts reads A and B from DRAM exactly once (33.6 MB). At 2048³ each row panel of A and column panel of B stays in the 32 MB L2 between its uses, so no tile re-fetches them.
- **Tiling cuts traffic into L2, and that lowers the L2 hit rate.** Each doubling of the tile edge halves the loads (268.4 M sectors at v2 T = 8, 16.8 M at v3 128). L1 → L2 reads fall 9×, from 4831 to 537 MB. The 1.05 M compulsory misses stay fixed, so they become a larger share, and the L2 read rate falls from 99.3% to 93.8%. Here a lower L2 hit rate means less redundant traffic, not worse caching.
- **BK changes nothing that reaches L2 for the 128 tiles.** It only changes the carveout (64 KB vs 28 KB of L1), and these tiles get no L1 hits at any BK.
- **L1 hits depend on which blocks share an SM.** v2 at T = 32 runs 1 block per SM and loads each element once per block, so its L1 rate is 0.0%. Small tiles get 19.5–43.8% from co-resident blocks that read the same panels. That reuse depends on the grid width modulo the SM count. With 64-wide tiles the grid is 32 blocks wide, and the first wave puts blocks b, b + 24, b + 48, … on one SM. On 24 SMs, co-resident blocks 24 apart can sit in the same grid row and share a row panel of A. Blocks 96 apart (3 grid rows) share a column panel of B. On 36 SMs co-resident blocks are 36k apart. That is never less than one grid row, and never a multiple of 32 for k < 8, so no two share a panel and the L1 rate drops to 0.0%. The 36-SM bars for these tiles carry 1074 MB into L2 instead of 688–864 MB. The real kernels, v2 T = 32 and v3 128x128x8x8, get no L1 reuse at either SM count. ncu agrees: 0.0% and 4.3%.

`analysis/plots/cachesim_tiles.png` shows the same: L1 hit rate, L2 read hit rate and L1 → L2 read traffic per tile, for 24 and 36 SMs.

### The 4070 Laptop (36 SMs, simulated only)

`analysis/cachesim.py validate --gpu rtx4070_laptop` replays the same 22 calls on 36 SMs with the 4060's measured launch configs. There is no ncu reference yet, so these are predictions:

| call | L1 ld hit 24 / 36 SMs (%) | L2 rd hit 24 / 36 SMs (%) | DRAM rd MB 24 / 36 SMs |
|---|---|---|---|
| gemm v2 4096x4096x4096 | 0.0 / 0.0 | 50.3 / 54.3 | 8533.9 / 7857.9 |
| gemm v3 1024x1024x1024 | 18.8 / 0.0 | 84.6 / 87.5 | 8.4 / 8.4 |
| gemm v3 4096x4096x4096 | 0.0 / 0.0 | 64.3 / 75.2 | 1533.0 / 1063.3 |
| attention v2 12x4096x64 | 5.8 / 0.0 | 37.6 / 41.2 | 1664.9 / 1665.0 |

At 4096³, B (67 MB) doesn't fit in L2, so GEMM v3 re-reads it from DRAM once for each group of grid rows that run together. 24 SMs × 2 blocks cover 1.5 of the 32 grid rows at a time, and 36 SMs cover 2.25. If A is read once, the simulator reads B about 22 times on 24 SMs and 15 times on 36 SMs, close to 32 / 1.5 = 21 and 32 / 2.25 = 14. That is 31% less DRAM traffic on the 4070, from the extra SMs alone. On the 4060, ncu measures 1575 MB, 2.7% above the simulator.

## Launch order: attention v1 vs v2

| N | phase | L2 rd hit v1 sim / ncu (%) | L2 rd hit v2 sim / ncu (%) | DRAM rd MB v1 / v2 (sim) |
|---|---|---|---|---|
| 256 | QK | 50.0 / 50.0 | 50.0 / 50.0 | 1.6 / 1.6 |
| 256 | PV | 16.7 / 16.7 | 16.7 / 16.7 | 3.9 / 3.9 |
| 512 | QK | 75.0 / 75.0 | 75.0 / 73.4 | 3.1 / 3.1 |
| 512 | PV | 25.0 / 25.0 | 25.0 / 25.0 | 14.2 / 14.2 |
| 1024 | QK | 84.6 / 84.8 | 85.0 / 86.8 | 6.3 / 6.3 |
| 1024 | PV | 29.2 / 29.2 | 29.2 / 28.6 | 53.5 / 53.5 |
| 2048 | QK | 93.8 / 93.7 | 93.8 / 92.4 | 12.6 / 12.6 |
| 2048 | PV | 31.2 / 31.2 | 31.2 / 31.1 | 207.6 / 207.6 |
| 4096 | QK | 96.4 / 96.8 | 96.4 / 96.8 | 25.2 / 25.2 |
| 4096 | PV | 25.8 / 32.1 | 28.0 / 31.5 | 821.1 / 821.9 |

Batching the heads doesn't change L2 reuse of Kᵀ or V. The simulated v1 and v2 rates differ by at most 0.4 points up to N = 2048 and by 2.2 points at N = 4096. The measured rates differ by at most 2.0 points. Simulated DRAM reads are identical up to N = 2048 and 0.1% apart at N = 4096. The batched grid runs head by head (gridDim.z is the slowest index), so the resident blocks almost always come from one head, like the per-head launches. The heads share no Kᵀ or V, so there is nothing for batching to reuse. v2's 7.9× speedup over v1 at N = 256 (`analysis/attention.md`) is occupancy, not caching.

The simulator also reproduces the phase rates themselves: Q·Kᵀ's within 1.8 points at every N, and P·V's within 0.6 points up to N = 2048. At N = 4096 the simulator's P·V rate is 6.3 (v1) and 3.5 (v2) points below ncu's. The validation table points to L1. At N = 4096 the simulator's whole-call L1 rate is 7.0% and 5.8% vs ncu's 2.9% and 2.5%, so the simulator catches in L1 some of V's reuse that the hardware serves from L2.

## P·V from the cache side

P's loads in the P·V GEMM (operand A), attention v2:

| N | sectors / request | L1 hit % | L2 hit % of L1 misses | ncu L2 rd hit % (whole launch) |
|---|---|---|---|---|
| 256 | 16.0 | 0.0 | 0.0 | 16.7 |
| 512 | 16.0 | 0.0 | 0.0 | 25.0 |
| 1024 | 16.0 | 0.0 | 0.0 | 28.6 |
| 2048 | 16.0 | 0.0 | 0.0 | 31.1 |
| 4096 | 16.0 | 0.0 | 0.0 | 31.5 |

- **Each request reads one 32 B sector from each of 16 lines.** A warp's 32 float4 loads of P cover 16 rows × 32 B. In `gemm_v3`'s tile fill, thread t loads row t/2 of the 128-row A tile. So each request is 512 B in 16 sectors, each in its own 128 B line, and those lines are 4·N bytes apart (16 KB at N = 4096). A block's 8 warps together read one sector from each of the tile's 128 rows per k-step.
- **Every P sector comes from DRAM exactly once.** The P loads get 0.0% in L1 and 0.0% in L2. P·V's output is N × 64, narrower than one 128-wide tile, so each row panel of P is loaded by exactly one block. The whole-launch L2 hits ncu measures (16.7–31.5%) are V's.
- **P·V moves the minimum bytes.** Its simulated DRAM reads at N = 4096 are 821.9 MB, 0.5% above the size of P and V (805.3 + 12.6 MB).

The simulator confirms the access pattern that `analysis/attention.md` blames for P·V's DRAM throughput. Each request takes one 32 B sector from each of 16 lines, 4·N bytes apart, and each sector of P is fetched from DRAM once. It rules out one other cause: at 32 B sector granularity there is no redundant or over-fetched traffic. No sector is read twice, and every fetched sector is used in full.

What it can't test is whether that pattern is slow at DRAM. The model has no timing and no DRAM model, so it doesn't see row-buffer locality. That is where `analysis/attention.md`'s P·V paragraph places the limit: about 85 GB/s, 34% of DRAM bandwidth, on the 4060, and 77 GB/s on the 4070. The simulator is consistent with that explanation but doesn't measure it.

## Assumptions

The defaults: L1 4-way with the set index = line mod set count; L2 32 MB, 16-way, XOR-hashed set index; LRU everywhere; caches cold at every launch; L1 write-through without allocation, with stores invalidating their sectors; L2 write-back and write-allocate per sector without a DRAM fill; buffers 2 MB-aligned in allocation order. NVIDIA doesn't publish associativity, hash or replacement. The runs below change one assumption at a time to show how much it matters. They are not fits: the defaults stay.

| run | setting | L1 ld hit (%) | L2 rd hit (%) | DRAM rd / wr MB |
|---|---|---|---|---|
| gemm v3 2048³ | defaults | 0.00 | 93.75 | 33.6 / 7.3 |
| gemm v3 2048³ | L2 modulo index | 0.00 | 93.75 | 33.6 / 7.3 |
| gemm v3 2048³ | L2 8-way | 0.00 | 93.75 | 33.6 / 8.4 |
| gemm v3 2048³ | L2 32-way | 0.00 | 93.75 | 33.6 / 7.3 |
| gemm v3 2048³ | L1 8-way | 0.00 | 93.75 | 33.6 / 7.3 |
| attention v2 12x4096x64 | defaults | 5.79 | 37.58 | 1664.9 / 1556.0 |
| attention v2 12x4096x64 | L2 modulo index | 5.79 | 37.57 | 1665.1 / 1556.0 |
| attention v2 12x4096x64 | L2 8-way | 5.79 | 37.58 | 1664.9 / 1556.0 |
| attention v2 12x4096x64 | L2 32-way | 5.79 | 37.57 | 1665.1 / 1556.1 |
| attention v2 12x4096x64 | L1 8-way | 5.78 | 37.59 | 1664.9 / 1556.0 |
| gemm v0 1024³ | defaults (L1 4-way) | 87.56 | 99.63 | 8.4 / 0.0 |
| gemm v0 1024³ | L1 8-way | 87.70 | 99.63 | 8.4 / 0.0 |
| gemm v0 1024³ | L1 16-way | 97.84 | 97.88 | 8.4 / 0.0 |
| gemm v0 1024³ | L1 32-way | 97.84 | 97.88 | 8.4 / 0.0 |
| attention v0 12x256x64, softmax v0 launches | L1 4, 8, 16 or 32-way | 0.00 | 95.83 | – |

- **L2 associativity and set hash:** no effect on the real GEMM and attention at these shapes. Hit rates move by at most 0.01 points and DRAM reads by at most 0.2 MB. The one exception is GEMM v3's DRAM writes at 8 ways (7.3 → 8.4 MB). Their L2 misses are compulsory or come from streams far larger than 32 MB, so which set a line maps to doesn't decide whether it misses. The XOR hash was chosen against conflicts between P's rows 16 KB apart. Modulo indexing doesn't create them at N = 4096 with 16 ways.
- **L1 associativity:** the only assumption that moves a validated number. It doesn't affect GEMM v3 or attention v2 (0.01 points at 8 ways). It decides GEMM v0's L1 gap (87.7% at 8 ways, 97.8% at 16). It doesn't explain softmax v0's (0.0% at every associativity).
- **Not varied:** LRU, the L1 write policy and the buffer alignment. The cold start isn't an assumption: it matches ncu's `--cache-control all`.
- **Issue order** (one memory instruction per warp per turn, SMs in lockstep) is the assumption the data questions most. It accounts for the softmax v0 L1 gap and the GEMM v3 L1 gaps, and the simulator has no switch to test it.
