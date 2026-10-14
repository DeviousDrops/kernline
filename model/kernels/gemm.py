"""GEMM v0-v4 (kernels/gemm/), C (M×N) = A (M×K) · B (K×N), dims MxNxK.

Per variant: the C tile one block computes (BM×BN), the K step between __syncthreads (BK),
threads per block, which grid axis walks the N tiles (raster), and the input element size.
Counts below are per warp, from each kernel's index math:

  v0  block 16×16, threadIdx.x walks rows. A warp spans 16 rows × 2 columns, so each k reads 16
      distinct lines of A (16 L1 wavefronts) and one line of B: 17 wavefronts per k.
  v1  block 32×8, a warp is one row of 32 columns: A broadcasts (1 wavefront), B is one line.
  v2  32×32 tiles in shared memory: per k one broadcast As load and one Bs load (2 wavefronts),
      plus 2 tile stores per 32 k.
  v3  128×128 tile, 8×8 per thread: per k two broadcast float4 As loads (1 wavefront each) and two
      float4 Bs loads over 64 consecutive words (2 each), plus 4 scalar and 1 float4 tile stores
      per 8 k: 7 wavefronts in 4.6 instructions per k.
  v4  WMMA 128×128 tile, 8 warps of 32×64: per 16 k, 2 A and 4 B fragment loads of 512 B
      (4 wavefronts each), plus the tile stores: 2 wavefronts per k.

DRAM comes from the L2 panel model (model/l2.py). v0 and v3 read A in isolated 32 B pieces per
row (v0: 16 rows per warp access; v3: a 128×8 tile per step), so their A misses count as narrow
DRAM traffic. L1<->L2 traffic is each block's two panels
(threads of one block share them through L1 or shared memory) plus C. FLOPs are what the grid
executes: v2-v4 compute their whole tile over K rounded up to BK, so ragged edges and narrow
outputs (attention's P·V, N = 64 in a 128-wide tile) cost full tiles.
"""
import math

from model import l2
from model.launch import Launch
from model.occupancy import occupancy
from model import resources

VARIANTS = ("v0", "v1", "v2", "v3", "v4")
# variant: (BM, BN, BK, threads, raster, input bytes)
TILES = {
    "v0": (16, 16, 1, 256, "m", 4),
    "v1": (8, 32, 1, 256, "n", 4),
    "v2": (32, 32, 32, 1024, "n", 4),
    "v3": (128, 128, 8, 256, "n", 4),
    "v4": (128, 128, 64, 256, "n", 2),
}
KERNEL_NAMES = {
    "v0": ["gemm_v0_kernel"], "v1": ["gemm_v1_kernel"], "v2": ["gemm_v2_kernel"],
    "v3": ["gemm_v3_kernel", "gemm_v3_scalar_kernel"], "v4": ["gemm_v4_kernel", "gemm_v4_scalar_kernel"],
}
# Per warp per k: (L1 wavefronts, shared wavefronts, global instructions, shared instructions).
PER_K = {
    "v0": (17, 0, 2, 0),
    "v1": (2, 0, 2, 0),
    "v2": (2 / 32, 2 + 2 / 32, 2 / 32, 2 + 2 / 32),
    "v3": (20 / 8, 6 + 8 / 8, 2 / 8, 4 + 5 / 8),
    "v4": (32 / 64, 1.5 + 0.5, 8 / 64, 6 * 2 / 16 + 8 / 64),
}
NARROW_A = ("v0", "v3")  # A misses arrive as isolated 32 B sectors
# Bytes one warp has in flight to memory (T2): v0/v1 about 4 loads ahead in the unrolled k loop,
# v2-v4 the tile loads issued before __syncthreads.
MLP = {"v0": 4 * 16 * 32, "v1": 4 * 5 * 32, "v2": 2 * 128, "v3": 2 * 512, "v4": 8 * 512}


def kernel_name(variant: str, M: int, N: int, K: int) -> str:
    if variant == "v3":
        return "gemm_v3_kernel" if K % 4 == 0 and N % 4 == 0 else "gemm_v3_scalar_kernel"
    if variant == "v4":
        return "gemm_v4_kernel" if K % 8 == 0 and N % 8 == 0 else "gemm_v4_scalar_kernel"
    return KERNEL_NAMES[variant][0]


def gemm_launch(M: int, N: int, K: int, variant: str, gpu, params,
                dram: tuple[float, float] | None = None) -> Launch:
    """One GEMM. `dram` = (A bytes, B + C bytes) overrides the steady-state panel model for callers
    that know better, e.g. attention, whose inputs were just written by the previous launch."""
    BM, BN, BK, threads, raster, elem = TILES[variant]
    gm, gn = math.ceil(M / BM), math.ceil(N / BN)
    blocks = gm * gn
    kernel = kernel_name(variant, M, N, K)
    res = resources.get(kernel)
    slots = gpu.sms * occupancy(gpu, threads, res["regs"], res["smem"]).blocks_per_sm
    Kp = math.ceil(K / BK) * BK
    if variant in ("v0", "v1"):
        flops = 2.0 * M * N * K
        warp_k = M * N / 32 * K  # only threads inside C do work
    else:
        flops = 2.0 * blocks * BM * BN * Kp
        warp_k = blocks * threads / 32 * Kp
    l1_wf, smem_wf, g_in, s_in = PER_K[variant]
    if dram is None:
        dram = l2.gemm_panels(M, N, K, BM, BN, elem, slots, raster, l2.usable(gpu, params),
                              lockstep=variant not in ("v0", "v1"))
    return Launch(
        kernel=kernel, blocks=blocks, threads=threads, flops=flops,
        pipe="tc" if variant == "v4" else "fp32",
        dram=dram[1] + (0.0 if variant in NARROW_A else dram[0]),
        dram_narrow=dram[0] if variant in NARROW_A else 0.0,
        l2=blocks * (BM + BN) * Kp * elem + 4.0 * M * N,
        smem_wavefronts=warp_k * smem_wf, g_instr=warp_k * g_in + M * N / 32,
        s_instr=warp_k * s_in,
        l1_wavefronts=warp_k * l1_wf + M * N / 32,  # + one store wavefront per 32 C elements
        mlp=MLP[variant],
    )


def launches(dims, variant, gpu, params) -> list[Launch]:
    M, N, K = dims
    return [gemm_launch(M, N, K, variant, gpu, params)]
