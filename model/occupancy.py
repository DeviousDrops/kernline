"""Theoretical occupancy for sm_89, the same quantity as Nsight Compute's "Theoretical Occupancy".

Blocks per SM is the minimum of four limits:
  warps      max warps per SM (48) / warps per block
  registers  registers are allocated per warp in units of 256, so a warp costs
             ceil(regs·32 / 256)·256 of the SM's 65,536
  smem       shared memory per block is rounded up to 128 bytes, plus 1 KB the driver reserves
             per block, against the SM's 100 KB
  blocks     24 resident blocks per SM
"""
import math
from dataclasses import dataclass

from model.spec import Gpu

REG_ALLOC_UNIT = 256  # registers per warp are allocated in multiples of this
SMEM_ALLOC_UNIT = 128  # bytes


@dataclass(frozen=True)
class Occupancy:
    blocks_per_sm: int
    warps_per_block: int
    limiter: str  # "warps" | "registers" | "smem" | "blocks"

    @property
    def warps_per_sm(self) -> int:
        return self.blocks_per_sm * self.warps_per_block

    def fraction(self, gpu: Gpu) -> float:
        return self.warps_per_sm / gpu.max_warps_per_sm


def smem_per_block(gpu: Gpu, smem: int) -> int:
    """Shared memory one block occupies: static + dynamic, rounded, plus the per-block reserve."""
    return math.ceil(smem / SMEM_ALLOC_UNIT) * SMEM_ALLOC_UNIT + gpu.smem_reserved_per_block


def occupancy(gpu: Gpu, threads: int, regs: int, smem: int) -> Occupancy:
    """`smem` is static + dynamic shared memory per block in bytes, before the driver reserve."""
    wpb = math.ceil(threads / gpu.warp_size)
    regs_per_warp = math.ceil(max(regs, 1) * gpu.warp_size / REG_ALLOC_UNIT) * REG_ALLOC_UNIT
    limits = {
        "warps": gpu.max_warps_per_sm // wpb,
        "registers": (gpu.regs_per_sm // regs_per_warp) // wpb,
        "smem": gpu.smem_per_sm // smem_per_block(gpu, smem),
        "blocks": gpu.max_blocks_per_sm,
    }
    limiter = min(limits, key=limits.get)
    return Occupancy(limits[limiter], wpb, limiter)
