"""What a descriptor tells the tiers about one kernel launch."""
from dataclasses import dataclass


@dataclass
class Launch:
    kernel: str  # key in model/resources.json (registers, static shared memory)
    blocks: int
    threads: int  # per block
    dyn_smem: int = 0  # dynamic shared memory per block, bytes
    flops: float = 0.0  # FLOPs the kernel executes, including zero-padded tile work
    pipe: str = "fp32"  # "fp32" (FMA pipe) or "tc" (tensor cores, FP16 in / FP32 accumulate)
    dram: float = 0.0  # modeled DRAM bytes, read + write, in long contiguous runs
    dram_narrow: float = 0.0  # DRAM bytes fetched as isolated 32 B sectors (strided tile rows)
    l2: float = 0.0  # bytes between L1 and L2 (L1 misses and stores)
    smem_wavefronts: float = 0.0  # shared-memory wavefronts (128 B each, one per SM cycle)
    g_instr: float = 0.0  # global load/store warp instructions (LG queue)
    s_instr: float = 0.0  # shared load/store warp instructions (MIO queue)
    l1_wavefronts: float = 0.0  # global load/store wavefronts in L1 (one distinct 128 B line each)
    mlp: float = 0.0  # bytes one warp keeps in flight to memory (T2, Little's law)

    def scaled(self, n: int) -> "Launch":
        """The same launch repeated n times as one grid (e.g. one GEMM per head, batched)."""
        out = Launch(**self.__dict__)
        out.blocks *= n
        for f in ("flops", "dram", "dram_narrow", "l2", "smem_wavefronts", "g_instr", "s_instr",
                  "l1_wavefronts"):
            setattr(out, f, getattr(self, f) * n)
        return out
