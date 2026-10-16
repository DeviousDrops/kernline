"""specs/<gpu>.json -> Gpu: measured ceilings (D3) plus the device limits occupancy needs."""
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Gpu:
    name: str
    sms: int
    warp_size: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    regs_per_sm: int
    smem_per_sm: int
    smem_per_block_optin: int
    smem_reserved_per_block: int
    l2_bytes: int
    # Ceilings, measured by tools/ceilings at (ceil_sm_mhz, ceil_mem_mhz).
    ceil_sm_mhz: float
    ceil_mem_mhz: float
    fp32_flops: float  # FLOP/s
    tc_flops: float  # FLOP/s, FP16 inputs with FP32 accumulation
    dram_bw: float  # B/s, read
    l2_bw: float  # B/s
    smem_bw: float  # B/s
    lat_cycles: dict  # {"l1", "l2", "dram"}: pointer-chase latency in SM cycles

    @property
    def max_warps_per_sm(self) -> int:
        return self.max_threads_per_sm // self.warp_size


def load(gpu: str) -> Gpu:
    spec = json.loads((ROOT / "specs" / f"{gpu}.json").read_text())
    dev, c = spec["device"], spec["ceilings"]
    if not c:
        raise ValueError(f"specs/{gpu}.json has no measured ceilings; run tools/ceilings/run.sh")
    return Gpu(
        name=spec["gpu"], sms=dev["sm_count"], warp_size=dev["warp_size"],
        max_threads_per_sm=dev["max_threads_per_sm"], max_blocks_per_sm=dev["max_blocks_per_sm"],
        regs_per_sm=dev["regs_per_sm"], smem_per_sm=dev["smem_per_sm_bytes"],
        smem_per_block_optin=dev["smem_per_block_optin_bytes"],
        smem_reserved_per_block=dev["smem_reserved_per_block_bytes"], l2_bytes=dev["l2_bytes"],
        ceil_sm_mhz=c["sm_clk_mhz"], ceil_mem_mhz=c["mem_clk_mhz"],
        fp32_flops=c["fp32_gflops"] * 1e9, tc_flops=c["fp16_tc_fp32acc_gflops"] * 1e9,
        dram_bw=c["dram_read_gbs"] * 1e9, l2_bw=c["l2_read_gbs"] * 1e9,
        smem_bw=c["smem_read_gbs"] * 1e9, lat_cycles=dict(c["latency_cycles"]),
    )


def gpus() -> list[str]:
    """Every GPU with measured ceilings."""
    out = []
    for p in sorted((ROOT / "specs").glob("*.json")):
        if json.loads(p.read_text()).get("ceilings"):
            out.append(p.stem)
    return out
