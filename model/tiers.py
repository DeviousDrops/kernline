"""T0 / T1 / T2 (docs/DESIGN.md D2). Each tier takes a descriptor's launches and adds effects.

Clocks: ceilings were measured at one SM and memory clock. A prediction is made at the clock the
kernel ran at (each result row records it, D7): compute, L2, shared memory and the LSU scale with
the SM clock, DRAM with the memory clock. Latencies are kept in cycles, so in seconds they shrink
as the clock rises.

T0  t = max over levels of work / ceiling:
      compute  FLOPs / (FP32 or tensor-core peak × params.fma_eff or tc_eff)
      dram     (DRAM bytes + narrow-sector DRAM bytes / params.narrow_eff) / DRAM read bandwidth
      l2       L1<->L2 bytes / L2 bandwidth
      smem     shared-memory wavefronts × 128 B / shared-memory bandwidth
      lsu      max(global-access wavefronts / (one per SM per cycle),
                   global instructions / params.ldg_per_clk, shared / params.lds_per_clk,
                   per SM per cycle)
T1  Occupancy and waves for the SM-side levels (compute, smem, lsu, l2). T0 spreads the work
    evenly over all SMs at full rate. T1 gives the busiest SM its ceil(blocks / SMs) blocks, run
    in rounds of at most `blocks per SM` (occupancy calculator), and lets a round with w resident
    warps run at min(1, w / params.sat_warps) of the SM's rate: one small block cannot keep an SM
    busy, and a partial last round costs a whole round.
T2  DRAM and L2 bandwidth capped by Little's law: resident warps × bytes each keeps in flight /
    latency. Plus params.launch_us per launch.
"""
import math
from dataclasses import dataclass, field

from model import resources
from model.launch import Launch
from model.occupancy import occupancy
from model.spec import Gpu

TIERS = ("T0", "T1", "T2")
SM_LEVELS = ("compute", "smem", "lsu", "l2")


@dataclass
class LaunchTime:
    launch: Launch
    seconds: float
    limiter: str
    parts: dict = field(default_factory=dict)  # level -> seconds
    blocks_per_sm: int = 0
    waves: float = 0.0


def launch_time(L: Launch, gpu: Gpu, params: dict, tier: str, sm_mhz: float,
                mem_mhz: float) -> LaunchTime:
    ti = TIERS.index(tier)
    fs, fm = sm_mhz / gpu.ceil_sm_mhz, mem_mhz / gpu.ceil_mem_mhz
    clk = sm_mhz * 1e6
    if L.pipe == "tc":
        peak = gpu.tc_flops * params["tc_eff"] * fs
    else:
        peak = gpu.fp32_flops * params["fma_eff"] * fs
    parts = {
        "compute": L.flops / peak,
        "dram": (L.dram + L.dram_narrow / params["narrow_eff"]) / (gpu.dram_bw * fm),
        "l2": L.l2 / (gpu.l2_bw * fs),
        "smem": L.smem_wavefronts * 128 / (gpu.smem_bw * fs),
        "lsu": max(L.l1_wavefronts, L.g_instr / params["ldg_per_clk"],
                   L.s_instr / params["lds_per_clk"]) / (gpu.sms * clk),
    }
    res = resources.get(L.kernel)
    occ = occupancy(gpu, L.threads, res["regs"], res["smem"] + L.dyn_smem)
    slots = gpu.sms * occ.blocks_per_sm
    waves = math.ceil(L.blocks / slots) if L.blocks else 0
    if ti >= 1 and L.blocks:
        per_sm = math.ceil(L.blocks / gpu.sms)
        full, rem = divmod(per_sm, occ.blocks_per_sm)
        rounds = [occ.blocks_per_sm] * full + ([rem] if rem else [])
        busy = sum(k / min(1.0, k * occ.warps_per_block / params["sat_warps"]) for k in rounds)
        q = busy / (L.blocks / gpu.sms)
        for k in SM_LEVELS:
            parts[k] *= q
    overhead = 0.0
    if ti >= 2 and L.blocks:
        warps = min(L.blocks, slots) * occ.warps_per_block
        if L.mlp:
            inflight = warps * L.mlp
            dram_lat = gpu.lat_cycles["dram"] / clk
            l2_lat = gpu.lat_cycles["l2"] / clk
            parts["dram"] = parts["dram"] * gpu.dram_bw * fm / min(gpu.dram_bw * fm, inflight / dram_lat)
            parts["l2"] = max(parts["l2"], L.l2 / (inflight / l2_lat))
        overhead = params["launch_us"] * 1e-6
    limiter = max(parts, key=parts.get)
    return LaunchTime(L, parts[limiter] + overhead, limiter, parts, occ.blocks_per_sm,
                      L.blocks / slots if slots else 0.0)


def predict(launches: list[Launch], gpu: Gpu, params: dict, tier: str, sm_mhz: float | None = None,
            mem_mhz: float | None = None) -> tuple[float, list[LaunchTime]]:
    """Seconds for one call (its launches run back to back) and the per-launch breakdown."""
    sm_mhz = sm_mhz or gpu.ceil_sm_mhz
    mem_mhz = mem_mhz or gpu.ceil_mem_mhz
    times = [launch_time(L, gpu, params, tier, sm_mhz, mem_mhz) for L in launches]
    return sum(t.seconds for t in times), times
