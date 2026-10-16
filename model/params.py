"""Model parameters tuned on the 4070 Laptop only (D1): model/params.json, written by
`python -m model tune`. Everything else the model uses is measured (specs/) or counted from the
kernels' index math (model/kernels/).

  l2_eff       fraction of the 32 MB L2 that holds reusable data (sets, other traffic)
  launch_us    GPU time per back-to-back launch that no roofline accounts for
  ldg_per_clk  global load/store instructions an SM issues per cycle, and
  lds_per_clk  shared ones. Narrow (32-bit) accesses cannot reach the bandwidth tools/ceilings
               measures with 128-bit loads; ncu shows their queues filling as lg_throttle and
               mio_throttle.
  narrow_eff   fraction of DRAM read bandwidth left when a stream arrives as isolated 32 B
               sectors from rows far apart (GEMM v3's 128×8 A tile when A lives in DRAM), which
               defeats DRAM row-buffer locality
  fma_eff      fraction of the FMA microbenchmark's peak a real FMA loop sustains, with its loads,
               address math and barriers in the same instruction stream
  tc_eff       the same for tensor-core (WMMA) loops
  sat_warps    resident warps an SM needs to run at its full rate (T1); fewer leave issue slots
               idle while warps wait on latency
"""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "params.json"
DEFAULTS = {"l2_eff": 0.85, "launch_us": 2.0, "ldg_per_clk": 0.5, "lds_per_clk": 0.6,
            "narrow_eff": 0.4, "fma_eff": 0.6, "tc_eff": 0.45, "sat_warps": 12}


def load() -> dict:
    return {**DEFAULTS, **(json.loads(PATH.read_text()) if PATH.exists() else {})}


def save(p: dict, note: str) -> None:
    PATH.write_text(json.dumps({**p, "tuned_on": note}, indent=1) + "\n")
