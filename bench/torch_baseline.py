#!/usr/bin/env python3
"""PyTorch baselines (D9) in the bench's CSV schema, timed exactly like build/bench.

    torch_baseline.py softmax   <rows> <cols>
    torch_baseline.py layernorm <rows> <cols>

Same method as bench/bench.cu: warm up for >= 3 launches and >= 500 ms, then 21 timed batches
of back-to-back launches lasting >= 5 ms each; time_ms is the median per launch. Clocks are the
median NVML reading taken while the batch is in flight. Runs under inference_mode (no autograd).
Each call returns a fresh tensor from PyTorch's caching allocator, which reuses the block freed by
the previous call, so no cudaMalloc happens inside the timed region.
"""
import ctypes
import math
import statistics
import sys

import torch
import torch.nn.functional as F

WARMUP, WARMUP_MS, BATCH_MS, RUNS = 3, 500.0, 5.0, 21


class Nvml:
    """Minimal ctypes NVML: SM and memory clock of device 0. Reads 0 if NVML is unavailable."""

    def __init__(self):
        self.ok = False
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            self.dev = ctypes.c_void_p()
            self.ok = (self.lib.nvmlInit_v2() == 0 and
                       self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.dev)) == 0)
        except OSError:
            pass

    def clocks(self) -> tuple[int, int]:
        if not self.ok:
            return 0, 0
        sm, mem = ctypes.c_uint(), ctypes.c_uint()
        self.lib.nvmlDeviceGetClockInfo(self.dev, 1, ctypes.byref(sm))   # NVML_CLOCK_SM
        self.lib.nvmlDeviceGetClockInfo(self.dev, 2, ctypes.byref(mem))  # NVML_CLOCK_MEM
        return sm.value, mem.value


def gpu_slug(name: str) -> str:
    """Same rule as common/gpu_info.h: "NVIDIA GeForce RTX 4060 Laptop GPU" -> "rtx4060_laptop"."""
    for drop in ("NVIDIA ", "GeForce ", " GPU"):
        name = name.replace(drop, "", 1)
    return name.replace("RTX ", "RTX", 1).replace(" ", "_").lower()


def time_launch(launch) -> tuple[float, int, int]:
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(n: int) -> float:
        start.record()
        for _ in range(n):
            launch()
        sample = nvml.clocks()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop), sample

    warm_ms, warm_n = 0.0, 0
    while warm_n < WARMUP or warm_ms < WARMUP_MS:
        warm_ms += timed(1)[0]
        warm_n += 1
    batch = max(1, math.ceil(BATCH_MS / (warm_ms / warm_n)))
    times, sms, mems = [], [], []
    for _ in range(RUNS):
        ms, (sm, mem) = timed(batch)
        times.append(ms / batch)
        sms.append(sm)
        mems.append(mem)
    return statistics.median(times), int(statistics.median(sms)), int(statistics.median(mems))


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in ("softmax", "layernorm"):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    kernel, rows, cols = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    torch.manual_seed(0)
    x = torch.rand(rows, cols, device="cuda") - 0.5
    elems = rows * cols
    with torch.inference_mode():
        if kernel == "softmax":
            ms, sm, mem = time_launch(lambda: torch.softmax(x, -1))
            flop, nbytes = 5 * elems, 8 * elems
        else:
            gamma = torch.rand(cols, device="cuda") - 0.5
            beta = torch.rand(cols, device="cuda") - 0.5
            ms, sm, mem = time_launch(lambda: F.layer_norm(x, (cols,), gamma, beta, 1e-5))
            flop, nbytes = 8 * elems, 8 * elems + 8 * cols
    gpu = gpu_slug(torch.cuda.get_device_name(0))
    print(f"{gpu},{sm},{mem},{kernel},torch,{rows}x{cols},{ms:.4f},{flop},{nbytes},,,")
    return 0


nvml = Nvml()

if __name__ == "__main__":
    sys.exit(main())
