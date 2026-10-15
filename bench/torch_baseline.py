#!/usr/bin/env python3
"""PyTorch baselines (D9) in the bench's CSV schema, timed exactly like build/bench.

    torch_baseline.py softmax   <rows> <cols>
    torch_baseline.py layernorm <rows> <cols>
    torch_baseline.py attention <heads> <N> <d>

Same method as bench/bench.cu: warm up for >= 3 launches and >= 500 ms (and until the memory
clock is at its maximum, for at most 5 s), then 21 timed batches
of back-to-back launches lasting >= 5 ms each; time_ms is the median per launch. Clocks are the
median NVML reading taken while the batch is in flight. Runs under inference_mode (no autograd).

Each timed batch is replayed from a CUDA graph captured once. A Python-level call costs 20-60 us
of CPU dispatch, longer than the kernel itself at small shapes, so eager back-to-back calls leave
the GPU idle between kernels and time the CPU instead. The C++ bench enqueues in ~3 us, so it
doesn't need this. The graph's output buffer is allocated once at capture, so no allocator runs
inside the timed region either.
"""
import ctypes
import math
import statistics
import sys

import torch
import torch.nn.functional as F

WARMUP, WARMUP_MS, WARMUP_MAX_MS, BATCH_MS, RUNS = 3, 500.0, 5000.0, 5.0, 21


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

    def mem_at_max(self) -> bool:
        """True once the memory clock is at its maximum (see common/gpu_info.h), or without NVML."""
        if not self.ok:
            return True
        cur, top = ctypes.c_uint(), ctypes.c_uint()
        if (self.lib.nvmlDeviceGetClockInfo(self.dev, 2, ctypes.byref(cur)) or
                self.lib.nvmlDeviceGetMaxClockInfo(self.dev, 2, ctypes.byref(top))):
            return True
        return cur.value >= top.value

    def warming(self, n: int, ms: float) -> bool:
        """The bench's warmup condition (common/bench_util.h, time_launch)."""
        return n < WARMUP or ms < WARMUP_MS or (ms < WARMUP_MAX_MS and not self.mem_at_max())


def gpu_slug(name: str) -> str:
    """Same rule as common/gpu_info.h: "NVIDIA GeForce RTX 4060 Laptop GPU" -> "rtx4060_laptop"."""
    for drop in ("NVIDIA ", "GeForce ", " GPU"):
        name = name.replace(drop, "", 1)
    return name.replace("RTX ", "RTX", 1).replace(" ", "_").lower()


def time_launch(launch) -> tuple[float, int, int]:
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(fn, n: int) -> float:
        start.record()
        for _ in range(n):
            fn()
        sample = nvml.clocks()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop), sample

    # Kernel time per call, measured from a captured graph so CPU dispatch is excluded.
    def capture(n: int) -> torch.cuda.CUDAGraph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                launch()
        return g

    side = torch.cuda.Stream()  # graph capture needs the ops warmed up on a side stream first
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(WARMUP):
            launch()
    torch.cuda.current_stream().wait_stream(side)

    probe = capture(1)
    warm_ms, warm_n = 0.0, 0
    while nvml.warming(warm_n, warm_ms):
        warm_ms += timed(probe.replay, 1)[0]
        warm_n += 1
    batch = max(1, math.ceil(BATCH_MS / (warm_ms / warm_n)))
    # One eager call in the range tools/profile.sh profiles (same marker as build/bench).
    torch.cuda.nvtx.range_push("kl_profile")
    launch()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    graph = capture(batch)
    times, sms, mems = [], [], []
    for _ in range(RUNS):
        ms, (sm, mem) = timed(graph.replay, 1)
        times.append(ms / batch)
        sms.append(sm)
        mems.append(mem)
    return statistics.median(times), int(statistics.median(sms)), int(statistics.median(mems))


def attention(heads: int, N: int, d: int) -> tuple[float, int, int, float, float]:
    """F.scaled_dot_product_attention forced to the math backend (D9): the unfused
    Q·Kᵀ -> softmax -> ·V that PyTorch falls back to, not its fused flash/efficient kernels.
    flop and bytes use the same conventions as bench_attention in bench/bench.cu."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q, k, v = (torch.rand(1, heads, N, d, device="cuda") - 0.5 for _ in range(3))
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        ms, sm, mem = time_launch(lambda: F.scaled_dot_product_attention(q, k, v))
    nn = heads * N * N
    return ms, sm, mem, 4 * nn * d + 5 * nn, 4 * (4 * heads * N * d + 4 * nn)


def main() -> int:
    if not ((len(sys.argv) == 4 and sys.argv[1] in ("softmax", "layernorm")) or
            (len(sys.argv) == 5 and sys.argv[1] == "attention")):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    gpu = gpu_slug(torch.cuda.get_device_name(0))
    torch.manual_seed(0)
    if sys.argv[1] == "attention":
        heads, N, d = map(int, sys.argv[2:5])
        ms, sm, mem, flop, nbytes = attention(heads, N, d)
        print(f"{gpu},{sm},{mem},attention,torch,{heads}x{N}x{d},{ms:.4f},{flop},{nbytes},,,")
        return 0
    kernel, rows, cols = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
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
    print(f"{gpu},{sm},{mem},{kernel},torch,{rows}x{cols},{ms:.4f},{flop},{nbytes},,,")
    return 0


nvml = Nvml()

if __name__ == "__main__":
    sys.exit(main())
