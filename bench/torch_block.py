#!/usr/bin/env python3
"""PyTorch version of the transformer block (D4) in the bench's CSV schema.

    torch_block.py <B> <S> [D=768] [H=12]

Same computation as transformer_block in src/api.cu: pre-LN GPT-2 block, FP32, no biases,
attention through SDPA's math (unfused) backend (D9). Prints two rows, kernel "block", dims
BxSxDxH:

  torch        eager: back-to-back Python calls, CPU dispatch included. This is the D4 baseline,
               what a PyTorch user gets without compiling anything.
  torch_graph  the same calls replayed from a CUDA graph (bench/torch_baseline.py's method), so
               only GPU time counts. The gap between the two rows is dispatch overhead.

Eager timing follows the bench method otherwise: warm up >= 3 calls and >= 500 ms (and until the
memory clock is at its maximum, for at most 5 s), then 21 batches of back-to-back calls lasting
>= 5 ms, median per call, NVML clocks sampled in flight.
"""
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_baseline import BATCH_MS, RUNS, gpu_slug, nvml, time_launch  # noqa: E402


def make_block(B: int, S: int, D: int, H: int):
    T, d = B * S, D // H
    g = lambda *shape: torch.rand(*shape, device="cuda") - 0.5  # same range as bench's inputs
    x = g(T, D)
    ln1_g, ln1_b, ln2_g, ln2_b = g(D), g(D), g(D), g(D)
    Wqkv, Wo, W1, W2 = g(D, 3 * D), g(D, D), g(D, 4 * D), g(4 * D, D)

    def block():
        h = F.layer_norm(x, (D,), ln1_g, ln1_b, 1e-5)
        q, k, v = (h @ Wqkv).view(B, S, 3, H, d).permute(2, 0, 3, 1, 4)
        with sdpa_kernel(SDPBackend.MATH):
            o = F.scaled_dot_product_attention(q, k, v)
        x1 = x + o.permute(0, 2, 1, 3).reshape(T, D) @ Wo
        h = F.layer_norm(x1, (D,), ln2_g, ln2_b, 1e-5)
        return x1 + F.gelu(h @ W1, approximate="tanh") @ W2

    return block


def time_eager(fn) -> tuple[float, int, int]:
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(n: int):
        start.record()
        for _ in range(n):
            fn()
        clocks = nvml.clocks()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop), clocks

    warm_ms, warm_n = 0.0, 0
    while nvml.warming(warm_n, warm_ms):
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
    if len(sys.argv) not in (3, 4, 5):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    B, S = int(sys.argv[1]), int(sys.argv[2])
    D = int(sys.argv[3]) if len(sys.argv) > 3 else 768
    H = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    torch.manual_seed(0)
    gpu = gpu_slug(torch.cuda.get_device_name(0))
    dims = f"{B}x{S}x{D}x{H}"
    # Cap the allocator at physical memory: under WSL2 an over-commit spills into system memory
    # (10x slower) instead of raising, which would record a bogus row rather than skip it.
    torch.cuda.set_per_process_memory_fraction(0.95)
    with torch.inference_mode():
        block = make_block(B, S, D, H)
        for variant, timer in (("torch", time_eager), ("torch_graph", time_launch)):
            torch.cuda.empty_cache()
            try:
                ms, sm, mem = timer(block)
            except torch.OutOfMemoryError:
                # A captured graph keeps every intermediate alive at once; at B·S = 16k tokens
                # that no longer fits in 8 GB. Skip the row rather than the whole suite.
                print(f"{variant} {dims}: out of memory, row skipped", file=sys.stderr)
                continue
            print(f"{gpu},{sm},{mem},block,{variant},{dims},{ms:.4f},,,,,")
    return 0


if __name__ == "__main__":
    sys.exit(main())
