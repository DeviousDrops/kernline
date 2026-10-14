"""The transformer block's elementwise and layout kernels (kernels/block/), dims TxD: T tokens of
width D (for gelu, D is the MLP width 4·d_model).

  add    y += x over T×D (residual): 2 reads, 1 write per element
  gelu   x = GELU(x) in place over T×D: 1 read, 1 write, ~20 FLOP-equivalent (tanhf)
  split  [T, 3D] -> three [B·H, S, d]: 1 read, 1 write per element of the 3·T·D
  merge  [B·H, S, d] -> [T, D]: 1 read, 1 write per element

All coalesced (one L1 wavefront per 32 elements per access). DRAM by the footprint rule. split and
merge spend ~40 integer instructions per element on index math (two 32-bit divisions by runtime
values); they are counted as FLOP-equivalents on the FP32 pipe, which issues at the same rate.
"""
import math

from model import l2
from model.launch import Launch

VARIANTS = ("add", "gelu", "split", "merge")
KERNEL_NAMES = {"add": ["add_inplace_kernel"], "gelu": ["gelu_inplace_kernel"],
                "split": ["split_heads_kernel"], "merge": ["merge_heads_kernel"]}
# variant: (elements per token-column unit, reads, writes, distinct buffers, flops per element)
SHAPE = {"add": (1, 2, 1, 2, 1), "gelu": (1, 1, 1, 1, 20), "split": (3, 1, 1, 2, 80),
         "merge": (1, 1, 1, 2, 80)}


def launch(variant: str, T: int, D: int, gpu, params) -> Launch:
    per, reads, writes, buffers, flops = SHAPE[variant]
    n = T * D * per
    if variant in ("split", "merge"):
        blocks = math.ceil(D * per / 256) * T  # one grid row per token
    else:
        blocks = min(math.ceil(n / 256), 4096)  # elementwise_grid in src/api.cu
    traffic = 4.0 * n * (reads + writes)
    footprint = 4.0 * n * buffers
    return Launch(
        kernel=KERNEL_NAMES[variant][0], blocks=blocks, threads=256, flops=flops * n,
        dram=0.0 if footprint <= l2.usable(gpu, params) else traffic, l2=traffic,
        g_instr=n / 32 * (reads + writes), l1_wavefronts=n / 32 * (reads + writes),
        mlp=128 * reads,
    )


def launches(dims, variant, gpu, params):
    T, D = dims
    return [launch(variant, T, D, gpu, params)]
