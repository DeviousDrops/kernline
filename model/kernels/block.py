"""The transformer block (block/block.cu, D4), dims BxSxDxH. A variant is one step (as in
results/<gpu>/block.csv) or "total", every step back to back. Each step is the descriptor of the
kernel it launches, at the shape the block gives it: the block's prediction is the sum of its
kernels' predictions.
"""
from model.kernels import attention, elementwise, gemm, layernorm

STEPS = ("ln1", "qkv", "split", "attn", "merge", "proj", "res1", "ln2", "fc1", "gelu", "fc2", "res2")
VARIANTS = STEPS + ("total",)
KERNEL_NAMES = {}


def step_launches(step: str, B: int, S: int, D: int, H: int, gpu, params):
    T, F, d = B * S, 4 * D, D // H
    if step in ("ln1", "ln2"):
        return layernorm.launches((T, D), "v2", gpu, params)
    gemms = {"qkv": (T, 3 * D, D), "proj": (T, D, D), "fc1": (T, F, D), "fc2": (T, D, F)}
    if step in gemms:
        return gemm.launches(gemms[step], "v3", gpu, params)
    if step == "attn":
        return attention.launches((B * H, S, d), "v2", gpu, params)
    kind, width = {"split": ("split", D), "merge": ("merge", D), "res1": ("add", D),
                   "res2": ("add", D), "gelu": ("gelu", F)}[step]
    return elementwise.launches((T, width), kind, gpu, params)


def launches(dims, variant, gpu, params):
    B, S, D, H = dims
    steps = STEPS if variant == "total" else (variant,)
    return [L for st in steps for L in step_launches(st, B, S, D, H, gpu, params)]
