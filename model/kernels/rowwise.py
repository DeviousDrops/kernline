"""Shared model of the row-wise kernels (softmax and LayerNorm along the last dim), rows×cols FP32.

Per variant, from kernels/softmax and kernels/layernorm:

  v0  one thread per row (128 per block). Each warp access touches 32 rows, so every load or
      store instruction is 32 L1 wavefronts. Three read passes plus the write.
  v1  one warp per row (8 per block), float4 when cols % 4 == 0 and buffers are aligned. Three
      read passes plus the write; passes 2-3 come from L2 while the rows in flight fit in it.
  v2  one warp per row with the row in registers when cols <= 2048 (one read pass), otherwise a
      two-pass online softmax / Welford stream.
  v3  one block per row with the row in shared memory (one read pass) when it fits the 99 KB
      opt-in; otherwise the launcher falls back to v2's stream.

DRAM (model/l2.py footprint rule): nothing if input + output fit in usable L2 (the bench's repeats
keep them there). Otherwise the first read pass and the write go to DRAM, and the later read
passes too unless the rows in flight (resident rows × row bytes) fit in usable L2.
LayerNorm also reads gamma and beta in its output pass; they are shared by every row.
"""
import math

from model import l2, resources
from model.launch import Launch
from model.occupancy import occupancy

VARIANTS = ("v0", "v1", "v2", "v3")
FLOPS_PER_ELEM = {"softmax": 5, "layernorm": 8}  # bench/bench.cu convention
SMEM_PASSES = {"softmax": 4, "layernorm": 3}  # v3 shared-memory accesses per element
V3_STATIC_SMEM = 128


def v2_cache(cols: int, vec: bool) -> int:
    """src/api.cu: float4s per lane cached in registers, 0 = float4 stream, -1 = scalar stream."""
    if not vec:
        return -1
    for k in (1, 2, 4, 8, 16):
        if cols <= 128 * k:
            return k
    return 0


def v3_shape(cols: int, gpu) -> tuple[int, int] | None:
    """src/api.cu v3_shape: (threads, dynamic smem) or None when the row does not fit."""
    smem = cols * 4
    if smem + V3_STATIC_SMEM > gpu.smem_per_block_optin:
        return None
    want = ((cols + 15) // 16 + 31) // 32 * 32
    return min(max(want, 128), 1024), smem


def launch(kind: str, rows: int, cols: int, variant: str, gpu, params,
           dram: float | None = None) -> Launch:
    """One row-wise launch. `dram` overrides the footprint rule (attention's in-place softmax)."""
    vec = cols % 4 == 0
    elems = rows * cols
    data = 4.0 * elems
    ln = kind == "layernorm"
    dyn_smem = 0
    smem_wf = smem_in = 0.0
    stream_fallback = False
    if variant == "v3":
        shape = v3_shape(cols, gpu)
        if shape is None:  # the launcher falls back to v2's stream
            variant, stream_fallback = "v2", True
        else:
            threads, dyn_smem = shape
    if variant == "v0":
        kernel, threads, rows_per_block, passes, mlp = f"{kind}_v0_kernel", 128, 128, 3, 1024
    elif variant == "v1":
        kernel = f"{kind}_v1_kernel" if vec else f"{kind}_v1_scalar_kernel"
        threads, rows_per_block, passes, mlp = 256, 8, 3, 2048 if vec else 512
    elif variant == "v2":
        k = (0 if vec else -1) if stream_fallback else v2_cache(cols, vec)
        kernel, threads, rows_per_block = f"{kind}_v2_kernel<{k}>", 256, 8
        passes, mlp = (1, 512 * k) if k > 0 else (2, 2048 if vec else 512)
    else:
        kernel = f"{kind}_v3_kernel<{'true' if vec else 'false'}>"
        rows_per_block, passes, mlp = 1, 1, 2048
        smem_wf = elems / 32 * SMEM_PASSES[kind]
        smem_in = smem_wf / 4 if vec else smem_wf
    blocks = math.ceil(rows / rows_per_block)
    width = 4 if vec else 1  # floats per lane per instruction
    if variant == "v0":
        l1_wf = elems * (passes + 1) + (elems / 32 * 2 if ln else 0)
        g_instr = elems / 32 * (passes + 1 + (2 if ln else 0))
    else:
        g_instr = elems / (32 * width) * (passes + 1 + (2 if ln else 0))
        l1_wf = elems / 32 * (passes + 1 + (2 if ln else 0))
    gamma_beta = 8.0 * cols if ln else 0.0
    if dram is None:
        cap = l2.usable(gpu, params)
        if 2 * data + gamma_beta <= cap:
            dram = 0.0
        else:
            res = resources.get(kernel)
            slots = gpu.sms * occupancy(gpu, threads, res["regs"], res["smem"] + dyn_smem).blocks_per_sm
            in_flight = min(rows, slots * rows_per_block) * cols * 4.0
            dram_passes = passes if in_flight > cap else 1
            dram = data * (dram_passes + 1) + gamma_beta
    return Launch(
        kernel=kernel, blocks=blocks, threads=threads, dyn_smem=dyn_smem,
        flops=FLOPS_PER_ELEM[kind] * elems, dram=dram,
        l2=data * (passes + 1) + (2 * data if ln and cols * 8 > 32 * 1024 else 0.0),
        smem_wavefronts=smem_wf, g_instr=g_instr, s_instr=smem_in, l1_wavefronts=l1_wf, mlp=mlp,
    )


def kernel_names(kind: str) -> dict:
    return {
        "v0": [f"{kind}_v0_kernel"], "v1": [f"{kind}_v1_kernel", f"{kind}_v1_scalar_kernel"],
        "v2": [f"{kind}_v2_kernel<{k}>" for k in (-1, 0, 1, 2, 4, 8, 16)],
        "v3": [f"{kind}_v3_kernel<true>", f"{kind}_v3_kernel<false>"],
    }
