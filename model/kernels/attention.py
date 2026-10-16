"""Unfused attention v0-v2 (src/api.cu), dims headsxNxd: O = softmax(Q·Kᵀ/sqrt(d))·V per head.

Launches per call, built from the GEMM and row-wise models:

  v0  transpose K, heads × gemm_v0 (Q·Kᵀ), a scale pass over S, softmax_v0, heads × gemm_v0 (P·V)
  v1  transpose K with the scale folded in, heads × gemm_v3, softmax v2 (N <= 2048) or v3,
      heads × gemm_v3
  v2  v1 with each GEMM batched over heads in one launch

DRAM, steady state over repeated calls: nothing if Q, K, V, O, Kᵀ and the N×N scores S all fit
in usable L2. Otherwise each launch moves its inputs and outputs, except that S is written once
(by Q·Kᵀ) and read back from L2 by the scale, softmax and P·V launches whenever S alone fits.
"""
import math

from model import l2
from model.kernels import gemm, rowwise
from model.launch import Launch

VARIANTS = ("v0", "v1", "v2")
KERNEL_NAMES = {
    "v0": ["transpose_scale_kernel", "gemm_v0_kernel", "scale_kernel", "softmax_v0_kernel"],
    "v1": ["transpose_scale_kernel", "gemm_v3_kernel", "gemm_v3_scalar_kernel"]
          + rowwise.kernel_names("softmax")["v2"] + rowwise.kernel_names("softmax")["v3"],
}
KERNEL_NAMES["v2"] = KERNEL_NAMES["v1"]


def launches(dims, variant, gpu, params) -> list[Launch]:
    heads, N, d = dims
    cap = l2.usable(gpu, params)
    qkv = 4.0 * heads * N * d  # bytes of one of Q, K, V, O, Kᵀ
    S = 4.0 * heads * N * N
    resident = 5 * qkv + S <= cap
    s_fits = S <= cap

    def dram(x):
        return 0.0 if resident else x

    n_t = heads * N * d
    out = [Launch(
        kernel="transpose_scale_kernel", blocks=math.ceil(d / 32) * math.ceil(N / 32) * heads,
        threads=256, flops=n_t, dram=dram(2 * qkv), l2=2 * qkv, smem_wavefronts=n_t / 32 * 2,
        g_instr=n_t / 32 * 2, s_instr=n_t / 32 * 2, l1_wavefronts=n_t / 32 * 2, mlp=4 * 128,
    )]
    gv = "v0" if variant == "v0" else "v3"
    # (A bytes, B + C bytes) per head. Q·Kᵀ: read Q_h and Kᵀ_h, write S_h. P·V: read P_h unless
    # S stayed in L2, read V_h, write O_h.
    qk = gemm.gemm_launch(N, N, d, gv, gpu, params,
                          dram=(dram(4.0 * N * d), dram(4.0 * (N * d + N * N))))
    pv = gemm.gemm_launch(N, d, N, gv, gpu, params,
                          dram=(dram(0 if s_fits else 4.0 * N * N), dram(4.0 * 2 * N * d)))
    gemm_qk = [qk.scaled(heads)] if variant == "v2" else [qk] * heads
    gemm_pv = [pv.scaled(heads)] if variant == "v2" else [pv] * heads
    out += gemm_qk
    if variant == "v0":
        n = heads * N * N
        out.append(Launch(kernel="scale_kernel", blocks=1024, threads=256, flops=n,
                          dram=dram(0 if s_fits else 2 * S), l2=2 * S, g_instr=n / 32 * 2,
                          l1_wavefronts=n / 32 * 2, mlp=128))
        sv = "v0"
    else:
        sv = "v2" if N <= 2048 else "v3"
    out.append(rowwise.launch("softmax", heads * N, N, sv, gpu, params,
                              dram=0.0 if (resident or s_fits) else None))
    out += gemm_pv
    return out
