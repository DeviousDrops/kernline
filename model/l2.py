"""DRAM traffic: which bytes the 32 MB L2 keeps between uses.

Two rules, both steady state, because the bench times back-to-back repeats of one call:

* Footprint rule: if everything a call touches fits in the usable L2 (params.l2_eff × L2 size),
  the previous repeat left it there and the call moves (almost) nothing to or from DRAM.
* Panel model for tiled GEMMs: block (i, j) reads row panel i of A and column panel j of B and
  writes tile C(i, j). Blocks run in waves of `slots` in launch order, and each wave reads every
  panel it needs once: its blocks stream the same K-slices together. A wave streams its whole
  footprint (every panel and tile it touches) through L2, so a panel last used in wave w1 is
  still there in wave w2 only if the waves w1 .. w2-1 together streamed no more than the usable
  capacity. The launch is simulated twice and the second pass counted, which is the steady state
  the bench measures.

  Sharing inside a wave needs the blocks in lockstep, which the __syncthreads tile loops (v2-v4)
  are. v0 and v1 blocks are free-running k loops that drift apart by more than L2's turnover
  time, so for them every block is its own "wave": reuse distance is counted block by block
  (measured on the 4070: v1 at 4096^3 moves 35 GB, the block-by-block count 33.5 GB, a lockstep
  wave count 20 GB).
"""
import math


def usable(gpu, params) -> float:
    return params["l2_eff"] * gpu.l2_bytes


def gemm_panels(M: int, N: int, K: int, BM: int, BN: int, elem: int, slots: int, raster: str,
                cap: float, lockstep: bool = True) -> tuple[float, float]:
    """Steady-state DRAM bytes of one tiled GEMM: (A panel misses, B panel + C tile misses).

    raster "n": blockIdx.x walks the N tiles fastest (v1-v4). "m": it walks the M tiles (v0).
    """
    gm, gn = math.ceil(M / BM), math.ceil(N / BN)
    blocks = gm * gn
    if not lockstep:
        slots = 1

    def size(p):
        kind, i = p[0], p[1]
        if kind == "A":
            return min(BM, M - i * BM) * K * elem
        if kind == "B":
            return K * min(BN, N - i * BN) * elem
        _, i, j = p
        return min(BM, M - i * BM) * min(BN, N - j * BN) * 4

    waves = []
    for w0 in range(0, blocks, slots):
        need = {}
        for b in range(w0, min(w0 + slots, blocks)):
            i, j = divmod(b, gn) if raster == "n" else (b % gm, b // gm)
            need[("A", i)] = need[("B", j)] = need[("C", i, j)] = None
        waves.append(list(need))
    sizes = [sum(size(p) for p in need) for need in waves]
    last: dict = {}  # panel -> index of the last wave that used it, over both passes
    streamed = [0.0]  # streamed[w] = footprint of waves 0 .. w-1, over both passes
    traffic = {"A": 0.0, "B": 0.0, "C": 0.0}
    for n, need in enumerate(waves + waves):
        for p in need:
            hit = p in last and streamed[n] - streamed[last[p]] <= cap
            if n >= len(waves) and not hit:
                traffic[p[0]] += size(p)
            last[p] = n
        streamed.append(streamed[-1] + sizes[n % len(waves)])
    return traffic["A"], traffic["B"] + traffic["C"]
