"""Cache simulator (docs/SIM.md): the C++ unit tests, generator addresses checked by hand
against the kernels' index math, and request/sector counts against Nsight Compute. No GPU."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SIM, UNIT = ROOT / "build" / "cachesim", ROOT / "build" / "cachesim_test"


def need(binary: Path) -> None:
    if not binary.exists():
        pytest.skip(f"{binary.name} not built; run cmake --build build")


def test_cpp_unit_tests():
    need(UNIT)
    r = subprocess.run([UNIT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def op(kernel, variant, dims, launch, block, warp, i, *extra):
    """build/cachesim --op: (mask, bytes, store, operand), and the 32 lane tokens."""
    need(SIM)
    out = subprocess.run([SIM, kernel, variant, dims, "--op", str(launch), str(block), str(warp),
                          str(i), *extra], check=True, capture_output=True, text=True,
                         cwd=ROOT).stdout.split()
    head = dict(kv.split("=") for kv in out[:4])
    return (int(head["mask"], 16), int(head["bytes"]), head["store"] == "1",
            int(head["operand"])), out[4:]


def test_gemm_v0_lanes_walk_rows():
    # kernels/gemm/v0_naive.cu: block (16, 16), row = blockIdx.x*16 + threadIdx.x, col =
    # blockIdx.y*16 + threadIdx.y; warp w holds threadIdx.y = 2w, 2w+1. M=64, N=48, K=32: gridDim.x
    # = 4, so block 5 is (1, 1). Op 7 is k = 3's B load, op 2K the store.
    (mask, nbytes, store, operand), lanes = op("gemm", "v0", "64x48x32", 0, 5, 2, 7)
    assert (mask, nbytes, store, operand) == (0xFFFFFFFF, 4, False, 1)
    assert lanes == [f"B+{3 * 48 + 16 + 4 + l // 16}" for l in range(32)]
    (_, _, store, _), lanes = op("gemm", "v0", "64x48x32", 0, 5, 2, 64)
    assert store and lanes == [f"C+{(16 + l % 16) * 48 + 20 + l // 16}" for l in range(32)]


def test_gemm_v0_masks_rows_past_m():
    # M = 20: block x = 1 covers rows 16..31, of which 16..19 exist.
    (mask, *_), _ = op("gemm", "v0", "20x20x8", 0, 1, 0, 0)
    assert mask == 0x000F000F


def test_gemm_v2_tile_loads():
    # kernels/gemm/v2_smem_tiled.cu, T = 32: warp w is threadIdx.y = w. Block 33 = (1, 1).
    # Ops alternate A, B per k-step; op 4 is k-step 2 (k0 = 64): A, op 5: B.
    (_, nbytes, _, operand), lanes = op("gemm", "v2", "1024x1024x1024", 0, 33, 5, 4)
    assert (nbytes, operand) == (4, 0) and lanes == [f"A+{37 * 1024 + 64 + l}" for l in range(32)]
    _, lanes = op("gemm", "v2", "1024x1024x1024", 0, 33, 5, 5)
    assert lanes == [f"B+{69 * 1024 + 32 + l}" for l in range(32)]


def test_gemm_v3_a_tile_float4s():
    # kernels/gemm/v3_reg_tiled.cu: a_row = tid/2, a_k = (tid%2)*4. Block (1, 2) is 1 + 2*8 = 17;
    # op 10 is k-step 5 (k0 = 40), A. Warp 3 = tids 96..127: rows 48 + l/2 of the block.
    (_, nbytes, store, operand), lanes = op("gemm", "v3", "1024x1024x1024", 0, 17, 3, 10)
    assert (nbytes, store, operand) == (16, False, 0)
    assert lanes == [f"A+{(256 + 48 + l // 2) * 1024 + 40 + (l % 2) * 4}" for l in range(32)]


def test_gemm_v3_stores_two_rows_per_warp():
    # Stores follow the 2*K/8 loads: s = 3 is row i = 1 of the thread's 8, half h = 1.
    # tx = tid % 16, ty = tid / 16; warp 3: ty = 6 + l/16.
    (_, nbytes, store, operand), lanes = op("gemm", "v3", "1024x1024x1024", 0, 17, 3, 256 + 3)
    assert (nbytes, store) == (16, True)
    assert lanes == [f"C+{(256 + (6 + l // 16) * 8 + 1) * 1024 + 128 + 64 + (l % 16) * 4}"
                     for l in range(32)]
