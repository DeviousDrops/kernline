"""Cache simulator (docs/SIM.md): the C++ unit tests, generator addresses checked by hand
against the kernels' index math, and request/sector counts against Nsight Compute. No GPU."""
import csv
import subprocess
import sys
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


# Buffers in bench order: Q, K, V, O, work (S = work, then Kt = work + heads*N*N floats).

def test_transpose_reads_k_rows_writes_kt_columns():
    # kernels/attention/helpers.cu, block (32, 8), 4 rows per thread. K is N x d = 256 x 64 per
    # head: gridDim = (2, 8, 12). Block 39 = (1, 3, 2); warp 5 = threadIdx.y 5.
    (_, nbytes, store, _), lanes = op("attention", "v1", "12x256x64", 0, 39, 5, 2)
    assert (nbytes, store) == (4, False)
    assert lanes == [f"K+{2 * 256 * 64 + (96 + 5 + 16) * 64 + 32 + l}" for l in range(32)]
    (_, _, store, _), lanes = op("attention", "v1", "12x256x64", 0, 39, 5, 6)
    kt = 12 * 256 * 256
    assert store and lanes == [f"work+{kt + 2 * 256 * 64 + (32 + 5 + 16) * 256 + 96 + l}" for l in range(32)]


def test_softmax_v2_cached_row_per_warp():
    # N = 256 -> softmax_v2_kernel<2> (launch 13 of v1), 8 rows per block, in place on S.
    (_, nbytes, store, _), lanes = op("attention", "v1", "12x256x64", 13, 10, 3, 1)
    assert (nbytes, store) == (16, False)
    assert lanes == [f"work+{83 * 256 + 128 + 4 * l}" for l in range(32)]
    (_, _, store, _), lanes2 = op("attention", "v1", "12x256x64", 13, 10, 3, 3)
    assert store and lanes2 == lanes


def test_softmax_v3_block_per_row():
    # N = 4096 -> softmax_v3_kernel<1> (launch 2 of v2), 256 threads, 4 float4 per thread per pass.
    (_, nbytes, store, _), lanes = op("attention", "v2", "12x4096x64", 2, 7, 2, 5)
    assert (nbytes, store) == (16, True)
    assert lanes == [f"work+{7 * 4096 + 4 * (64 + l + 256)}" for l in range(32)]


def test_scale_and_softmax_v0():
    # v0: launch 13 = scale_kernel (grid 1024 x 256, grid-stride), 14 = softmax_v0_kernel.
    _, lanes = op("attention", "v0", "12x256x64", 13, 3, 1, 0)
    assert lanes == [f"work+{3 * 256 + 32 + l}" for l in range(32)]
    (_, _, store, _), lanes = op("attention", "v0", "12x256x64", 13, 3, 1, 5)
    assert store and lanes == [f"work+{3 * 256 + 32 + l + 2 * 1024 * 256}" for l in range(32)]
    # softmax_v0: one row per thread; op 2*256 + 2*10 + 1 is pass 3's store of column 10.
    (_, _, store, _), lanes = op("attention", "v0", "12x256x64", 14, 2, 1, 533)
    assert store and lanes == [f"work+{(288 + l) * 256 + 10}" for l in range(32)]


def test_attention_v2_batches_pv_and_masks_the_empty_half():
    # P·V: N x 64 output, so the v3 tile's second column half (h = 1) has no columns: no lane
    # stores (mask 0). Launch 3; gridDim = (1, 2, 12), so block (0, by=1, z=4) = 1 + 2*4 = 9.
    (mask, *_), _ = op("attention", "v2", "12x256x64", 3, 1 + 2 * 4, 0, 2 * 32 + 1)
    assert mask == 0
    (_, _, _, operand), lanes = op("attention", "v2", "12x256x64", 3, 1 + 2 * 4, 0, 0)
    assert operand == 0 and lanes[0] == f"work+{4 * 256 * 256 + 128 * 256}"


REF = ROOT / "results" / "sim" / "ncu_rtx4060_laptop.csv"
CONFIG = ROOT / "results" / "sim" / "launch_config_rtx4060_laptop.csv"


def cases():
    for line in (ROOT / "sim" / "cases.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            yield tuple(line.split())


def test_reference_covers_every_case():
    if not REF.exists():
        pytest.skip("no ncu reference yet; run tools/sim_ncu.sh")
    have = {(r["kernel"], r["variant"], r["dims"]) for r in csv.DictReader(REF.open())}
    assert set(cases()) <= have
    configs = {(r["kernel_name"], r["block_size"], r["smem_per_block"]) for r in csv.DictReader(CONFIG.open())}
    assert ("gemm_v3_kernel", "256", "9344") in configs


COUNTS = ["ld_requests", "ld_sectors", "st_requests", "st_sectors"]
LIVE = [("gemm", "v2", "1024x1024x1024"), ("gemm", "v3", "1024x1024x1024"),
        ("attention", "v1", "12x256x64"), ("attention", "v2", "12x256x64")]


@pytest.mark.parametrize("kernel,variant,dims", LIVE)
def test_requests_and_sectors_match_ncu_exactly(kernel, variant, dims):
    """Request and sector counts depend only on the index math, so they must be exact
    (docs/SIM.md, "Generator check"). A mismatch means a generator is wrong, not the caches."""
    need(SIM)
    if not REF.exists():
        pytest.skip("no ncu reference yet; run tools/sim_ncu.sh")
    ref = {int(r["launch"]): r for r in csv.DictReader(REF.open())
           if (r["kernel"], r["variant"], r["dims"]) == (kernel, variant, dims)}
    out = subprocess.run([SIM, kernel, variant, dims], check=True, capture_output=True, text=True,
                         cwd=ROOT).stdout
    sim = list(csv.DictReader(out.splitlines()))
    assert len(sim) == len(ref)
    for r in sim:
        n = ref[int(r["launch"])]
        assert r["sim_kernel"] == n["kernel_name"], r["launch"]
        assert r["simulated_only"] == "0", f"launch {r['launch']}: config not found in {CONFIG.name}"
        assert {c: int(r[c]) for c in COUNTS} == {c: int(n[c]) for c in COUNTS}, f"launch {r['launch']}"


def test_committed_validation_counts_match_ncu():
    path = ROOT / "results" / "sim" / "rtx4060_laptop.csv"
    if not path.exists():
        pytest.skip("no validation run yet; run analysis/cachesim.py validate")
    rows = list(csv.DictReader(path.open()))
    assert {(r["kernel"], r["variant"], r["dims"]) for r in rows} == set(cases())
    for r in rows:
        where = (r["kernel"], r["variant"], r["dims"], r["launch"])
        assert r["sim_kernel"] == r["ncu_kernel"], where
        for c in COUNTS:
            assert r[c] == r["ncu_" + c], (where, c)


def test_report_classifies_phases_and_full_sector_stores():
    """The report's tables rest on these two: attention launches split by phase, and which
    launches store whole 32 B sectors (the alternative L2 write-hit definition)."""
    sys.path.insert(0, str(ROOT / "analysis"))
    import cachesim
    names = ["transpose_scale_kernel", "gemm_v0_kernel", "gemm_v0_kernel", "scale_kernel",
             "softmax_v0_kernel", "gemm_v0_kernel", "gemm_v0_kernel"]
    assert cachesim.phase_of([{"sim_kernel": k} for k in names]) == \
        ["prep", "QK", "QK", "scale", "softmax", "PV", "PV"]

    def launch(kernel, requests, sectors):
        return {"sim_kernel": kernel, "st_requests": str(requests), "st_sectors": str(sectors)}
    assert cachesim.full_sector_stores(launch("gemm_v3_kernel", 10, 160))      # 512 B in 16 sectors
    assert cachesim.full_sector_stores(launch("transpose_scale_kernel", 10, 40))  # 128 B in 4
    assert not cachesim.full_sector_stores(launch("gemm_v0_kernel", 10, 160))   # 8 B per sector
    assert not cachesim.full_sector_stores(launch("softmax_v0_kernel", 10, 320))  # 4 B per sector


def test_sweep_covers_both_gpus_and_every_tile():
    path = ROOT / "results" / "sim" / "sweep.csv"
    if not path.exists():
        pytest.skip("no sweep yet; run analysis/cachesim.py sweep")
    rows = list(csv.DictReader(path.open()))
    seen = {(r["gpu"], r["variant"], r["tile"]) for r in rows}
    for gpu in ("rtx4060_laptop", "rtx4070_laptop"):
        for t in ("8", "16", "32"):
            assert (gpu, "v2", t) in seen
        for bm in (64, 128):
            for bk in (8, 16, 32):
                assert (gpu, "v3", f"{bm}x{bm}x{bk}x8") in seen
    real = [r for r in rows if (r["variant"], r["tile"]) in (("v2", "32"), ("v3", "128x128x8x8"))]
    assert real and all(r["simulated_only"] == "0" for r in real)
