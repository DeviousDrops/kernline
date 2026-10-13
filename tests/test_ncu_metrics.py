"""tools/ncu_metrics.py against a committed report whose numbers are already in results/."""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ncu_metrics  # noqa: E402

REPORT = ROOT / "results/ncu/rtx4070_laptop/gemm_v0_1024x1024x1024.ncu-repz"
NCU = shutil.which("ncu") or "/opt/nsight-compute/ncu"
pytestmark = pytest.mark.skipif(not Path(NCU).exists(), reason="ncu not installed")


def recorded() -> dict:
    """The results row tools/profile.sh wrote from REPORT; re-profiling rewrites both together."""
    with (ROOT / "results/rtx4070_laptop/gemm.csv").open(newline="") as f:
        return next(r for r in csv.DictReader(f)
                    if (r["variant"], r["dims"]) == ("v0", "1024x1024x1024"))


def test_extract_matches_recorded_row():
    m = ncu_metrics.extract(ncu_metrics.read_report(REPORT))
    assert m["kernels_in_report"] == 1
    row = recorded()
    assert m["dram_read_bytes"] + m["dram_write_bytes"] == int(row["dram_bytes"])
    assert m["achieved_occ_pct"] == pytest.approx(float(row["achieved_occ"]), abs=0.01)
    assert m["gpu_time_ms"] > 0
    assert m["block_size"] == 256


def test_top_stall_is_per_issued_instruction():
    # GEMM v4 waits on the tensor pipe. The per-warp latency metric this replaced ranked a rarer,
    # longer stall first and reported ~1e5 "cycles"; per issue it is math_pipe_throttle at ~15.
    m = ncu_metrics.extract(ncu_metrics.read_report(
        ROOT / "results/ncu/rtx4070_laptop/gemm_v4_4096x4096x4096.ncu-repz"))
    assert m["top_stall"] == "math_pipe_throttle"
    assert 5 < m["top_stall_per_issue"] < 50


def test_upsert_replaces_matching_row(tmp_path):
    out = tmp_path / "gemm.csv"
    row = "rtx4070_laptop,2490,8001,gemm,v0,1024x1024x1024,10.0,2147483648,12582912,,,"
    for _ in range(2):
        subprocess.run([sys.executable, ROOT / "tools/ncu_metrics.py", REPORT, "--kernel", "gemm",
                        "--variant", "v0", "--dims", "1024x1024x1024", "--bench-row", row,
                        "--results", out, "--metrics-csv", tmp_path / "metrics.csv"], check=True)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["dram_bytes"] == recorded()["dram_bytes"] and rows[0]["notes"] == "ncu"
    assert len(list(csv.DictReader((tmp_path / "metrics.csv").open()))) == 1
