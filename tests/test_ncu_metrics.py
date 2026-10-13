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


def test_extract_matches_recorded_row():
    m = ncu_metrics.extract(ncu_metrics.read_report(REPORT))
    assert m["kernels_in_report"] == 1
    assert m["dram_read_bytes"] == 40713088          # value recorded in results/rtx4070_laptop
    assert m["achieved_occ_pct"] == pytest.approx(98.02, abs=0.01)
    assert m["gpu_time_ms"] == pytest.approx(10.03, abs=0.01)
    assert m["block_size"] == 256


def test_upsert_replaces_matching_row(tmp_path):
    out = tmp_path / "gemm.csv"
    row = "rtx4070_laptop,2490,8001,gemm,v0,1024x1024x1024,10.0,2147483648,12582912,,,"
    for _ in range(2):
        subprocess.run([sys.executable, ROOT / "tools/ncu_metrics.py", REPORT, "--kernel", "gemm",
                        "--variant", "v0", "--dims", "1024x1024x1024", "--bench-row", row,
                        "--results", out, "--metrics-csv", tmp_path / "metrics.csv"], check=True)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["dram_bytes"] == "40713088" and rows[0]["notes"] == "ncu"
    assert len(list(csv.DictReader((tmp_path / "metrics.csv").open()))) == 1
