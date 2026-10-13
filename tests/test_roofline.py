"""analysis/roofline.py data selection: no GPU needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import roofline  # noqa: E402

ROWS = [
    {"variant": "cublas", "dims": "1024x1024x1024", "flop": "2e9", "time_ms": "1", "bytes_theory": "1e7", "dram_bytes": ""},
    {"variant": "v1", "dims": "4096x4096x4096", "flop": "1.28e11", "time_ms": "100", "bytes_theory": "2e8", "dram_bytes": "6.4e9"},
    {"variant": "v1", "dims": "1024x1024x1024", "flop": "2e9", "time_ms": "5", "bytes_theory": "1e7", "dram_bytes": ""},
    {"variant": "v10", "dims": "64x64x64", "flop": "5e5", "time_ms": "0.01", "bytes_theory": "5e4", "dram_bytes": ""},
    {"variant": "v0", "dims": "1024x1024x1024", "flop": "2e9", "time_ms": "10", "bytes_theory": "1e7", "dram_bytes": "4e7"},
]


def test_pick_rows_takes_largest_dims_and_orders_variants_numerically():
    picked = roofline.pick_rows(ROWS, None)
    assert list(picked) == ["v0", "v1", "v10", "cublas"]
    assert picked["v1"]["dims"] == "4096x4096x4096"


def test_pick_rows_honours_requested_dims():
    assert set(roofline.pick_rows(ROWS, "1024x1024x1024")) == {"v0", "v1", "cublas"}


def test_point_prefers_measured_dram_bytes():
    ai, gflops, measured = roofline.point(ROWS[4])
    assert measured and ai == pytest.approx(2e9 / 4e7) and gflops == pytest.approx(200)
    ai, _, measured = roofline.point(ROWS[0])
    assert not measured and ai == pytest.approx(2e9 / 1e7)
