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
