"""Measured rows the model is tuned and validated on: results/<gpu>/*.csv (frozen CSV schema)."""
import csv
from dataclasses import dataclass
from pathlib import Path

from model import kernels

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Row:
    gpu: str
    kernel: str
    variant: str
    dims: tuple
    time_ms: float
    sm_mhz: float
    mem_mhz: float

    @property
    def dims_str(self) -> str:
        return "x".join(map(str, self.dims))


def rows(gpu: str, modeled_only: bool = True) -> list[Row]:
    """Every result row for `gpu`; by default only kernels and variants that have a descriptor."""
    out = []
    for path in sorted((ROOT / "results" / gpu).glob("*.csv")):
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                kernel, variant = r["kernel"], r["variant"]
                if modeled_only and (kernel not in kernels.KERNELS
                                     or variant not in kernels.get(kernel).VARIANTS):
                    continue
                out.append(Row(r["gpu"], kernel, variant, kernels.parse_dims(r["dims"]),
                               float(r["time_ms"]), float(r["sm_clk_mhz"] or 0),
                               float(r["mem_clk_mhz"] or 0)))
    return out
