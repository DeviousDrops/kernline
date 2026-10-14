"""Cross-check the occupancy calculator against Nsight Compute (PLAN-A week 5).

`python -m model occupancy` reads every committed report, records each kernel's launch shape and
ncu's theoretical occupancy in model/occupancy_ncu.csv, and prints any row the calculator gets
wrong. tests/test_model.py replays that fixture, so the check runs without ncu installed.
"""
import csv
import sys
from pathlib import Path

from model import spec
from model.occupancy import occupancy

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "occupancy_ncu.csv"
FIELDS = ["gpu", "kernel", "threads", "regs", "smem", "ncu_blocks_per_sm", "ncu_occupancy_pct",
          "ncu_waves_per_sm"]


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def collect() -> list[dict]:
    sys.path.insert(0, str(ROOT / "tools"))
    from ncu_metrics import read_report

    rows, seen = [], set()
    for report in sorted((ROOT / "results" / "ncu").glob("*/*.ncu-repz")):
        gpu = report.parent.name
        for k in read_report(report):
            name = k["Kernel Name"].split("(")[0].replace("void ", "")
            row = {
                "gpu": gpu, "kernel": name, "threads": int(_num(k["launch__block_size"])),
                "regs": int(_num(k["launch__registers_per_thread"])),
                "smem": int(_num(k["launch__shared_mem_per_block_static"])
                            + _num(k["launch__shared_mem_per_block_dynamic"])),
                "ncu_blocks_per_sm": int(min(_num(k[f"launch__occupancy_limit_{lim}"]) for lim in
                                             ("blocks", "registers", "shared_mem", "warps"))),
                "ncu_occupancy_pct": _num(k["sm__maximum_warps_per_active_cycle_pct"]),
                "ncu_waves_per_sm": _num(k["launch__waves_per_multiprocessor"]),
            }
            key = tuple(row[f] for f in FIELDS[:5])
            if key not in seen:  # one row per distinct launch shape
                seen.add(key)
                rows.append(row)
    return rows


def check(rows: list[dict]) -> list[str]:
    errors = []
    for r in rows:
        gpu = spec.load(r["gpu"])
        occ = occupancy(gpu, int(r["threads"]), int(r["regs"]), int(r["smem"]))
        pct = 100 * occ.fraction(gpu)
        if occ.blocks_per_sm != int(r["ncu_blocks_per_sm"]) or abs(pct - float(r["ncu_occupancy_pct"])) > 0.01:
            errors.append(f"{r['gpu']} {r['kernel']} threads={r['threads']} regs={r['regs']} "
                          f"smem={r['smem']}: model {occ.blocks_per_sm} blocks/SM ({pct:.2f}%), "
                          f"ncu {r['ncu_blocks_per_sm']} ({r['ncu_occupancy_pct']}%)")
    return errors


def main() -> int:
    rows = collect()
    with FIXTURE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    errors = check(rows)
    print(f"{len(rows)} distinct launch shapes in {FIXTURE.name}; {len(errors)} disagree with ncu")
    for e in errors:
        print("  " + e)
    return 1 if errors else 0
