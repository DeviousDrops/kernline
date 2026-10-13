#!/usr/bin/env python3
"""Pull the frozen metric set (docs/DESIGN.md) out of an .ncu-repz and record it.

    ncu_metrics.py REPORT                         print the metrics as key=value lines
    ncu_metrics.py REPORT --kernel gemm --variant v0 --dims 1024x1024x1024 \
        --bench-row "<row from bench>" --results results/<gpu>/gemm.csv \
        --metrics-csv results/ncu/<gpu>/metrics.csv

The second form upserts the bench row (dram_bytes and achieved_occ filled from ncu) into the
results CSV, and the full metric row into metrics.csv. Rows are keyed by gpu/kernel/variant/dims,
so re-profiling replaces instead of duplicating.
"""
import argparse
import csv
import io
import shutil
import subprocess
import sys
from pathlib import Path

RESULTS_HEADER = ["gpu", "sm_clk_mhz", "mem_clk_mhz", "kernel", "variant", "dims", "time_ms",
                  "flop", "bytes_theory", "dram_bytes", "achieved_occ", "notes"]

# Warps stalled on each reason per issue cycle (what Nsight Compute's "Warp State" chart shows):
# a value of 21 means an average of 21 warp-cycles of that stall for every instruction issued.
# Not smsp__average_warp_latency_issue_stalled_*, which is total cycles per warp and grows with
# kernel length.
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"

# metrics.csv column -> (ncu metric, how to combine across kernels in one report)
COLUMNS = {
    "gpu_time_ms": ("gpu__time_duration.sum", "sum"),
    "dram_read_bytes": ("dram__bytes_read.sum", "sum"),
    "dram_write_bytes": ("dram__bytes_write.sum", "sum"),
    "sm_throughput_pct": ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "time_avg"),
    "achieved_occ_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active", "time_avg"),
    "l1_hit_pct": ("l1tex__t_sector_hit_rate.pct", "time_avg"),
    "l2_hit_pct": ("lts__t_sector_hit_rate.pct", "time_avg"),
    "smem_bank_conflicts": ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "sum"),
    "regs_per_thread": ("launch__registers_per_thread", "max"),
    "smem_per_block_bytes": ("launch__shared_mem_per_block", "max"),
    "block_size": ("launch__block_size", "max"),
    "grid_size": ("launch__grid_size", "max"),
}
METRICS_HEADER = (["gpu", "kernel", "variant", "dims", "kernels_in_report"] + list(COLUMNS)
                  + ["top_stall", "top_stall_per_issue"])


def ncu_binary() -> str:
    for cand in (shutil.which("ncu"), "/opt/nsight-compute/ncu", "/opt/cuda/nsight_compute/ncu"):
        if cand and Path(cand).exists():
            return cand
    sys.exit("ncu not found on PATH")


def read_report(path: Path) -> list[dict]:
    """One dict per profiled kernel, values in base units (bytes, ns, %)."""
    out = subprocess.run([ncu_binary(), "--import", str(path), "--csv", "--page", "raw",
                          "--print-units", "base", "--print-fp"],
                         check=True, capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    header, data = rows[0], rows[2:]  # rows[1] holds the units
    return [dict(zip(header, r)) for r in data if r]


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def extract(kernels: list[dict]) -> dict:
    """Combine every kernel in the report into one row (cuBLAS may launch more than one)."""
    times = [_num(k.get("gpu__time_duration.sum")) or 0.0 for k in kernels]
    total_t = sum(times) or 1.0
    row = {"kernels_in_report": len(kernels)}
    for col, (metric, how) in COLUMNS.items():
        vals = [_num(k.get(metric)) for k in kernels]
        if all(v is None for v in vals):
            row[col] = ""
            continue
        vals = [v or 0.0 for v in vals]
        if how == "sum":
            row[col] = sum(vals)
        elif how == "max":
            row[col] = max(vals)
        else:
            row[col] = sum(v * t for v, t in zip(vals, times)) / total_t
    row["gpu_time_ms"] = row["gpu_time_ms"] / 1e6 if row["gpu_time_ms"] != "" else ""  # ns -> ms

    stalls = {}
    for k, t in zip(kernels, times):
        for name, val in k.items():
            if (name.startswith(STALL_PREFIX) and name.endswith(STALL_SUFFIX)
                    and _num(val) is not None):
                reason = name[len(STALL_PREFIX):-len(STALL_SUFFIX)]
                stalls[reason] = stalls.get(reason, 0.0) + _num(val) * t / total_t
    stalls.pop("selected", None)  # "selected" is issuing, not stalling
    if stalls:
        top = max(stalls, key=stalls.get)
        row["top_stall"], row["top_stall_per_issue"] = top, stalls[top]
    else:
        row["top_stall"] = row["top_stall_per_issue"] = ""
    return row


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.0f}" if v.is_integer() or abs(v) >= 1e6 else f"{v:.4g}"
    return str(v)


def upsert(path: Path, header: list[str], row: dict, key: tuple[str, ...]) -> None:
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if tuple(r.get(k, "") for k in key) != tuple(str(row[k]) for k in key)]
    rows.append({h: _fmt(row.get(h, "")) for h in header})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", type=Path)
    ap.add_argument("--kernel")
    ap.add_argument("--variant")
    ap.add_argument("--dims")
    ap.add_argument("--bench-row", help="CSV row printed by build/bench")
    ap.add_argument("--results", type=Path, help="results/<gpu>/<kernel>.csv to upsert into")
    ap.add_argument("--metrics-csv", type=Path, help="results/ncu/<gpu>/metrics.csv to upsert into")
    args = ap.parse_args()

    m = extract(read_report(args.report))
    if not (args.bench_row or args.metrics_csv):
        for k, v in m.items():
            print(f"{k}={_fmt(v)}")
        return

    if args.bench_row and args.results:
        bench = dict(zip(RESULTS_HEADER, next(csv.reader([args.bench_row]))))
        bench["dram_bytes"] = m["dram_read_bytes"] + m["dram_write_bytes"]
        bench["achieved_occ"] = m["achieved_occ_pct"]
        bench["notes"] = "ncu"
        upsert(args.results, RESULTS_HEADER, bench, ("gpu", "kernel", "variant", "dims"))
        gpu = bench["gpu"]
    else:
        gpu = args.metrics_csv.parent.name
    if args.metrics_csv:
        m.update(gpu=gpu, kernel=args.kernel, variant=args.variant, dims=args.dims)
        upsert(args.metrics_csv, METRICS_HEADER, m, ("gpu", "kernel", "variant", "dims"))


if __name__ == "__main__":
    main()
