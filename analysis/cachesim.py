#!/usr/bin/env python3
"""Cache simulator vs Nsight Compute, launch by launch (docs/SIM.md).

    python analysis/cachesim.py validate [--gpu rtx4060_laptop]   every case in sim/cases.txt
    python analysis/cachesim.py sweep                             GEMM tile sweeps (Task 6)
    python analysis/cachesim.py report                            tables + plots (Task 6)

validate runs build/cachesim on the GPU's SM count with the 4060's measured launch configs
(both laptops have the same per-SM limits) and writes results/sim/<gpu>.csv: the simulator's
raw counters next to ncu's (ncu_*), where tools/sim_ncu.sh recorded a reference for that GPU.
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "build" / "cachesim"
SIMDIR = ROOT / "results" / "sim"
CONFIG = SIMDIR / "launch_config_rtx4060_laptop.csv"
ID = ["gpu", "kernel", "variant", "dims", "tile", "launch", "sim_kernel", "simulated_only",
      "resident", "l1_kb"]
RAW = ["ld_requests", "ld_sectors", "l1_hit_sectors", "st_requests", "st_sectors",
       "l2_rd_sectors", "l2_rd_hit_sectors", "l2_wr_sectors", "l2_wr_hit_sectors",
       "dram_read_bytes", "dram_write_bytes"]
OPERAND_A = ["a_ld_requests", "a_ld_sectors", "a_l1_hit_sectors", "a_l2_rd_hit_sectors"]
FIELDS = ID + ["ncu_kernel"] + [x for c in RAW for x in (c, "ncu_" + c)] + OPERAND_A
COUNTS = ["ld_requests", "ld_sectors", "st_requests", "st_sectors"]
RATES = {"L1 ld hit": ("l1_hit_sectors", "ld_sectors"),
         "L2 rd hit": ("l2_rd_hit_sectors", "l2_rd_sectors"),
         "L2 wr hit": ("l2_wr_hit_sectors", "l2_wr_sectors")}


def cases():
    for line in (ROOT / "sim" / "cases.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            yield tuple(line.split())


def sm_count(gpu: str) -> int:
    return json.loads((ROOT / "specs" / f"{gpu}.json").read_text())["device"]["sm_count"]


def simulate(gpu: str, kernel: str, variant: str, dims: str, tile: str | None = None) -> list[dict]:
    cmd = [str(SIM), kernel, variant, dims, "--gpu", gpu, "--sms", str(sm_count(gpu)),
           "--launch-config", str(CONFIG)]
    if tile:
        cmd += ["--tile", tile]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout
    return list(csv.DictReader(out.splitlines()))


def rate(r: dict, num: str, den: str, prefix: str = "") -> float:
    d = float(r.get(prefix + den) or 0)
    return 100 * float(r[prefix + num]) / d if d else float("nan")


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def totals(rows: list[dict], prefix: str = "") -> dict:
    return {c: sum(int(r[prefix + c] or 0) for r in rows) for c in RAW}


def validate(gpu: str) -> None:
    ref_path = SIMDIR / f"ncu_{gpu}.csv"
    ref = {}
    if ref_path.exists():
        ref = {(r["kernel"], r["variant"], r["dims"], r["launch"]): r
               for r in csv.DictReader(ref_path.open())}
    out = []
    for kernel, variant, dims in cases():
        print(f"== {kernel} {variant} {dims}", file=sys.stderr, flush=True)
        for r in simulate(gpu, kernel, variant, dims):
            n = ref.get((kernel, variant, dims, r["launch"]))
            if n:
                r["ncu_kernel"] = n["kernel_name"]
                r.update({"ncu_" + c: n[c] for c in RAW})
            out.append(r)
    write(SIMDIR / f"{gpu}.csv", FIELDS, out)
    print_validation(out)


def print_validation(rows: list[dict]) -> None:
    calls = defaultdict(list)
    for r in rows:
        calls[(r["kernel"], r["variant"], r["dims"])].append(r)
    print("| call | launches | counts exact | " + " | ".join(f"{k} sim / ncu (%)" for k in RATES)
          + " | DRAM rd MB sim / ncu (approx.) | DRAM wr MB sim / ncu (approx.) |")
    print("|---" * (5 + len(RATES)) + "|")
    starred = False
    for (kernel, variant, dims), rs in calls.items():
        have_ref = all(r.get("ncu_kernel") for r in rs)
        s, n = totals(rs), totals(rs, "ncu_") if have_ref else None
        exact = "–" if not have_ref else "yes" if all(
            r["sim_kernel"] == r["ncu_kernel"] and all(r[c] == r["ncu_" + c] for c in COUNTS)
            for r in rs) else "**no**"
        cells = []
        for h, t in RATES.values():
            if not n:
                ncu = "–"
            elif n[h] > n[t]:
                ncu, starred = "n/a*", True
            else:
                ncu = f"{rate(n, h, t):.1f}"
            cells.append(f"{rate(s, h, t):.1f} / {ncu}")
        mb = [f"{s[c] / 1e6:.1f} / {n[c] / 1e6:.1f}" if n else f"{s[c] / 1e6:.1f} / –"
              for c in ("dram_read_bytes", "dram_write_bytes")]
        print(f"| {kernel} {variant} {dims} | {len(rs)} | {exact} | " + " | ".join(cells + mb) + " |")
    if starred:
        print("\n* ncu collected these counters over several replay passes; for this call a hit "
              "count exceeds its total, so the measured rate isn't meaningful (docs/SIM.md).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["validate"])
    ap.add_argument("--gpu", default="rtx4060_laptop")
    args = ap.parse_args()
    if args.command == "validate":
        validate(args.gpu)


if __name__ == "__main__":
    main()
