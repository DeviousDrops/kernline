#!/usr/bin/env python3
"""Cache simulator vs Nsight Compute, launch by launch (docs/SIM.md).

    python analysis/cachesim.py validate [--gpu rtx4060_laptop]   every case in sim/cases.txt
    python analysis/cachesim.py sweep                             GEMM 2048^3 tile sweeps -> sweep.csv
    python analysis/cachesim.py report                            tables (stdout) + plots

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


def consistent(n: dict, key: str) -> bool:
    """False if ncu's summed hit count exceeds its total: the two came from different replay
    passes (docs/SIM.md), so their ratio isn't a hit rate."""
    h, t = RATES[key]
    return n[h] <= n[t]


def ncu_rate(n: dict, key: str) -> str:
    return f"{rate(n, *RATES[key]):.1f}" if consistent(n, key) else "n/a*"


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
        for key, (h, t) in RATES.items():
            ncu = ncu_rate(n, key) if n else "–"
            starred |= ncu == "n/a*"
            cells.append(f"{rate(s, h, t):.1f} / {ncu}")
        mb = [f"{s[c] / 1e6:.1f} / {n[c] / 1e6:.1f}" if n else f"{s[c] / 1e6:.1f} / –"
              for c in ("dram_read_bytes", "dram_write_bytes")]
        print(f"| {kernel} {variant} {dims} | {len(rs)} | {exact} | " + " | ".join(cells + mb) + " |")
    if starred:
        print("\n* ncu collected these counters over several replay passes; for this call a hit "
              "count exceeds its total, so the measured rate isn't meaningful (docs/SIM.md).")


SWEEP_GPUS = ("rtx4060_laptop", "rtx4070_laptop")
SWEEP = [("v2", t) for t in ("8", "16", "32")] + \
        [("v3", f"{bm},{bm},{bk},8") for bm in (64, 128) for bk in (8, 16, 32)]


def sweep() -> None:
    """GEMM 2048^3 at every tile on both SM counts (docs/SIM.md, Experiments 2)."""
    out = []
    for gpu in SWEEP_GPUS:
        for variant, tile in SWEEP:
            print(f"== {gpu} gemm {variant} {tile}", file=sys.stderr, flush=True)
            out += simulate(gpu, "gemm", variant, "2048x2048x2048", tile)
    write(SIMDIR / "sweep.csv", FIELDS, out)


def phase_of(rows: list[dict]) -> list[str]:
    """Attention launches by phase: prep, QK (GEMMs before softmax), scale, softmax, PV."""
    phases, seen_softmax = [], False
    for r in rows:
        k = r["sim_kernel"]
        if k.startswith("softmax"):
            phases.append("softmax")
            seen_softmax = True
        elif k.startswith("gemm"):
            phases.append("PV" if seen_softmax else "QK")
        else:
            phases.append("prep" if k.startswith("transpose") else "scale")
    return phases


# Kernels whose stores are 128-bit (STG.E.128); the rest store 32 bits per lane.
STORE_BYTES = {"gemm_v3": 16, "softmax_v2": 16, "softmax_v3": 16}


def full_sector_stores(r: dict) -> bool:
    """True if the launch's stores write whole 32 B sectors: a full warp of 4 B (16 B) lanes
    covers 4 (16) sectors per request exactly when its lanes are contiguous."""
    width = next((b for k, b in STORE_BYTES.items() if r["sim_kernel"].startswith(k)), 4)
    return int(r["st_sectors"]) == int(r["st_requests"]) * width


def wr_hits_if_full_sectors_hit(rows: list[dict]) -> int:
    """L2 write hits under an alternative definition: a write that covers its whole sector needs
    no DRAM fill, so it counts as a hit even if the sector was absent. The default model counts
    a write to an absent sector as a miss; this is computed from its counters, not simulated."""
    return sum(int(r["l2_wr_sectors"] if full_sector_stores(r) else r["l2_wr_hit_sectors"])
               for r in rows)


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, str(ROOT / "analysis"))
    from roofline import BLUES, GRID, INK, INK_2, INK_3, SURFACE

    rows = list(csv.DictReader((SIMDIR / "rtx4060_laptop.csv").open()))
    print("## Validation (rtx4060_laptop, per call, counters summed over launches)\n")
    print_validation(rows)

    calls = defaultdict(list)
    kernels = defaultdict(list)
    for r in rows:
        calls[(r["kernel"], r["variant"], r["dims"])].append(r)
        kernels[r["sim_kernel"]].append(r)

    print("\n## Per kernel (rtx4060_laptop, every launch of a kernel in the 22 calls summed)\n")
    print("| kernel | launches | L1 ld hit sim / ncu (%) | L2 rd hit sim / ncu (%) "
          "| L2 rd misses sim / ncu (M sectors) "
          "| L2 wr hit sim / sim if full-sector writes hit / ncu (%) "
          "| launches where ncu L2 wr hits = store sectors |")
    print("|---|---|---|---|---|---|---|")
    for k, rs in kernels.items():
        s, n = totals(rs), totals(rs, "ncu_")
        alt = 100 * wr_hits_if_full_sectors_hit(rs) / s["l2_wr_sectors"]
        same = sum(r["ncu_l2_wr_hit_sectors"] == r["ncu_st_sectors"] for r in rs)
        miss = [(t["l2_rd_sectors"] - t["l2_rd_hit_sectors"]) / 1e6 for t in (s, n)]
        print(f"| {k} | {len(rs)} | {rate(s, *RATES['L1 ld hit']):.1f} / {ncu_rate(n, 'L1 ld hit')}"
              f" | {rate(s, *RATES['L2 rd hit']):.1f} / {ncu_rate(n, 'L2 rd hit')}"
              f" | {miss[0]:.2f} / {miss[1]:.2f}"
              f" | {rate(s, *RATES['L2 wr hit']):.1f} / {alt:.1f} / {ncu_rate(n, 'L2 wr hit')}"
              f" | {same} of {len(rs)} |")
    print("\nn/a*: as above, ncu's summed hit count exceeds its total (replay passes).")

    print("\n## DRAM reads where the simulator, ncu's DRAM bytes or ncu's L2 read misses differ "
          "by more than 5%\n")
    print("| call | sim DRAM rd MB | ncu dram__bytes_read MB | ncu L2 read misses × 32 B (MB) |")
    print("|---|---|---|---|")
    for (kernel, variant, dims), rs in calls.items():
        s, n = totals(rs), totals(rs, "ncu_")
        mb = [s["dram_read_bytes"], n["dram_read_bytes"],
              32 * (n["l2_rd_sectors"] - n["l2_rd_hit_sectors"])]
        if max(mb) > 1.05 * min(mb):
            print(f"| {kernel} {variant} {dims} | " + " | ".join(f"{b / 1e6:.1f}" for b in mb) + " |")

    print("\n## Scheduling: attention v1 (per-head launches) vs v2 (one batched grid)\n")
    print("| N | phase | L2 rd hit v1 sim / ncu (%) | L2 rd hit v2 sim / ncu (%) | DRAM rd MB v1 / v2 (sim) |")
    print("|---|---|---|---|---|")
    for N in (256, 512, 1024, 2048, 4096):
        by = {}
        for v in ("v1", "v2"):
            rs = calls.get(("attention", v, f"12x{N}x64"), [])
            by[v] = defaultdict(list)
            for r, p in zip(rs, phase_of(rs)):
                by[v][p].append(r)
        for p in ("QK", "PV"):
            s1, s2 = totals(by["v1"][p]), totals(by["v2"][p])
            n1, n2 = totals(by["v1"][p], "ncu_"), totals(by["v2"][p], "ncu_")
            print(f"| {N} | {p} | {rate(s1, *RATES['L2 rd hit']):.1f} / {ncu_rate(n1, 'L2 rd hit')}"
                  f" | {rate(s2, *RATES['L2 rd hit']):.1f} / {ncu_rate(n2, 'L2 rd hit')}"
                  f" | {s1['dram_read_bytes'] / 1e6:.1f} / {s2['dram_read_bytes'] / 1e6:.1f} |")

    print("\n## P·V's P loads (operand A of the PV GEMM), attention v2\n")
    print("| N | sectors / request | L1 hit % | L2 hit % of L1 misses | ncu L2 rd hit % (whole launch) |")
    print("|---|---|---|---|---|")
    for N in (256, 512, 1024, 2048, 4096):
        rs = calls.get(("attention", "v2", f"12x{N}x64"), [])
        pv = [r for r, p in zip(rs, phase_of(rs)) if p == "PV"]
        if not pv:
            continue
        r = pv[0]
        req, sec, l1, l2 = (int(r[c]) for c in OPERAND_A)
        n = totals(pv, "ncu_")
        print(f"| {N} | {sec / req:.1f} | {100 * l1 / sec:.1f} | {100 * l2 / max(1, sec - l1):.1f}"
              f" | {ncu_rate(n, 'L2 rd hit')} |")

    sweep_rows = list(csv.DictReader((SIMDIR / "sweep.csv").open()))
    sw = {(r["gpu"], r["variant"], r["tile"]): r for r in sweep_rows}
    labels = [f"{v} {t.replace(',', 'x')}" for v, t in SWEEP]
    print("\n## GEMM 2048³ by tile (simulated)\n")
    print("| tile | real kernel | resident blocks / SM | L1 KB | loads (M sectors) "
          "| L1 ld hit 24 / 36 SMs (%) | L1 → L2 reads MB 24 / 36 SMs "
          "| L2 rd hit 24 / 36 SMs (%) | DRAM rd MB 24 / 36 SMs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (v, t), label in zip(SWEEP, labels):
        a, b = (sw[(gpu, v, t.replace(",", "x"))] for gpu in SWEEP_GPUS)
        print(f"| {label} | {'no' if a['simulated_only'] == '1' else 'yes'} | {a['resident']} | {a['l1_kb']}"
              f" | {int(a['ld_sectors']) / 1e6:.1f}"
              f" | {rate(a, *RATES['L1 ld hit']):.1f} / {rate(b, *RATES['L1 ld hit']):.1f}"
              f" | {32 * int(a['l2_rd_sectors']) / 1e6:.0f} / {32 * int(b['l2_rd_sectors']) / 1e6:.0f}"
              f" | {rate(a, *RATES['L2 rd hit']):.1f} / {rate(b, *RATES['L2 rd hit']):.1f}"
              f" | {int(a['dram_read_bytes']) / 1e6:.1f} / {int(b['dram_read_bytes']) / 1e6:.1f} |")

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_3)
        ax.tick_params(colors=INK_2, labelsize=8)

    # Plot 1: per-call hit rates, simulated vs measured. Calls whose ncu counters are
    # inconsistent for that rate are left out rather than plotted at a meaningless value.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), facecolor=SURFACE)
    for ax, key in zip(axes, ("L1 ld hit", "L2 rd hit")):
        gaps = defaultdict(list)  # (kernel, variant) -> points more than 5 points off the diagonal
        for (kernel, variant, dims), rs in calls.items():
            s, n = totals(rs), totals(rs, "ncu_")
            if not all(r.get("ncu_kernel") for r in rs) or not consistent(n, key):
                continue
            x, y = rate(n, *RATES[key]), rate(s, *RATES[key])
            ax.scatter(x, y, s=36, zorder=3, color=BLUES[2] if kernel == "gemm" else BLUES[5],
                       marker="o" if kernel == "gemm" else "s")
            if abs(y - x) > 5:
                gaps[(kernel, variant)].append((x, y))
        for (kernel, variant), pts in gaps.items():
            # Label beside the cluster's outer point on the side facing the plot's centre.
            right = sum(p[0] for p in pts) / len(pts) > 50
            x = min(p[0] for p in pts) if right else max(p[0] for p in pts)
            y = sum(p[1] for p in pts) / len(pts)
            ax.annotate(f"{kernel} {variant}", xy=(x, y), xytext=(-10 if right else 10, 0),
                        textcoords="offset points", ha="right" if right else "left", va="center",
                        fontsize=8, color=INK_2)
        ax.plot([0, 100], [0, 100], color=INK_3, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_aspect("equal")
        ax.set_title(f"{key} rate", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("ncu (%)", fontsize=9, color=INK_2)
        ax.set_ylabel("simulated (%)", fontsize=9, color=INK_2)
        ax.grid(True, color=GRID, lw=0.8)
        style(ax)
    axes[0].legend(handles=[
        Line2D([], [], ls="", marker="o", color=BLUES[2], label="GEMM call"),
        Line2D([], [], ls="", marker="s", color=BLUES[5], label="attention call"),
        Line2D([], [], color=INK_3, lw=1, ls=(0, (4, 3)), label="sim = ncu")],
        frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Cache simulator vs ncu, rtx4060_laptop, per call", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.text(0.01, 0.01, "counters summed over each call's launches · labelled: more than 5 "
             "points off the diagonal · a call whose ncu hit count exceeds its total "
             "(replay passes) would be left out", fontsize=7.5, color=INK_3, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / "analysis" / "plots" / "cachesim_validation.png", dpi=150)
    plt.close(fig)

    # Plot 2: tile sweeps, L1 / L2 hit rate and L2 read traffic per tile, 24 vs 36 SMs. DRAM
    # reads are the same for every bar at 2048^3 (A and B once), so they go in the footnote.
    dram = {int(r["dram_read_bytes"]) for r in sweep_rows}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), facecolor=SURFACE)
    for ax, (title, fn) in zip(axes, (
            ("L1 load hit (%)", lambda r: rate(r, *RATES["L1 ld hit"])),
            ("L2 read hit (%)", lambda r: rate(r, *RATES["L2 rd hit"])),
            ("L1 → L2 reads (MB)", lambda r: 32 * int(r["l2_rd_sectors"]) / 1e6))):
        for k, gpu in enumerate(SWEEP_GPUS):
            ys = [fn(sw[(gpu, v, t.replace(",", "x"))]) for v, t in SWEEP]
            ax.bar([i + 0.4 * k - 0.2 for i in range(len(SWEEP))], ys, width=0.38,
                   color=BLUES[2 + 2 * k], label=f"{sm_count(gpu)} SMs")
        ax.set_xticks(range(len(SWEEP)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("GEMM 2048³ by tile (simulated; v2 32 and v3 128x128x8x8 are the real kernels)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "DRAM reads: " + " / ".join(f"{b / 1e6:.1f} MB" for b in sorted(dram))
             + " for every tile on both SM counts · no bar = 0%", fontsize=7.5, color=INK_3,
             va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / "analysis" / "plots" / "cachesim_tiles.png", dpi=150)
    plt.close(fig)
    print("\nanalysis/plots/cachesim_validation.png\nanalysis/plots/cachesim_tiles.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["validate", "sweep", "report"])
    ap.add_argument("--gpu", default="rtx4060_laptop")
    args = ap.parse_args()
    if args.command == "validate":
        validate(args.gpu)
    elif args.command == "sweep":
        sweep()
    else:
        report()


if __name__ == "__main__":
    main()
