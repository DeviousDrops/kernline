#!/usr/bin/env python3
"""Where unfused attention spends its time and DRAM traffic, phase by phase, from the ncu reports.

    python analysis/attention_breakdown.py [--gpu rtx4060_laptop] [--variant v2]

Reads results/ncu/<gpu>/attention_<variant>_<heads>x<N>x<d>.ncu-repz (one profiled call each,
from tools/profile.sh) and splits the call's kernels into phases by name and launch order:

    prep     K transpose (with the 1/sqrt(d) scale folded in for v1/v2)
    QK^T     GEMM kernels launched before softmax
    scale    v0's separate pass over the scores (PyTorch: its elementwise kernels)
    softmax
    PV       GEMM kernels launched after softmax

Prints a markdown table per variant (time and DRAM bytes per phase, achieved FLOP/byte against the
GPU's ridge point) and writes analysis/plots/attention_<gpu>_<variant>_breakdown.png. Times are ncu's
gpu__time_duration at base clocks, so they're for proportions, not absolute speed.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ncu_metrics  # noqa: E402

PHASES = ["prep", "QK^T", "scale", "softmax", "PV"]
GEMM = re.compile(r"gemm|sgemm|cutlass|xmma|bmm|ampere_|sm\d+_", re.I)

# Reference palette (dataviz skill), categorical slots 1-5 in fixed order; ink and surface.
COLORS = {"prep": "#e87ba4", "QK^T": "#2a78d6", "scale": "#eda100", "softmax": "#eb6834",
          "PV": "#1baf7a"}
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"


def _num(s) -> float:
    return float(str(s).replace(",", "") or 0)


def phases(kernels: list[dict]) -> dict[str, dict]:
    """Sum time (ms) and DRAM bytes per phase. Kernels arrive in launch order (ncu ID)."""
    kernels = sorted(kernels, key=lambda k: int(k.get("ID", 0)))
    seen_softmax = False
    out = {p: {"ms": 0.0, "bytes": 0.0, "kernels": 0} for p in PHASES}
    for k in kernels:
        name = k.get("Kernel Name", "")
        if "softmax" in name.lower():
            phase, seen_softmax = "softmax", True
        elif "transpose" in name.lower():
            phase = "prep"
        elif GEMM.search(name):
            phase = "PV" if seen_softmax else "QK^T"
        else:
            phase = "scale"
        out[phase]["ms"] += _num(k.get("gpu__time_duration.sum")) / 1e6
        out[phase]["bytes"] += _num(k.get("dram__bytes_read.sum")) + _num(k.get("dram__bytes_write.sum"))
        out[phase]["kernels"] += 1
    return out


def flops(phase: str, heads: int, N: int, d: int) -> float:
    return {"QK^T": 2.0 * heads * N * N * d, "PV": 2.0 * heads * N * N * d,
            "softmax": 5.0 * heads * N * N}.get(phase, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", default=None, help="default: every GPU with attention reports")
    ap.add_argument("--variant", action="append", help="default: every variant found")
    args = ap.parse_args()

    gpus = [args.gpu] if args.gpu else sorted(p.name for p in (ROOT / "results/ncu").iterdir())
    for gpu in gpus:
        reports = sorted((ROOT / "results/ncu" / gpu).glob("attention_*.ncu-repz"))
        if not reports:
            continue
        spec = json.loads((ROOT / "specs" / f"{gpu}.json").read_text())["ceilings"]
        ridge = spec["fp32_gflops"] / spec["dram_read_gbs"]
        by_variant: dict[str, list] = {}
        for rep in reports:
            _, variant, dims = rep.stem.split("_", 2)
            if args.variant and variant not in args.variant:
                continue
            heads, N, d = map(int, dims.split("x"))
            by_variant.setdefault(variant, []).append((N, heads, d, phases(ncu_metrics.read_report(rep))))
        for variant, rows in sorted(by_variant.items()):
            rows.sort()
            print(f"\n### {gpu} attention {variant} (ridge {ridge:.0f} FLOP/byte)\n")
            print("| N | " + " | ".join(f"{p} ms (%)" for p in PHASES) + " | N² traffic share | "
                  "QK^T FLOP/B | PV FLOP/B |")
            print("|---" * (len(PHASES) + 4) + "|")
            for N, heads, d, ph in rows:
                total = sum(v["ms"] for v in ph.values()) or 1.0
                nbytes = sum(v["bytes"] for v in ph.values()) or 1.0
                n2 = 4.0 * 4 * heads * N * N  # the 4 unfused passes over the scores
                ai = {p: flops(p, heads, N, d) / ph[p]["bytes"] if ph[p]["bytes"] else 0 for p in ("QK^T", "PV")}
                cells = " | ".join(f"{ph[p]['ms']:.3f} ({ph[p]['ms'] / total:.0%})" if ph[p]["kernels"] else "–"
                                   for p in PHASES)
                print(f"| {N} | {cells} | {min(1.0, n2 / nbytes):.0%} | {ai['QK^T']:.0f} | {ai['PV']:.0f} |")
            plot(gpu, variant, rows)


def plot(gpu: str, variant: str, rows: list) -> None:
    Ns = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    bottoms = [0.0] * len(rows)
    for p in PHASES:
        shares = []
        for _, _, _, ph in rows:
            total = sum(v["ms"] for v in ph.values()) or 1.0
            shares.append(100 * ph[p]["ms"] / total)
        if not any(shares):
            continue
        x = range(len(rows))
        ax.bar(x, shares, bottom=bottoms, color=COLORS[p], width=0.62, label=p,
               edgecolor=SURFACE, linewidth=2)
        bottoms = [b + s for b, s in zip(bottoms, shares)]
    ax.set_xticks(range(len(rows)), [str(n) for n in Ns])
    ax.set_xlabel("sequence length N  (12 heads, d = 64)", fontsize=9, color=INK_2)
    ax.set_ylabel("share of kernel time (%)", fontsize=9, color=INK_2)
    ax.set_ylim(0, 100)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_title(f"{gpu}: unfused attention {variant}, time by phase", fontsize=10, color=INK,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.tight_layout()
    out = ROOT / "analysis" / "plots" / f"attention_{gpu}_{variant}_breakdown.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
