#!/usr/bin/env python3
"""Roofline charts from committed results: one figure per kernel, one panel per GPU.

    python analysis/roofline.py                 # every kernel in results/, largest dims each
    python analysis/roofline.py --kernel gemm --dims 2048x2048x2048

Inputs: specs/<gpu>.json (measured ceilings from tools/ceilings) and results/<gpu>/<kernel>.csv
(bench rows; dram_bytes filled by tools/profile.sh). Output: analysis/plots/<kernel>.png.

Encoding: x = arithmetic intensity (FLOP per DRAM byte), y = achieved GFLOP/s, both log.
Variants v0, v1, ... are one blue ramp, light to dark, so later steps read as "further along".
Library baselines (cuBLAS, torch) are neutral gray. Filled markers use ncu-measured DRAM bytes;
hollow markers fall back to theoretical bytes when the row has not been profiled yet.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill): light surface, ink, sequential blue ramp, neutral baseline.
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # steps 250..700
BASELINE = INK_2
BASELINES = {"cublas", "torch", "cudnn"}
# Variants that take FP16 inputs and run on tensor cores: they get the FP16 tensor-core roof.
TENSOR_CORE_VARIANTS = {"gemm": {"v4"}}


def load_specs() -> dict:
    specs = {}
    for p in sorted((ROOT / "specs").glob("*.json")):
        s = json.loads(p.read_text())
        if s.get("ceilings"):
            specs[s["gpu"]] = s
    return specs


def load_rows(gpu: str, kernel: str) -> list[dict]:
    path = ROOT / "results" / gpu / f"{kernel}.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _dims_size(d: str) -> int:
    out = 1
    for x in d.split("x"):
        out *= int(x)
    return out


def _variant_key(v: str):
    m = re.fullmatch(r"v(\d+)", v)
    return (0, int(m.group(1)), v) if m else (1, 0, v)


def pick_rows(rows: list[dict], dims: str | None) -> dict[str, dict]:
    """One row per variant: the requested dims, else the largest dims that variant has."""
    best: dict[str, dict] = {}
    for r in rows:
        if dims and r["dims"] != dims:
            continue
        cur = best.get(r["variant"])
        if cur is None or _dims_size(r["dims"]) > _dims_size(cur["dims"]):
            best[r["variant"]] = r
    return dict(sorted(best.items(), key=lambda kv: _variant_key(kv[0])))


def point(r: dict) -> tuple[float, float, bool]:
    flop, t_ms = float(r["flop"]), float(r["time_ms"])
    measured = bool(r.get("dram_bytes"))
    nbytes = float(r["dram_bytes"] if measured else r["bytes_theory"])
    return flop / nbytes, flop / (t_ms * 1e6), measured


def place_slope_labels(labels) -> None:
    """Write each bandwidth roof's label along its line. Call after layout is final: the on-screen
    angle of a log-log line depends on the axes' pixel aspect."""
    for ax, x, y, bw, text, color in labels:
        p0, p1 = ax.transData.transform([(x, bw * x), (x * 10, bw * x * 10)])
        angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.annotate(text, xy=(x, y), xytext=(0, 4), textcoords="offset points", fontsize=8,
                    color=color, rotation=angle, rotation_mode="anchor", ha="left", va="bottom")


def declutter(texts, pad_pt: float = 2.0) -> None:
    """Nudge point labels down until none overlaps an earlier one (greedy, top to bottom).
    Call after layout is final, since it measures rendered extents."""
    if not texts:
        return
    fig = texts[0].figure
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    placed = []
    for t in sorted(texts, key=lambda t: -t.get_window_extent(renderer).y1):
        for _ in range(40):
            box = t.get_window_extent(renderer).expanded(1.0, 1.0)
            hit = next((p for p in placed if box.overlaps(p)), None)
            if hit is None:
                break
            dy = (box.y1 - hit.y0) / px_per_pt + pad_pt  # move just below the box it hits
            x, y = t.xyann
            t.xyann = (x, y - dy)
        placed.append(t.get_window_extent(renderer))


def draw_panel(ax, spec: dict, picked: dict[str, dict], xlim: tuple[float, float],
               kernel: str) -> tuple[list, list]:
    c = spec["ceilings"]
    peak = c["fp32_gflops"]
    tensor = [v for v in picked if v in TENSOR_CORE_VARIANTS.get(kernel, ())]
    tc_peak = c.get("fp16_tc_fp32acc_gflops") if tensor else None
    roofs = [("DRAM", c["dram_read_gbs"], "-", INK_2), ("L2", c["l2_read_gbs"], (0, (4, 3)), INK_3)]
    xs = [xlim[0] * (xlim[1] / xlim[0]) ** (i / 200) for i in range(201)]
    slope_labels = []
    # Labels sit low on each slope, below where memory-bound kernels land near the roof.
    for (name, bw, style, color), y in zip(roofs, (peak / 1000, peak / 150)):
        ax.plot(xs, [min(peak, bw * x) for x in xs], ls=style, lw=1.25, color=color, zorder=1)
        slope_labels.append((ax, y / bw, y, bw, f"{name} {bw:.0f} GB/s", color))
    ax.axhline(peak, color=INK_2, lw=1.25, zorder=1)
    ax.annotate(f"FP32 {peak / 1000:.1f} TFLOP/s", xy=(xlim[0], peak), xytext=(4, 4),
                textcoords="offset points", ha="left", fontsize=8, color=INK_2)
    if tc_peak:
        # FP16-input variants run on the tensor pipe, so their compute roof is this one; their
        # bandwidth roofs are the same lines, since the x axis is FLOP per byte actually moved.
        ax.axhline(tc_peak, color=INK_3, lw=1.25, ls=(0, (1, 2)), zorder=1)
        ax.annotate(f"FP16 tensor core {tc_peak / 1000:.1f} TFLOP/s ({', '.join(tensor)})",
                    xy=(xlim[0], tc_peak), xytext=(4, 4), textcoords="offset points",
                    ha="left", fontsize=8, color=INK_3)

    steps = [v for v in picked if v not in BASELINES]
    texts = []
    for v, r in picked.items():
        x, y, measured = point(r)
        if v in BASELINES:
            color = BASELINE
        else:
            i = steps.index(v)
            color = BLUES[min(len(BLUES) - 1, i * (len(BLUES) - 1) // max(1, len(steps) - 1))]
        ax.scatter([x], [y], s=64, marker="o", zorder=3, linewidths=2,
                   facecolors=color if measured else SURFACE, edgecolors=color)
        texts.append(ax.annotate(f"{v}  {y:,.0f}", xy=(x, y), xytext=(8, 0),
                                 textcoords="offset points", fontsize=8, color=INK,
                                 va="center", ha="left"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(peak / 2000, max(peak, tc_peak or 0) * 3)
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=8)
    clk = c.get("sm_clk_mhz")
    dims = sorted({r["dims"] for r in picked.values()}, key=_dims_size)
    ax.set_title(f"{spec['gpu']}  ({spec['device']['sm_count']} SMs, ceilings at {clk} MHz)\n"
                 f"{' / '.join(dims)}", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Arithmetic intensity (FLOP / DRAM byte)", fontsize=9, color=INK_2)
    return slope_labels, texts


def plot_kernel(kernel: str, specs: dict, dims: str | None, out_dir: Path) -> Path | None:
    panels = [(gpu, pick_rows(load_rows(gpu, kernel), dims)) for gpu in specs]
    panels = [(g, p) for g, p in panels if p]
    if not panels:
        return None
    ais = [point(r)[0] for _, p in panels for r in p.values()]
    xlim = (min(ais + [0.05]) / 3, max(ais + [200.0]) * 3)

    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.8), sharey=False,
                             squeeze=False, facecolor=SURFACE)
    labels, point_labels = [], []
    for ax, (gpu, picked) in zip(axes[0], panels):
        slopes, texts = draw_panel(ax, specs[gpu], picked, xlim, kernel)
        labels += slopes
        point_labels.append(texts)
    axes[0][0].set_ylabel("Achieved GFLOP/s", fontsize=9, color=INK_2)
    fig.suptitle(f"{kernel}: roofline", fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "labels: variant  GFLOP/s · filled = ncu-measured DRAM bytes · "
             "hollow = theoretical bytes\nblue light→dark = optimization steps · "
             "gray = library baseline",
             fontsize=7.5, color=INK_3, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    place_slope_labels(labels)
    for texts in point_labels:
        declutter(texts)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{kernel}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", help="only this kernel (default: every results/<gpu>/<kernel>.csv)")
    ap.add_argument("--dims", help="plot this dims string instead of each variant's largest")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "plots")
    args = ap.parse_args()

    specs = load_specs()
    kernels = [args.kernel] if args.kernel else sorted(
        {p.stem for gpu in specs for p in (ROOT / "results" / gpu).glob("*.csv")})
    for k in kernels:
        out = plot_kernel(k, specs, args.dims, args.out)
        print(out or f"{k}: no rows for any GPU with measured ceilings")


if __name__ == "__main__":
    main()
