"""Predicting result rows, error tables, tuning, the block comparison, and plots."""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from model import data, kernels, params as params_mod, spec, tiers
from model.kernels import block as block_desc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model"
CAP = math.log(3)  # tuning ignores how wrong a row is beyond 3x, so a few outliers can't steer it


def row_launches(r: data.Row, gpu, p) -> list:
    return kernels.get(r.kernel).launches(r.dims, r.variant, gpu, p)


def predict_rows(rows, gpu, p, launches=None) -> list[dict]:
    """Per row, seconds at each tier, at the clocks the row was measured at."""
    out = []
    for i, r in enumerate(rows):
        ls = launches[i] if launches else row_launches(r, gpu, p)
        sm, mem = r.sm_mhz or gpu.ceil_sm_mhz, r.mem_mhz or gpu.ceil_mem_mhz
        out.append({t: tiers.predict(ls, gpu, p, t, sm, mem)[0] for t in tiers.TIERS})
    return out


def ratio(pred: dict, r: data.Row, tier: str) -> float:
    return pred[tier] * 1e3 / r.time_ms


def write_predictions(gpu: str, rows, preds) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{gpu}_predictions.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["gpu", "kernel", "variant", "dims", "sm_clk_mhz", "mem_clk_mhz", "time_ms"]
                   + [f"{t}_ms" for t in tiers.TIERS])
        for r, pr in zip(rows, preds):
            w.writerow([gpu, r.kernel, r.variant, r.dims_str, int(r.sm_mhz), int(r.mem_mhz),
                        r.time_ms] + [f"{pr[t] * 1e3:.5g}" for t in tiers.TIERS])
    return path


def _pct(x: float) -> str:
    return f"{x:.0%}" if x < 10 else f"{x:.0f}x"


def error_table(rows, preds, gpu: str, by_variant: bool = True) -> str:
    """Markdown: per kernel (and variant), median |error| per tier, T2 bias and worst row."""
    groups = defaultdict(list)
    for r, pr in zip(rows, preds):
        groups[(r.kernel, r.variant if by_variant else "")].append((r, pr))
    head = ("| kernel | variant | rows | " + " | ".join(f"{t} median abs err" for t in tiers.TIERS)
            + " | T2 bias | T2 worst row |")
    lines = [f"Model vs measured, {gpu}. Error = predicted / measured - 1; bias = geometric mean "
             "of predicted / measured.", "", head, "|" + "---|" * (head.count("|") - 1)]

    def summarize(items, label, variant):
        cells = [f"{statistics.median(abs(ratio(pr, r, t) - 1) for r, pr in items):.0%}"
                 for t in tiers.TIERS]
        logs = [math.log(ratio(pr, r, "T2")) for r, pr in items]
        r_w, pr_w = max(items, key=lambda it: abs(math.log(ratio(it[1], it[0], "T2"))))
        worst = f"{r_w.dims_str} {ratio(pr_w, r_w, 'T2') - 1:+.0%}"
        return (f"| {label} | {variant} | {len(items)} | " + " | ".join(cells)
                + f" | {math.exp(statistics.fmean(logs)):.2f} | {worst} |")

    for (k, v), items in sorted(groups.items()):
        lines.append(summarize(items, k, v))
    lines.append(summarize(list(zip(rows, preds)), "**all**", ""))
    return "\n".join(lines)


GRID = {
    "l2_eff": [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    "launch_us": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5],
    "ldg_per_clk": [round(0.25 + 0.05 * i, 2) for i in range(16)],
    "lds_per_clk": [round(0.25 + 0.05 * i, 2) for i in range(16)],
    "narrow_eff": [round(0.15 + 0.05 * i, 2) for i in range(18)],
    "fma_eff": [round(0.4 + 0.025 * i, 3) for i in range(25)],
    "tc_eff": [round(0.2 + 0.025 * i, 3) for i in range(33)],
    "sat_warps": [4, 8, 12, 16, 20, 24, 32, 40, 48],
}


def tune(rows, gpu, rounds: int = 3) -> tuple[dict, float]:
    """Coordinate descent over GRID minimizing mean capped |log(pred / measured)| at T2."""
    p = dict(params_mod.DEFAULTS)
    cache = {}

    def loss(q):
        key = q["l2_eff"]
        if key not in cache:  # only l2_eff changes the launches; the rest only the tiers
            cache[key] = [row_launches(r, gpu, q) for r in rows]
        total = 0.0
        for r, ls in zip(rows, cache[key]):
            sm, mem = r.sm_mhz or gpu.ceil_sm_mhz, r.mem_mhz or gpu.ceil_mem_mhz
            t = tiers.predict(ls, gpu, q, "T2", sm, mem)[0]
            total += min(abs(math.log(t * 1e3 / r.time_ms)), CAP)
        return total / len(rows)

    best = loss(p)
    for _ in range(rounds):
        for name, values in GRID.items():
            for v in values:
                q = {**p, name: v}
                l = loss(q)
                if l < best - 1e-9:
                    best, p = l, q
    preds = predict_rows(rows, gpu, p, [cache[p["l2_eff"]][i] for i in range(len(rows))])
    med = statistics.median(abs(ratio(pr, r, "T2") - 1) for r, pr in zip(rows, preds))
    return p, med


def block_table(gpu_name: str, gpu, p) -> str:
    """D4: per (B, S): measured total and steps, the model's sum of steps, PyTorch eager/graph."""
    path = ROOT / "results" / gpu_name / "block.csv"
    if not path.exists():
        return f"no {path.relative_to(ROOT)}; run tools/suite.sh block"
    by_dims = defaultdict(dict)
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            by_dims[r["dims"]][r["variant"]] = r
    lines = [f"Transformer block on {gpu_name} (GPT-2 small layer, FP32). Times in ms; the model "
             "columns sum its per-step predictions at each step's measured clocks.", "",
             "| B×S×D×H | measured total | measured Σ steps | model Σ steps T0 | T1 | T2 | T2 vs "
             "total | PyTorch eager | PyTorch graph | ours vs eager |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for dims, vs in sorted(by_dims.items(), key=lambda kv: [int(x) for x in kv[0].split("x")]):
        B, S, D, H = map(int, dims.split("x"))
        steps = [vs[s] for s in block_desc.STEPS if s in vs]
        sums = {t: 0.0 for t in tiers.TIERS}
        rows_detail = []
        for r in steps:
            ls = block_desc.launches((B, S, D, H), r["variant"], gpu, p)
            sm, mem = float(r["sm_clk_mhz"]) or gpu.ceil_sm_mhz, float(r["mem_clk_mhz"]) or gpu.ceil_mem_mhz
            pr = {}
            for t in tiers.TIERS:
                pr[t], times = tiers.predict(ls, gpu, p, t, sm, mem)
                sums[t] += pr[t] * 1e3
            lim = max(times, key=lambda x: x.seconds).limiter
            rows_detail.append(f"| {r['variant']} | {float(r['time_ms']):.4f} | {pr['T2'] * 1e3:.4f} "
                               f"| {pr['T2'] * 1e3 / float(r['time_ms']) - 1:+.0%} | {lim} |")
        total = float(vs["total"]["time_ms"]) if "total" in vs else float("nan")
        meas_steps = sum(float(r["time_ms"]) for r in steps)
        eager = float(vs["torch"]["time_ms"]) if "torch" in vs else float("nan")
        graph = float(vs["torch_graph"]["time_ms"]) if "torch_graph" in vs else float("nan")
        graph_cell = f"{graph:.3f}" if graph == graph else "— (out of memory)"
        lines.append(f"| {dims} | {total:.3f} | {meas_steps:.3f} | {sums['T0']:.3f} | {sums['T1']:.3f} "
                     f"| {sums['T2']:.3f} | {sums['T2'] / total - 1:+.0%} | {eager:.3f} | "
                     f"{graph_cell} | {eager / total:.2f}× |")
        detail.append((dims, rows_detail))
    dims, rows_detail = detail[-1]
    lines += ["", f"Per step at {dims}:", "", "| step | measured | model T2 | error | limiter |",
              "|---|---|---|---|---|"] + rows_detail
    return "\n".join(lines)


# Categorical slots 1-5 of the reference palette (validated light-mode: CVD and normal-vision
# separation pass; three slots sit below 3:1 on the surface, so every kernel also gets its own
# marker shape and a legend entry, and results/model/<gpu>_predictions.csv is the table view).
KERNEL_STYLE = {
    "gemm": ("#2a78d6", "o"), "attention": ("#eb6834", "s"), "softmax": ("#1baf7a", "^"),
    "layernorm": ("#eda100", "D"), "block": ("#e87ba4", "v"),
}
SURFACE, INK, INK_2, INK_3, GRIDLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"


def scatter(gpu: str, rows, preds) -> Path:
    """analysis/plots/model_<gpu>.png: predicted vs measured per tier, one panel each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE, sharey=True)
    meas = [r.time_ms for r in rows]
    lo, hi = min(meas) / 3, max(meas) * 3
    for ax, tier in zip(axes, tiers.TIERS):
        ax.set_facecolor(SURFACE)
        ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1.25, zorder=1)
        ax.fill_between([lo, hi], [lo / 1.25, hi / 1.25], [lo * 1.25, hi * 1.25], color=GRIDLINE,
                        lw=0, zorder=0)
        for kernel, (color, marker) in KERNEL_STYLE.items():
            pts = [(r.time_ms, pr[tier] * 1e3) for r, pr in zip(rows, preds) if r.kernel == kernel]
            if pts:
                ax.scatter(*zip(*pts), s=36, marker=marker, color=color, edgecolors=SURFACE,
                           linewidths=1.5, zorder=3, label=kernel)
        med = statistics.median(abs(ratio(pr, r, tier) - 1) for r, pr in zip(rows, preds))
        ax.set_title(f"{tier}: median |error| {med:.0%}", loc="left", fontsize=10, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, which="major", color=GRIDLINE, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_3)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_xlabel("Measured time (ms)", fontsize=9, color=INK_2)
    axes[0].set_ylabel("Predicted time (ms)", fontsize=9, color=INK_2)
    axes[-1].legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK_2)
    tuned = "tuned on these rows" if gpu == "rtx4070_laptop" else "held out: no refitting"
    fig.suptitle(f"Model vs measured, {gpu} ({tuned}), {len(rows)} rows", fontsize=12, color=INK,
                 x=0.01, ha="left")
    fig.text(0.01, 0.01, "line = perfect prediction · band = within ±25% · one point per result "
             "row, predicted at the clock it ran at", fontsize=7.5, color=INK_3)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = ROOT / "analysis" / "plots" / f"model_{gpu}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
