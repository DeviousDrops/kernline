"""python -m model <command>. Commands:

  predict    one kernel: python -m model predict gemm v3 4096x4096x4096 --gpu rtx4060_laptop
  validate   error table per tier and kernel against results/<gpu>/ (D2)
  tune       fit model/params.json on the 4070 Laptop's rows only (D1)
  block      transformer block: measured steps and total vs model vs PyTorch (D4)
  plots      predicted-vs-measured scatter per GPU and tier (analysis/plots/model_<gpu>.png)
  resources  regenerate model/resources.json from build/libkernels.so
  occupancy  check the occupancy calculator against every committed ncu report
"""
import argparse
import sys

from model import data, kernels, params, report, spec, tiers

TUNE_GPU = "rtx4070_laptop"


def cmd_predict(a) -> int:
    gpu, p = spec.load(a.gpu), params.load()
    ls = kernels.get(a.kernel).launches(kernels.parse_dims(a.dims), a.variant, gpu, p)
    sm = a.sm_mhz or gpu.ceil_sm_mhz
    mem = a.mem_mhz or gpu.ceil_mem_mhz
    print(f"{a.kernel} {a.variant} {a.dims} on {a.gpu} at SM {sm:.0f} / memory {mem:.0f} MHz")
    for tier in tiers.TIERS:
        total, times = tiers.predict(ls, gpu, p, tier, sm, mem)
        print(f"  {tier}  {total * 1e3:10.4f} ms")
        if tier == a.tier:
            for t in times[: a.max_launches]:
                parts = "  ".join(f"{k} {v * 1e3:.4f}" for k, v in t.parts.items())
                print(f"        {t.launch.kernel:28s} blocks {t.launch.blocks:7d} "
                      f"({t.blocks_per_sm}/SM, {t.waves:.2f} waves)  limiter {t.limiter:8s} {parts}")
            if len(times) > a.max_launches:
                print(f"        ... {len(times) - a.max_launches} more launches")
    return 0


def cmd_validate(a) -> int:
    rows = data.rows(a.gpu)
    if not rows:
        print(f"no modeled rows in results/{a.gpu}/")
        return 1
    preds = report.predict_rows(rows, spec.load(a.gpu), params.load())
    report.write_predictions(a.gpu, rows, preds)
    print(report.error_table(rows, preds, a.gpu, by_variant=not a.by_kernel))
    return 0


def cmd_tune(a) -> int:
    rows = data.rows(TUNE_GPU)
    fitted, err = report.tune(rows, spec.load(TUNE_GPU))
    params.save(fitted, f"{TUNE_GPU}, {len(rows)} rows, median |error| {err:.1%} at T2")
    print(f"tuned on {len(rows)} {TUNE_GPU} rows: {fitted}  (T2 median |error| {err:.1%})")
    return 0


def cmd_block(a) -> int:
    print(report.block_table(a.gpu, spec.load(a.gpu), params.load()))
    return 0


def cmd_plots(a) -> int:
    for g in a.gpus or spec.gpus():
        rows = data.rows(g)
        if rows:
            preds = report.predict_rows(rows, spec.load(g), params.load())
            print(report.scatter(g, rows, preds))
    return 0


def cmd_resources(a) -> int:
    from model import resources
    print(f"{len(resources.refresh())} kernels -> {resources.PATH}")
    return 0


def cmd_occupancy(a) -> int:
    from model import ncu_check
    return ncu_check.main()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m model", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="predict one kernel launch")
    p.add_argument("kernel", choices=kernels.KERNELS)
    p.add_argument("variant")
    p.add_argument("dims", help="x-joined, as in results/*.csv, e.g. 4096x4096x4096")
    p.add_argument("--gpu", default=TUNE_GPU)
    p.add_argument("--sm-mhz", type=float, help="default: the clock the ceilings were measured at")
    p.add_argument("--mem-mhz", type=float)
    p.add_argument("--tier", default="T2", choices=tiers.TIERS, help="tier to break down")
    p.add_argument("--max-launches", type=int, default=8)
    p = sub.add_parser("validate", help="error table against measured rows")
    p.add_argument("--gpu", default="rtx4060_laptop")
    p.add_argument("--by-kernel", action="store_true", help="one line per kernel, not per variant")
    sub.add_parser("tune", help=f"fit model/params.json on {TUNE_GPU} rows only")
    p = sub.add_parser("block", help="transformer block: steps and total")
    p.add_argument("--gpu", default=TUNE_GPU)
    p = sub.add_parser("plots", help="predicted vs measured scatter plots")
    p.add_argument("gpus", nargs="*")
    sub.add_parser("resources", help="regenerate model/resources.json")
    sub.add_parser("occupancy", help="check occupancy against ncu reports")
    a = ap.parse_args(argv)
    return globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    sys.exit(main())
