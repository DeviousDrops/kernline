#!/usr/bin/env python3
"""Turn tools/sim_ncu.sh's reports into the cache simulator's reference tables (docs/SIM.md).

    sim_ncu.py --metrics                  the counter list sim_ncu.sh passes to ncu
    sim_ncu.py results/sim/ncu/<gpu>      write the two tables for that GPU

results/sim/ncu_<gpu>.csv has one row per profiled launch, in launch order, with ncu's
per-operation L1/L2 sector counters. results/sim/launch_config_<gpu>.csv has resident blocks per
SM and the shared-memory carveout for each distinct (kernel, block size, shared memory), which
build/cachesim reads (sim/launch_config.h).
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ncu_metrics import _num, read_report  # noqa: E402

COUNTERS = {
    "ld_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1_hit_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "st_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "st_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l2_rd_sectors": "lts__t_sectors_srcunit_tex_op_read.sum",
    "l2_rd_hit_sectors": "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "l2_wr_sectors": "lts__t_sectors_srcunit_tex_op_write.sum",
    "l2_wr_hit_sectors": "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum",
    "dram_read_bytes": "dram__bytes_read.sum",
    "dram_write_bytes": "dram__bytes_write.sum",
}
LIMITS = ("blocks", "registers", "shared_mem", "warps")
NCU_FIELDS = (["gpu", "kernel", "variant", "dims", "launch", "kernel_name", "block_size",
               "grid_size", "smem_per_block", "smem_config_bytes", "resident_blocks"]
              + list(COUNTERS))
CONFIG_FIELDS = ["gpu", "kernel_name", "block_size", "smem_per_block", "resident_blocks",
                 "smem_config_bytes"]


def kernel_name(k: dict) -> str:
    """'void softmax_v2_kernel<8>(const float *, ...)' -> 'softmax_v2_kernel<8>'."""
    return k["Kernel Name"].split("(")[0].removeprefix("void ").strip()


def as_int(k: dict, metric: str) -> int:
    v = _num(k.get(metric))
    if v is None:
        raise KeyError(f"{metric} missing from the report; was sim_ncu.sh's metric list changed?")
    return int(v)


def rows_for(report: Path) -> list[dict]:
    kernel, variant, dims = report.stem.split("_", 2)
    out = []
    for n, k in enumerate(sorted(read_report(report), key=lambda k: int(k["ID"]))):
        row = dict(gpu=report.parent.name, kernel=kernel, variant=variant, dims=dims, launch=n,
                   kernel_name=kernel_name(k), block_size=as_int(k, "launch__block_size"),
                   grid_size=as_int(k, "launch__grid_size"),
                   smem_per_block=as_int(k, "launch__shared_mem_per_block"),
                   smem_config_bytes=as_int(k, "launch__shared_mem_config_size"),
                   resident_blocks=min(as_int(k, f"launch__occupancy_limit_{x}") for x in LIMITS))
        row.update({c: as_int(k, m) for c, m in COUNTERS.items()})
        out.append(row)
    return out


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    if sys.argv[1:] == ["--metrics"]:
        print(",".join(COUNTERS.values()))
        return 0
    rdir = Path(sys.argv[1])
    gpu = rdir.name
    rows = [r for rep in sorted(rdir.glob("*.ncu-repz")) for r in rows_for(rep)]
    write(ROOT / "results" / "sim" / f"ncu_{gpu}.csv", NCU_FIELDS, rows)
    config: dict[tuple, dict] = {}
    for r in rows:
        key = (r["kernel_name"], r["block_size"], r["smem_per_block"])
        seen = config.setdefault(key, r)
        if (seen["resident_blocks"], seen["smem_config_bytes"]) != (r["resident_blocks"], r["smem_config_bytes"]):
            raise ValueError(f"{key}: launches disagree on resident blocks or carveout")
    write(ROOT / "results" / "sim" / f"launch_config_{gpu}.csv", CONFIG_FIELDS,
          sorted(config.values(), key=lambda r: (r["kernel_name"], r["block_size"], r["smem_per_block"])))
    print(f"{len(rows)} launches, {len(config)} launch configs for {gpu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
