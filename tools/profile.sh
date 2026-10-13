#!/usr/bin/env bash
# tools/profile.sh <kernel> <variant> <dims...>     e.g. tools/profile.sh gemm v0 1024 1024 1024
#
# 1. Times the kernel with build/bench (median of 21, boost clocks).
# 2. Profiles one complete call with Nsight Compute at locked base clocks, so counters are
#    comparable between the two laptops (ncu 2026.x defaults to --clock-control boost, so pass
#    base). "One call" is every kernel inside the bench's `kl_profile` NVTX range, taken after
#    warmup: one kernel for GEMM/softmax/layernorm, 2·heads + 2 or 3 for attention. Metrics in
#    metrics.csv combine all of them (bytes and time summed, rates time-weighted).
# 3. Saves results/ncu/<gpu>/<kernel>_<variant>_<dims>.ncu-repz, upserts the bench row (with
#    dram_bytes and achieved_occ from ncu) into results/<gpu>/<kernel>.csv, and the full metric
#    row into results/ncu/<gpu>/metrics.csv.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '2p' "$0" | sed 's/^# //' >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BENCH=build/bench
[ -x "$BENCH" ] || { echo "build/bench missing; run cmake --build build" >&2; exit 1; }
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
NCU=$(command -v ncu || echo /opt/nsight-compute/ncu)

kernel=$1 variant=$2
shift 2
dims=$(IFS=x; echo "$*")

# variant "torch" is the PyTorch baseline (D9); it marks one eager call with the same NVTX range.
cmd=("$BENCH" "$kernel" "$variant" "$@")
[ "$variant" = torch ] && cmd=("$PY" bench/torch_baseline.py "$kernel" "$@")

row=$("${cmd[@]}")
gpu=${row%%,*}
name="${kernel}_${variant}_${dims}"
mkdir -p "results/ncu/$gpu"

METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,regex:smsp__average_warps_issue_stalled_.*_per_issue_active"  # ncu rejects ^/$ anchors

"$NCU" --clock-control base --nvtx --nvtx-include "kl_profile/" \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
    --section LaunchStats --section WarpStateStats --section SchedulerStats \
    --metrics "$METRICS" -f -o "results/ncu/$gpu/$name" \
    "${cmd[@]}" >/dev/null

"$PY" tools/ncu_metrics.py "results/ncu/$gpu/$name.ncu-repz" \
    --kernel "$kernel" --variant "$variant" --dims "$dims" --bench-row "$row" \
    --results "results/$gpu/$kernel.csv" --metrics-csv "results/ncu/$gpu/metrics.csv"

grep -h ",$kernel,$variant,$dims," "results/$gpu/$kernel.csv"
