#!/usr/bin/env bash
# tools/profile.sh <kernel> <variant> <dims...>     e.g. tools/profile.sh gemm v0 1024 1024 1024
#
# 1. Times the kernel with build/bench (median of 21, boost clocks).
# 2. Profiles one launch with Nsight Compute at locked base clocks, so counters are comparable
#    between the two laptops. ncu 2026.x defaults to --clock-control boost, so pass base.
# 3. Saves results/ncu/<gpu>/<kernel>_<variant>_<dims>.ncu-repz, upserts the bench row (with
#    dram_bytes and achieved_occ from ncu) into results/<gpu>/<kernel>.csv, and the full metric
#    row into results/ncu/<gpu>/metrics.csv.
#
# SKIP=<n> overrides how many launches to skip before profiling (default 3, past cold caches).
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

# variant "torch" is the PyTorch baseline (D9). It launches setup kernels (randn, ...) before the
# op, so ncu only counts launches whose name matches the op.
cmd=("$BENCH" "$kernel" "$variant" "$@")
filter=()
if [ "$variant" = torch ]; then
    cmd=("$PY" bench/torch_baseline.py "$kernel" "$@")
    case $kernel in
        softmax) filter=(--kernel-name "regex:(?i)softmax") ;;
        layernorm) filter=(--kernel-name "regex:(?i)layer_?norm") ;;
    esac
fi

row=$("${cmd[@]}")
gpu=${row%%,*}
name="${kernel}_${variant}_${dims}"
mkdir -p "results/ncu/$gpu"

METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,regex:smsp__average_warps_issue_stalled_.*_per_issue_active"  # ncu rejects ^/$ anchors

"$NCU" --clock-control base "${filter[@]}" --launch-skip "${SKIP:-3}" --launch-count 1 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
    --section LaunchStats --section WarpStateStats --section SchedulerStats \
    --metrics "$METRICS" -f -o "results/ncu/$gpu/$name" \
    "${cmd[@]}" >/dev/null

"$PY" tools/ncu_metrics.py "results/ncu/$gpu/$name.ncu-repz" \
    --kernel "$kernel" --variant "$variant" --dims "$dims" --bench-row "$row" \
    --results "results/$gpu/$kernel.csv" --metrics-csv "results/ncu/$gpu/metrics.csv"

grep -h ",$kernel,$variant,$dims," "results/$gpu/$kernel.csv"
