#!/usr/bin/env bash
# tools/sim_ncu.sh: ncu reference counters for the cache simulator (docs/SIM.md, "Comparison
# with ncu"). Profiles every case in sim/cases.txt through the bench's kl_profile range at base
# clocks, with ncu's default cache control (caches flushed before each kernel, which is what the
# simulator's cold start models), then writes results/sim/ncu_<gpu>.csv and
# results/sim/launch_config_<gpu>.csv. Run on an idle, plugged-in laptop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -x build/bench ] || { echo "build/bench missing; run cmake --build build" >&2; exit 1; }
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
NCU=$(command -v ncu || echo /opt/nsight-compute/ncu)

gpu=$(build/bench gemm v3 128 128 128 | cut -d, -f1)
out="results/sim/ncu/$gpu"
mkdir -p "$out"
metrics=$("$PY" tools/sim_ncu.py --metrics)

grep -v '^#' sim/cases.txt | while read -r kernel variant dims; do
    [ -n "$kernel" ] || continue
    echo "== $kernel $variant $dims" >&2
    # shellcheck disable=SC2086  # dims split into the bench's separate arguments on purpose
    "$NCU" --clock-control base --cache-control all --nvtx --nvtx-include "kl_profile/" \
        --section LaunchStats --section Occupancy --metrics "$metrics" \
        -f -o "$out/${kernel}_${variant}_${dims}" \
        build/bench "$kernel" "$variant" ${dims//x/ } >/dev/null </dev/null
done
"$PY" tools/sim_ncu.py "$out"
