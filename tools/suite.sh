#!/usr/bin/env bash
# tools/suite.sh [kernel ...]: the full bench + ncu suite (D8) on this GPU, every kernel and
# variant from both parts at the shapes the model is tuned (4070) and validated (4060) on.
# Default: all of gemm softmax layernorm attention block. Run on an idle, plugged-in laptop.
#
# Every case goes through tools/profile.sh (bench row + ncu report + metrics row), except the
# transformer block, whose steps are separate launches already covered by the kernel rows:
# build/block and bench/torch_block.py rows are upserted into results/<gpu>/block.csv as is.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
kernels=("$@")
[ ${#kernels[@]} -eq 0 ] && kernels=(gemm softmax layernorm attention block)

run() { echo "== $*" >&2; tools/profile.sh "$@" | tail -1; }

for k in "${kernels[@]}"; do
    case $k in
    gemm)
        # Squares, then the four GPT-2 small projections at T = 512 and 4096 tokens
        # (QKV, out-proj, FC1, FC2), then two squares whose tile counts fall between waves.
        shapes=("1024 1024 1024" "2048 2048 2048" "4096 4096 4096"
                "512 2304 768" "512 768 768" "512 3072 768" "512 768 3072"
                "4096 2304 768" "4096 768 768" "4096 3072 768" "4096 768 3072"
                "1536 1536 1536" "3072 3072 3072")
        for s in "${shapes[@]}"; do
            for v in v0 v1 v2 v3 v4 cublas; do run gemm $v $s; done
        done ;;
    softmax)
        for s in "6144 512" "12288 1024" "24576 2048" "4096 16384"; do
            for v in v0 v1 v2 v3 torch; do run softmax $v $s; done
        done ;;
    layernorm)
        for s in "1024 768" "8192 768" "4096 16384"; do
            for v in v0 v1 v2 v3 torch; do run layernorm $v $s; done
        done ;;
    attention)
        for n in 256 512 1024 2048 4096; do
            for v in v0 v1 v2 torch; do run attention $v 12 $n 64; done
        done ;;
    block)
        [ -x build/block ] || { echo "build/block missing; run cmake --build build" >&2; exit 1; }
        for s in "1 512" "1 2048" "4 1024" "8 512" "8 2048"; do
            echo "== block $s" >&2
            rows=$(build/block $s; "$PY" bench/torch_block.py $s)
            gpu=${rows%%,*}
            "$PY" -c '
import csv, sys
from pathlib import Path
sys.path.insert(0, "tools")
from ncu_metrics import RESULTS_HEADER, upsert
for row in csv.reader(sys.argv[2].splitlines()):
    upsert(Path(sys.argv[1]), RESULTS_HEADER, dict(zip(RESULTS_HEADER, row)),
           ("gpu", "kernel", "variant", "dims"))' "results/$gpu/block.csv" "$rows"
            echo "$rows" | grep ',total,\|,torch'
        done ;;
    *) echo "unknown kernel $k" >&2; exit 2 ;;
    esac
done
