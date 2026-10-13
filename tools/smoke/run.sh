#!/usr/bin/env bash
# Day-1 go/no-go: build vadd, run it, then profile it with Nsight Compute.
set -euo pipefail
cd "$(dirname "$0")"

nvcc -O3 -arch=sm_89 -lineinfo -o vadd vadd.cu
./vadd
ncu --metrics gpu__time_duration.sum,dram__bytes_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed ./vadd
