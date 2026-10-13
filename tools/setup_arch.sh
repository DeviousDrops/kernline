#!/usr/bin/env bash
# One-time Arch WSL setup for KernLine. Run as root (the WSL default) or with sudo.
# Windows side first: latest NVIDIA driver (>= 580) + GPU perf counters enabled for all users.
set -euo pipefail

if [ ! -d /etc/pacman.d/gnupg ]; then
    pacman-key --init
    pacman-key --populate archlinux
fi
pacman -Syu --noconfirm --needed base-devel git cmake ninja cuda nsight-compute python uv

# Silence perl locale warnings on a fresh install.
if ! locale -a | grep -qi '^en_US.utf8$'; then
    sed -i 's/^#en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
    locale-gen
fi

# Drop any stale CUDA PATH lines, then add a clean one.
sed -i '/opt\/cuda/d' ~/.bashrc
cat >> ~/.bashrc <<'EOF'
export PATH=/opt/cuda/bin:/opt/nsight-compute:$PATH
EOF

export PATH=/opt/cuda/bin:/opt/nsight-compute:$PATH
nvcc --version | tail -1
ncu --version | tail -1
