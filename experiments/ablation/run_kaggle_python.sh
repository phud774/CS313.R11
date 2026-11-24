#!/usr/bin/env bash
set -euo pipefail

workspace=${1:?Pass the Kaggle working directory}
shift
venv="$workspace/.dgmae-venv"
python="$venv/bin/python"
site_packages=$("$python" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')

# DGL's conda extensions need CUDA libraries installed by the PyTorch wheels.
if [[ ! -r "$site_packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12" ]]; then
    printf 'CUDA NVRTC library is missing from %s\n' "$site_packages/nvidia/cuda_nvrtc/lib" >&2
    exit 1
fi
library_dirs=()
for directory in "$site_packages"/nvidia/*/lib; do
    if [[ -d "$directory" ]]; then
        library_dirs+=("$directory")
    fi
done
library_dirs+=("$venv/lib")
wheel_library_path=$(IFS=:; printf '%s' "${library_dirs[*]}")
export LD_LIBRARY_PATH="$wheel_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$python" "$@"
