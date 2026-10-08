#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
unit="zimortal-active-$(date +%Y%m%d-%H%M%S)"
# Generation uses CPU torch to avoid loading CUDA libraries in every worker.
python="${ZIMORTAL_PYTHON:-$PWD/.venv/bin/python}"
exec systemd-run --user --unit="$unit" \
  --property=MemoryMax=8G --property=MemorySwapMax=1G --property=CPUQuota=600% \
  --property=Nice=10 --property=RemainAfterExit=yes \
  --property="WorkingDirectory=$PWD" \
  --property="StandardOutput=append:$PWD/logs/$unit.log" \
  --property="StandardError=append:$PWD/logs/$unit.log" \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  /usr/bin/flock --nonblock "$PWD/logs/cashq-training.lock" \
  "$python" -m zimortal.training.active "$@"
