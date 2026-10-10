#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
unit="zimortal-online-rl-$(date +%Y%m%d-%H%M%S)"
python="${ZIMORTAL_PYTHON:-$PWD/.venv-cuda/bin/python}"
exec systemd-run --user --unit="$unit" \
  --property=MemoryMax=8G --property=MemorySwapMax=1G --property=CPUQuota=300% \
  --property=Nice=10 --property=RemainAfterExit=yes \
  --property="WorkingDirectory=$PWD" \
  --property="StandardOutput=append:$PWD/logs/$unit.log" \
  --property="StandardError=append:$PWD/logs/$unit.log" \
  --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /usr/bin/flock --nonblock "$PWD/logs/cashq-training.lock" \
  "$python" -m zimortal.training.online_rl "$@"
