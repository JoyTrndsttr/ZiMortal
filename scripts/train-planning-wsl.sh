#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
unit="zimortal-planning-$(date +%Y%m%d-%H%M%S)"
python="${ZIMORTAL_PYTHON:-$PWD/.venv-cuda/bin/python}"
exec systemd-run --user --unit="$unit" \
  --property=MemoryMax=8G --property=MemorySwapMax=1G \
  --property=MemoryAccounting=yes --property=CPUAccounting=yes --property=RemainAfterExit=yes \
  --property=CPUQuota=600% --property=Nice=10 \
  --property="WorkingDirectory=$PWD" \
  --property="StandardOutput=append:$PWD/logs/$unit.log" \
  --property="StandardError=append:$PWD/logs/$unit.log" \
  --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 \
  /usr/bin/flock --nonblock "$PWD/logs/planning-training.lock" \
  "$python" -m zimortal.training.planning train \
  --data data/generated/planning-pilot-v2 --device cuda \
  --output checkpoints/planning-cuda-resnet.pt \
  --report docs/training/planning-cuda-pilot.json "$@"
