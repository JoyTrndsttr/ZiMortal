#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
unit="zimortal-online-rl-$(date +%Y%m%d-%H%M%S)"
python="${ZIMORTAL_PYTHON:-$PWD/.venv-cuda/bin/python}"
module="${ZIMORTAL_TRAINING_MODULE:-zimortal.training.online_rl}"
# 兼容旧启动命令，但训练指标只写本地，不逐代产生 Git 提交。
training_args=()
for arg in "$@"; do
  if [[ "$arg" != "--commit-reports" ]]; then
    training_args+=("$arg")
  fi
done
exec systemd-run --user --unit="$unit" \
  --property=MemoryMax=8G --property=MemorySwapMax=1G --property=CPUQuota=300% \
  --property=Nice=10 --property=RemainAfterExit=yes \
  --property="WorkingDirectory=$PWD" \
  --property="StandardOutput=append:$PWD/logs/$unit.log" \
  --property="StandardError=append:$PWD/logs/$unit.log" \
  --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /usr/bin/flock --nonblock "$PWD/logs/cashq-training.lock" \
  "$python" -m "$module" "${training_args[@]}"
