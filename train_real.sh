#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for scene in flower computer box luckycat; do
  python3 train_real.py --config "config/${scene}.yaml" --eval --gpu_id "${GPU_ID:-0}" "$@"
done
