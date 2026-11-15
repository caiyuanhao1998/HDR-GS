#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for scene in sponza sofa bear chair desk diningroom dog bathroom; do
  python3 train_synthetic.py --config "config/${scene}.yaml" --eval --gpu_id "${GPU_ID:-0}" --syn "$@"
done
