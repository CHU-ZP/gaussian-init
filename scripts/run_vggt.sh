#!/usr/bin/env bash
set -euo pipefail

SCENE_ROOT="${1:-data/scene_x}"

python -m preprocess.run_vggt \
  --images "${SCENE_ROOT}/images" \
  --output "${SCENE_ROOT}/vggt/predictions.npz" \
  "${@:2}"
