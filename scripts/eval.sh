#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/log_ellipse.yaml}"
SCENE_ROOT="${2:-data/scene_x}"

python -m gsplat_train.eval \
  --config "${CONFIG}" \
  --scene-root "${SCENE_ROOT}"
