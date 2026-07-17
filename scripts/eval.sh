#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/gsplat_train.yaml}"
SCENE_ROOT="${2:-data/scene_x}"

uv run python -m gsplat_train.eval \
  --config "${CONFIG}" \
  --scene-root "${SCENE_ROOT}"
