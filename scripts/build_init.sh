#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/v0_uniform.yaml}"
SCENE_ROOT="${2:-data/scene_x}"

python -m init.build_init \
  --config "${CONFIG}" \
  --scene-root "${SCENE_ROOT}"
