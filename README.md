# VGGT PCA gsplat Initialization

This repository separates the pipeline into three responsibilities:

1. VGGT provides dense geometry.
2. The PCA initialization modules generate Gaussian parameters.
3. gsplat handles optimization and rendering.

The first usable path implemented here is Stage 2 from the design notes:
load VGGT-style dense predictions, sample pixels, estimate local 3D PCA
covariances, optionally fuse proposals, and save an initialization file that a
gsplat trainer can consume.

## Repository Layout

```text
data/
preprocess/
init/
gsplat_train/
configs/
scripts/
tests/
```

Expected VGGT prediction format:

```text
depth:        [V, H, W]          optional when world_points exists
confidence:  [V, H, W]
world_points:[V, H, W, 3]
intrinsics:  [V, 3, 3]
extrinsics:  [V, 4, 4]
```

## Quick Start

Create the uv-managed environment. This project standardizes on Python 3.11
and PyTorch 2.3.1 with CUDA 12.1 wheels because VGGT pins the matching Torch
stack.

```bash
mkdir -p external
test -d external/vggt || git clone https://github.com/facebookresearch/vggt.git external/vggt
uv sync --extra dev
uv run python scripts/verify_env.py
```

Create a placeholder VGGT prediction for wiring checks:

```bash
uv run python -m preprocess.run_vggt \
  --images data/scene_x/images \
  --output data/scene_x/vggt/predictions.npz \
  --mock-plane
```

Run real VGGT on a scene:

```bash
uv run python -m preprocess.run_vggt \
  --images data/scene_x/images \
  --output data/scene_x/vggt/predictions.npz \
  --device cuda \
  --preprocess-mode crop
```

On small GPUs, lower the maximum side length:

```bash
uv run python -m preprocess.run_vggt \
  --images data/scene_x/images \
  --output data/scene_x/vggt/predictions.npz \
  --device cuda \
  --preprocess-mode crop \
  --max-resolution 336
```

Build Gaussian initialization:

```bash
uv run python -m init.build_init \
  --config configs/v0_uniform.yaml \
  --scene-root data/scene_x
```

The output is written to:

```text
data/scene_x/init/fused_gaussians.pt
```

The saved torch dictionary contains:

```text
means:       [N, 3]
scales:      [N, 3]
quats:       [N, 4]  # w, x, y, z
opacities:   [N]
colors:      [N, 3]
covariances: [N, 3, 3]
confidences: [N]
view_ids:    [N]
scores:      [N]
```

## Development Stages

- Stage 0: gsplat baseline from random or COLMAP initialization.
- Stage 1: VGGT dense geometry export and visualization.
- Stage 2: uniform pixel sampling plus local 3D PCA.
- Stage 3: salient and hybrid sampling.
- Stage 4: robust filtering before and after PCA.
- Stage 5: multi-view covariance fusion.
- Stage 6: full ablation-ready method.

## Notes

`preprocess.run_vggt` currently provides a mock geometry path for local smoke
checks. Replace that entry point with the actual VGGT call when the VGGT code
and weights are available.

`gsplat_train.train` is intentionally thin for now: it loads the initialization
file and verifies dependencies, but the concrete gsplat rasterization loop
should be filled in once the target camera and dataset format is finalized.

See `docs/environment.md` for the full environment policy.
