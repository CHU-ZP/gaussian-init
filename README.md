# Dense-Geometry Gaussian Initialization

**[Project website](https://chu-zp.github.io/gaussian-init/)** ·
**[Technical report](report/technical_report_en.pdf)** ·
**[Environment guide](docs/environment.md)**

This repository initializes anisotropic 3D Gaussians from pixel-aligned dense
geometry. It constructs structure-aware and regular-grid regions in each view,
extracts continuous local 3D surfaces, fits Gaussian parameters with PCA, and
fuses duplicate representations across views.

```text
multi-view RGB + dense geometry
  -> multiscale Lab-LoG ellipses + uncovered-grid regions
  -> 3D continuity filtering
  -> region-level PCA Gaussians
  -> multi-view similarity-graph fusion
  -> fused_gaussians.pt
```

VGGT is used as the dense-geometry estimator in the reported experiment, but
the initializer only requires aligned RGB and per-pixel 3D coordinates. Gaussian
optimization and rendering use gsplat.

## Setup

The environment uses Python 3.10, PyTorch 2.3.1 with CUDA 12.1, gsplat 1.5.3,
and the CUDA 12 build of PyCOLMAP.

```bash
mkdir -p external
git clone https://github.com/facebookresearch/vggt.git external/vggt
uv sync --extra dev --extra train --extra colmap
uv run python scripts/verify_env.py
```

See [docs/environment.md](docs/environment.md) for server setup and CUDA notes.

## Quick start

Prepare 48 uniformly spaced Tanks and Temples Truck views:

```bash
uv run python scripts/prepare_tnt_truck.py \
  --accept-license \
  --output data/tnt_truck_48 \
  --num-images 48
```

Estimate dense geometry with VGGT:

```bash
uv run python -m preprocess.run_vggt \
  --images data/tnt_truck_48/images \
  --output data/tnt_truck_48/vggt/predictions.npz \
  --device cuda \
  --preprocess-mode crop \
  --max-resolution 518 \
  --head-frames-chunk-size 1
```

Build and inspect the proposed initialization:

```bash
uv run python -m init.build_init \
  --config configs/log_ellipse.yaml \
  --scene-root data/tnt_truck_48

uv run python scripts/view_gaussians.py \
  --input data/tnt_truck_48/init/fused_gaussians.pt
```

The final initialization is written to:

```text
data/tnt_truck_48/init/fused_gaussians.pt
```

## COLMAP comparison

Build the aligned COLMAP sparse initialization, then train both pipelines and
generate convergence reports and progress animations:

```bash
uv run --extra colmap python scripts/run_colmap_pipeline.py \
  --scene-root data/tnt_truck_48 \
  --skip-training

uv run python scripts/run_dense_gif_comparison.py \
  --scene-root data/tnt_truck_48
```

Both pipelines use the same gsplat optimization and densification settings from
[configs/gsplat_compare_48.yaml](configs/gsplat_compare_48.yaml).

## Core implementation

```text
preprocess/run_vggt.py       VGGT inference and aligned geometry export
init/sampling.py             multichannel multiscale LoG proposals
init/grid_supplement.py      uncovered-grid region completion
init/continuity.py           fixed-scale 3D continuity
init/regions.py              region extraction
init/pca.py                  anisotropic Gaussian fitting
init/fusion.py               multi-view similarity-graph fusion
gsplat_train/                optimization, evaluation, and rendering
```

Production parameters are in [configs/log_ellipse.yaml](configs/log_ellipse.yaml).
Method details and references are provided in the
[technical report](report/technical_report_en.pdf).

## Validation

```bash
uv run ruff check .
uv run pytest -q
```

The project uses [VGGT](https://github.com/facebookresearch/vggt),
[gsplat](https://github.com/nerfstudio-project/gsplat),
[COLMAP](https://colmap.github.io/), and the
[Tanks and Temples](https://www.tanksandtemples.org/) dataset. Refer to their
respective licenses for permitted use.
