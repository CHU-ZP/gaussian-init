# Dense-Geometry Gaussian Initialization

This repository builds anisotropic 3D Gaussian initializations from
pixel-aligned dense geometry and compares the resulting VGGT pipeline with an
aligned COLMAP sparse pipeline under the same gsplat optimizer.

The production initializer is identified as:

```text
dense_lab_log_ellipse_grid_region_pca
```

## Production pipeline

```text
RGB + per-pixel world coordinates
├─ normalized CIELAB multiscale LoG
│  └─ strict x/y/scale extrema
│     └─ multichannel structure-tensor ellipses
└─ ellipse coverage
   └─ uncovered regular-grid supplements
      └─ near-convex color/3D-continuous regions

ellipse and grid regions
└─ fixed per-view 3D edge-length connectivity
   └─ covariance/PCA Gaussian fitting
      └─ overlap/normal/scale/color similarity-graph fusion
         └─ fused_gaussians.pt
```

Important properties:

- VGGT is the included dense-geometry backend, but initialization only requires
  aligned RGB and world coordinates (or depth, intrinsics, and extrinsics).
- Confidence is optional metadata and is never used for filtering or weighting.
- LoG responses are computed independently on normalized Lab channels, divided
  by one robust `MAD + epsilon` scale per channel, and fused by vector magnitude.
- Structure tensors use all Lab gradients and determine only ellipse shape;
  they are not Harris keypoint detectors.
- Every 3D connectivity decision uses one fixed reference per view: the median
  valid neighboring 3D step per image pixel. Local depth discontinuities cannot
  increase their own threshold.
- Grid supplementation adds low-texture coverage without subtracting irregular
  ellipse masks from selected cells.
- Saved initialization scales and opacities are activated values. The gsplat
  adapter converts them to log-scales and logits only when training starts.

The production parameters live in
[`configs/log_ellipse.yaml`](configs/log_ellipse.yaml).

## Environment

The uv environment pins Python 3.10, PyTorch 2.3.1 with CUDA 12.1 wheels,
gsplat 1.5.3, VGGT, Viser, and the CUDA 12 PyCOLMAP package.

```bash
mkdir -p external
test -d external/vggt || git clone https://github.com/facebookresearch/vggt.git external/vggt
uv sync --extra dev --extra train --extra colmap
uv run python scripts/verify_env.py
```

Do not install the external VGGT requirements separately. See
[`docs/environment.md`](docs/environment.md) for server and CUDA details.

## Prepare Tanks and Temples Truck

Review the dataset license, then prepare uniformly spaced frames with the
official downloader wrapper:

```bash
uv run python scripts/prepare_tnt_truck.py \
  --accept-license \
  --output data/tnt_truck_48 \
  --num-images 48
```

The original download cache is kept under `data/downloads/tanks_and_temples/`.

## Run VGGT

```bash
uv run python -m preprocess.run_vggt \
  --images data/tnt_truck_48/images \
  --output data/tnt_truck_48/vggt/predictions.npz \
  --device cuda \
  --preprocess-mode crop \
  --max-resolution 518 \
  --head-frames-chunk-size 1
```

The repository runner keeps the aggregator in CUDA mixed precision and runs the
camera, depth, and point heads in float32. It stores processed RGB and a content
mask beside the geometry so every later pixel lookup remains aligned.

Inspect geometry and cameras:

```bash
uv run python scripts/view_vggt.py \
  --input data/tnt_truck_48/vggt/predictions.npz
```

## Build the production initialization

```bash
uv run python -m init.build_init \
  --config configs/log_ellipse.yaml \
  --scene-root data/tnt_truck_48
```

Outputs:

```text
data/tnt_truck_48/init/proposals.pt
data/tnt_truck_48/init/fused_gaussians.pt
data/tnt_truck_48/init/debug.ply
```

`proposals.pt` is the fitted per-view proposal set before similarity-graph
fusion. `fused_gaussians.pt` is the training-ready final initialization.

The Torch payload contains:

```text
means:       [N, 3]
scales:      [N, 3]       activated principal-axis scales
quats:       [N, 4]       wxyz
opacities:   [N]          activated opacity, initialized to 0.2
sh_dc:       [N, 3]
covariances: [N, 3, 3]
view_ids:    [N]
scores:      [N]
metadata:    method, source paths, filtering and fusion statistics
```

## Inspect sampling and Gaussians

Multiscale blur, fused Lab-LoG response, candidates, and ellipses:

```bash
uv run python scripts/view_log_sampling.py \
  --config configs/log_ellipse.yaml \
  --scene-root data/tnt_truck_48 \
  --views 0,12,24,36 \
  --output data/tnt_truck_48/init/log_sampling
```

Candidates before and after same-scale ellipse merging:

```bash
uv run python scripts/view_ellipse_merging.py \
  --config configs/log_ellipse.yaml \
  --scene-root data/tnt_truck_48 \
  --views 0,12,24,36 \
  --output data/tnt_truck_48/init/ellipse_merging
```

Final 2D ellipses:

```bash
uv run python scripts/view_ellipses.py \
  --config configs/log_ellipse.yaml \
  --scene-root data/tnt_truck_48 \
  --output data/tnt_truck_48/init/ellipse_debug
```

Interactive 3D Gaussian viewer:

```bash
uv run python scripts/view_gaussians.py \
  --input data/tnt_truck_48/init/fused_gaussians.pt
```

On a remote server, forward the Viser port:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@server
```

## Aligned COLMAP pipeline

The comparison uses the same processed 48 RGB frames but lets COLMAP estimate
its own PINHOLE cameras and sparse points. It uses CUDA SIFT, guided exhaustive
matching, incremental SfM, requires every view to register, and does not
undistort images. A shared translation and uniform scale align the COLMAP camera
gauge to VGGT without changing COLMAP projections.

Run reconstruction and initialization only:

```bash
uv run --extra colmap python scripts/run_colmap_pipeline.py \
  --scene-root data/tnt_truck_48 \
  --skip-training
```

Important artifacts:

```text
data/tnt_truck_48/colmap/sparse/0/
data/tnt_truck_48/colmap/reconstruction.json
data/tnt_truck_48/colmap/scene.npz
data/tnt_truck_48/init/colmap_sparse_gaussians.pt
```

Each COLMAP sparse point becomes one isotropic Gaussian. Its scale is the RMS
distance to the exact three nearest sparse points, its opacity is 0.2, and its
rotation is the identity quaternion.

## Retrain both pipelines and create dense GIFs

[`scripts/run_dense_gif_comparison.py`](scripts/run_dense_gif_comparison.py)
trains the VGGT strategy-sampling and COLMAP sparse initializations sequentially
with the same optimizer and densification settings.

```bash
uv run python scripts/run_dense_gif_comparison.py \
  --scene-root data/tnt_truck_48
```

The default preview schedule stores real renders for views 12 and 36:

```text
step 0 through 100: every step
step 200 through 30000: every 100 steps
```

Full 48-view metrics remain at 500-step intervals. Periodic checkpoints are
disabled; a normal run writes only its final checkpoint, while Ctrl+C writes one
recovery checkpoint.

Outputs:

```text
data/tnt_truck_48/comparisons/dense_gif/
├── vggt_log_grid/
├── colmap_sparse/
├── dense_training_config.yaml
├── dense_gif_manifest.json
└── report/
    ├── summary.csv
    ├── summary.json
    ├── convergence.svg
    ├── render_comparison_view_012.png
    ├── render_comparison_view_036.png
    ├── training_progress_view_012.gif
    └── training_progress_view_036.gif
```

Use `--restart` to replace this script's existing output root, or select a new
one with `--output-root`.

The lower-level COLMAP runner remains available when only that pipeline needs
to be rebuilt or trained:

```bash
uv run --extra colmap python scripts/run_colmap_pipeline.py \
  --scene-root data/tnt_truck_48
```

The shared training settings are in
[`configs/gsplat_compare_48.yaml`](configs/gsplat_compare_48.yaml). This is an
all-view reconstruction/convergence benchmark, not a held-out novel-view test.

## Validate the repository

```bash
uv run ruff check .
uv run pytest -q
```
