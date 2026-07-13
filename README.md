# VGGT LoG-Ellipse gsplat Initialization

This repository initializes anisotropic 3D Gaussians from VGGT dense geometry.
The implemented path is:

1. VGGT predicts dense scene coordinates and confidence.
2. Scale-normalized Laplacian-of-Gaussian responses locate signed extrema in
   joint image/scale space.
3. A local structure tensor turns every keypoint into an oriented, area-bounded
   image-space ellipse.
4. All valid scene coordinates inside that ellipse contribute to a local 3D
   covariance, represented as the confidence-weighted second moment around the
   keypoint's fixed 3D center.
5. Covariance eigendecomposition initializes Gaussian scale and rotation; the
   keypoint scene coordinate initializes its center and keypoint RGB initializes
   the degree-zero spherical-harmonic coefficient.
6. Optional confidence-weighted voxel fusion combines overlapping proposals.

The ellipse covariance reduction uses chunked PyTorch tensor operations. It runs
on CUDA when available and uses the same implementation on CPU otherwise.

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
confidence:   [V, H, W]
world_points: [V, H, W, 3]
intrinsics:      [V, 3, 3]
extrinsics_c2w:  [V, 4, 4]
extrinsics_w2c:  [V, 4, 4]
processed_images: [V, H, W, 3]  optional but emitted by this repository
processed_valid_mask: [V, H, W] optional; excludes crop/pad batch padding
```

## Quick Start

The environment uses Python 3.11 and PyTorch 2.3.1 CUDA 12.1 wheels because
VGGT pins the matching Torch stack.

```bash
mkdir -p external
test -d external/vggt || git clone https://github.com/facebookresearch/vggt.git external/vggt
uv sync --extra dev
uv run python scripts/verify_env.py
```

### Prepare the real Tanks and Temples Truck scene

The repository includes a wrapper around the official Tanks and Temples Python
downloader. It downloads only `Truck.zip`, supplies the resource key published
on the official page, validates the response length, pinned archive MD5, and ZIP
CRC, safely extracts it, and uniformly selects 12 frames by default:

```bash
uv run python scripts/prepare_tnt_truck.py --accept-license
```

The dataset terms must be reviewed at
<https://www.tanksandtemples.org/license/> before passing `--accept-license`.
The official download page is <https://www.tanksandtemples.org/download/>.

Prepared files are placed under:

```text
data/tnt_truck/
├── images/
└── dataset_manifest.json
```

Use more uniformly spaced views or copy instead of symlinking with:

```bash
uv run python scripts/prepare_tnt_truck.py \
  --accept-license \
  --num-images 24 \
  --copy \
  --force
```

The script prints the exact VGGT and initialization commands for the prepared
scene when it finishes.

Create deterministic plane geometry for a wiring check:

```bash
uv run python -m preprocess.run_vggt \
  --images data/scene_x/images \
  --output data/scene_x/vggt/predictions.npz \
  --mock-plane
```

Run real VGGT:

```bash
uv run python -m preprocess.run_vggt \
  --images data/scene_x/images \
  --output data/scene_x/vggt/predictions.npz \
  --device cuda \
  --preprocess-mode crop \
  --head-frames-chunk-size 1
```

On smaller GPUs, reduce the maximum input side:

```bash
uv run python -m preprocess.run_vggt \
  --images data/scene_x/images \
  --output data/scene_x/vggt/predictions.npz \
  --device cuda \
  --preprocess-mode crop \
  --max-resolution 336 \
  --head-frames-chunk-size 1
```

Build the Gaussian initialization:

```bash
uv run python -m init.build_init \
  --config configs/log_ellipse.yaml \
  --scene-root data/scene_x
```

View the dense VGGT geometry, processed RGB images, and predicted cameras before
initialization:

```bash
uv run python scripts/view_vggt.py \
  --input data/scene_x/vggt/predictions.npz
```

The viewer applies the same content mask and default bottom-25% confidence
filter as initialization. It keeps every second pixel and caps the browser point
cloud at 500,000 points by default; use `--stride` and `--max-points` to change
those display-only limits.

Outputs are written to:

```text
data/scene_x/init/proposals.pt
data/scene_x/init/fused_gaussians.pt
data/scene_x/init/debug.ply
```

View the initialized Gaussian splats interactively in a web browser:

```bash
uv run python scripts/view_gaussians.py \
  --input data/scene_x/init/fused_gaussians.pt
```

Open <http://127.0.0.1:8080> and drag with the mouse to rotate the view. When
running on a remote server, forward the port from your local machine first:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@server
```

The saved Torch dictionary contains:

```text
means:       [N, 3]
scales:      [N, 3]
quats:       [N, 4]  # w, x, y, z
opacities:   [N]
sh_dc:       [N, 3]  # (RGB - 0.5) / 0.28209479177387814
covariances: [N, 3, 3]
confidences: [N]
view_ids:    [N]
scores:      [N]     # absolute scale-normalized LoG response
```

## Configuration

`sampling.sigmas` defines the LoD scale space. Extrema are detected jointly
across adjacent scales and 3x3 image neighborhoods. One geometric guard level
is generated internally on either side, so every configured sigma is usable.
`response_threshold` and
the cross-scale NMS parameters control keypoint density.

`sampling.confidence_percentile` removes the requested bottom percentage of
finite, content-valid VGGT confidence scores. VGGT confidence is an unbounded
ranking score with a lower bound of one, not a probability, so an absolute
threshold in `[0, 1]` is not meaningful.

The structure tensor determines ellipse orientation and anisotropy.
`min_ellipse_area`, `max_ellipse_area`, and `max_axis_ratio` bound its support.
There is no uniform, single-scale gradient, hybrid, or fixed square-patch path.

The `covariance` section controls valid-point coverage, confidence weighting,
3D distance rejection, compute device, and per-chunk pixel budget. Setting
`device: auto` selects CUDA when available.

The preprocessing command stores the exact crop/pad/resized RGB tensor and its
content mask as `processed_images` and `processed_valid_mask`. This keeps LoG
keypoints and SH colors pixel-aligned with VGGT scene coordinates and prevents
white padding seams from creating Gaussians. For external prediction files
without those fields, source images must already have exactly the prediction
resolution; arbitrary resizing is rejected because it would corrupt the
correspondence.

Only the VGGT aggregator uses CUDA mixed precision. Camera, depth, and point
heads run in float32, matching VGGT's official forward path. The
`--head-frames-chunk-size` option limits peak head activation memory and defaults
to one frame. Prediction files created by the previous mixed-precision-head
implementation must be regenerated; initialization rejects those stale files.

## Current Boundary

The LoG/ellipse initialization, 3D covariance estimation, Gaussian parameter
construction, SH DC conversion, and voxel fusion are implemented and tested.

`gsplat_train.train` and `gsplat_train.eval` still only load the resulting
state. The concrete rasterization, optimization, checkpoint, and rendering
metric loops remain to be integrated once the target camera/dataset contract is
fixed.

See `docs/environment.md` for the environment policy.
