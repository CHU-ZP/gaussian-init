# VGGT Multichannel Lab LoG-Ellipse Initialization

This repository initializes anisotropic 3D Gaussians from VGGT dense geometry.
The implemented path is:

1. VGGT predicts dense scene coordinates and confidence.
2. RGB is converted to normalized CIELAB. Scale-normalized
   Laplacian-of-Gaussian responses are computed for L, a, and b, divided by each
   channel's `MAD + epsilon`, and fused as a weighted vector magnitude.
3. Strict magnitude maxima are located jointly in image/scale space, and a
   multichannel structure tensor turns every keypoint into an oriented,
   area-bounded image-space ellipse.
4. Similar same-scale ellipses are merged by spatial overlap, orientation,
   color, and a fixed per-view 3D path-continuity limit.
5. Inside each ellipse, the center-connected pixel component is extracted with
   the same fixed per-view 3D edge limit. Its scene coordinates produce a
   confidence-weighted second moment around the keypoint's fixed 3D center.
6. Covariance eigendecomposition initializes Gaussian scale and rotation; the
   keypoint scene coordinate initializes its center and keypoint RGB initializes
   the degree-zero spherical-harmonic coefficient.
7. Optional confidence-weighted similarity-graph fusion uses voxels only for
   candidate lookup, then combines proposals that also agree in 3D overlap,
   normal direction, axis scales, and Lab color.

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

The environment uses Python 3.10 and PyTorch 2.3.1 CUDA 12.1 wheels so VGGT and
the official precompiled gsplat wheel share one pinned stack.

```bash
mkdir -p external
test -d external/vggt || git clone https://github.com/facebookresearch/vggt.git external/vggt
uv sync --extra dev --extra train
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

Inspect the detected 2D support ellipses before building 3D Gaussians:

```bash
uv run python scripts/view_ellipses.py \
  --config configs/log_ellipse.yaml \
  --scene-root data/scene_x \
  --output data/scene_x/init/ellipse_debug
```

The command writes one RGB overlay per view, `ellipses.csv`, and `summary.json`.
Ellipses with axis ratio below 2 are green, ratios from 2 to 4 are yellow, and
ratios at or above 4 are red. Use `--views 0,3,6-10` to inspect selected views,
or `--max-ellipses-per-view 100` to draw only the most elongated ellipses.

Inspect how LoG sampling changes at every Gaussian blur strength:

```bash
uv run python scripts/view_log_sampling.py \
  --config configs/log_ellipse.yaml \
  --scene-root data/scene_x \
  --views 0,8,16 \
  --output data/scene_x/init/log_sampling
```

For each view, the command writes one contact sheet with one row per configured
`sigma`, plus a full-resolution row image for every scale. Its four columns show
the three blurred Lab channels reconstructed to RGB, the dominant normalized
LoG channel, the fused Lab LoG magnitude, and final ellipse supports on the
original RGB. Yellow, magenta, and cyan identify L-, a-, and b-dominant
responses respectively; white rings/dots are selected keypoints. The fused
response panel shows thresholded `3x3x3` candidates before ellipse merging and
the final merged points. Green is the discrete 2D ellipse support before 3D
continuity.
Per-scale channel counts and coverage are saved in `scales.csv`. Purple regions
are excluded by the VGGT content/confidence mask. Use `--views all` for every
view.

To inspect every per-scale candidate before same-scale ellipse merging:

```bash
uv run python scripts/view_ellipse_merging.py \
  --config configs/log_ellipse.yaml \
  --scene-root data/scene_x \
  --views 0,8,16 \
  --output data/scene_x/init/ellipse_merging
```

These candidates have already passed the strict `3x3x3` image/scale extrema
test and the LoG response threshold, but have not entered same-scale ellipse
merging. Each four-column row compares the reconstructed Lab blur,
dominant-channel map, all raw ellipses on RGB, and merged supports. Ellipse color
identifies the dominant L/a/b response. Per-channel counts, coverage, overlap,
and output rate are written to `ellipse_merge_scales.csv`.

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
scores:      [N]     # fused normalized Lab LoG magnitude
```

## Configuration

`sampling.sigmas` defines the LoD scale space. The detector computes one
scale-normalized LoG response per normalized CIELAB channel.
Each channel uses one robust scale over all usable LoD levels,
`MAD(response) + response_mad_epsilon`; using one value per channel preserves
the relative response strength between sigma levels. The fused response is
`sqrt(L_hat^2 + chroma_weight * a_hat^2 + chroma_weight * b_hat^2)` and is
therefore nonnegative and has no bright/dark sign. Strict maxima are detected
jointly across adjacent scales and `3x3` image neighborhoods. One geometric
guard level is generated internally on either side, so every configured sigma
is usable. `response_threshold` controls raw candidate density. The production
path computes every candidate's structure-tensor ellipse and merges same-scale
ellipses that pass spatial-hash overlap lookup, discrete ellipse IoU,
orientation, center-pixel Lab color, and VGGT 3D path-continuity tests.
The path test walks every pixel between the two centers: all pixels must be
valid, and every adjacent 3D step must stay below one fixed per-view limit. The
reference scale is the median 3D step per image pixel over all valid 4/8-neighbor
edges in that view; the configured ratio multiplies this reference. A local
depth anomaly therefore cannot inflate its own tolerance.

Compatible edges are processed from highest to lowest IoU. Before each
union-find merge, confidence/response-weighted 2D moment matching checks the
prospective component. Its area may be at most 1.5 times its largest member and
at most 1500 pixels, and its axis ratio must remain within the production limit
of 4. An invalid edge is skipped while earlier valid subcomponents are kept;
the whole component is not discarded or reset.

`sampling.confidence_percentile` removes the requested bottom percentage of
finite, content-valid VGGT confidence scores. VGGT confidence is an unbounded
ranking score with a lower bound of one, not a probability, so an absolute
threshold in `[0, 1]` is not meaningful.

The production structure tensor sums weighted L, a, and b gradient outer
products, so a stable chromatic edge can determine ellipse orientation even
when its grayscale contrast is weak. The structure tensor determines ellipse
orientation and anisotropy.
`min_ellipse_area`, `max_ellipse_area`, and `max_axis_ratio` bound each raw
support. Same-scale merged supports use the separate relative/absolute limits
described above, so a merge may exceed the raw 800-pixel cap but never 1500.

The production config caps the 2D ellipse axis ratio at 4, requires at least 16
valid pixels per covariance, keeps only the locally continuous 3D component
connected to the keypoint, and caps the covariance condition number at 10,000.
Continuity compares every adjacent 3D step with three times the view-wide median
valid 3D step per image pixel. This gives every ellipse and merge path in one
view the same fixed edge-length limit while remaining invariant to a global
scene rescaling. The per-view reference scales are stored in output metadata as
`stats.continuity_reference_scales`.

The `covariance` section controls valid-point coverage, confidence weighting,
4/8-neighbor continuity and its maximum step ratio, compute device, and
per-chunk pixel budget. Setting `device: auto` selects CUDA when available.

The `fusion` section controls a pair graph built only inside each voxel.
An edge requires regularized combined-covariance Mahalanobis distance, normal
angle, sorted three-axis scale ratio, and CIELAB Delta E to pass their configured
limits. Union-find extracts connected components and confidence-weighted moment
matching produces one Gaussian per compatible component. A component whose
merged covariance fails the PCA filters falls back to its original proposals;
fusion therefore never drops a component solely because its merged Gaussian is
numerically invalid. Pair rejection counters and fallback counts are stored in
the output metadata under `stats`.

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
to one frame. Initialization requires prediction files carrying the matching
float32-head precision contract.

## gsplat Optimization

The training path consumes the activated initialization format and converts it
to gsplat's raw trainable convention: log-scales, opacity logits, wxyz
quaternions, `[N, 1, 3]` SH DC coefficients, and zero-initialized higher SH
bands. It validates the VGGT world-point/camera projection before rendering.

Install the training environment:

```bash
uv sync --extra dev --extra train
```

Render the initialization on one held-out view before optimizing:

```bash
uv run python -m gsplat_train.eval \
  --config configs/gsplat_train.yaml \
  --scene-root data/tnt_truck_48 \
  --model init/fused_gaussians.pt \
  --split test \
  --max-views 1 \
  --output gsplat/init_eval
```

Run a short fixed-topology optimization smoke test:

```bash
uv run python -m gsplat_train.train \
  --config configs/gsplat_train.yaml \
  --scene-root data/tnt_truck_48 \
  --max-steps 500 \
  --disable-densification \
  --output-dir gsplat/fixed_smoke
```

Run the configured 30,000-step optimization with `DefaultStrategy`
densification and pruning:

```bash
uv run python -m gsplat_train.train \
  --config configs/gsplat_train.yaml \
  --scene-root data/tnt_truck_48
```

Resume from a scene-relative checkpoint while increasing the total target
step count if needed:

```bash
uv run python -m gsplat_train.train \
  --config configs/gsplat_train.yaml \
  --scene-root data/tnt_truck_48 \
  --resume gsplat/checkpoints/step_006999.pt \
  --max-steps 30000
```

Evaluate the final model and save target/render/error/alpha images plus JSON
metrics:

```bash
uv run python -m gsplat_train.eval \
  --config configs/gsplat_train.yaml \
  --scene-root data/tnt_truck_48 \
  --split test
```

The final `gsplat/final_gaussians.pt` export contains activated scales and
opacities plus covariances reconstructed from the optimized scale/quaternion,
so the existing Viser viewer can open it directly:

```bash
uv run python scripts/view_gaussians.py \
  --input data/tnt_truck_48/gsplat/final_gaussians.pt
```

Training uses deterministic every-eighth-view holdout by default, masked
L1+SSIM loss, progressive SH degree, a scene-scale-adjusted position learning
rate, checkpoint/resume including optimizer and densification state, and a
viewer-compatible final export. Initialization and training settings remain in
separate config files: `configs/log_ellipse.yaml` and
`configs/gsplat_train.yaml`. The adapter also compensates the pinned gsplat
1.5.3 opacity-reset precedence bug without modifying the installed package.

See `docs/environment.md` for the environment policy.
