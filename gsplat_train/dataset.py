from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from init.io import load_images, load_vggt_predictions, resolve_scene_path


@dataclass(frozen=True)
class SceneData:
    images: np.ndarray
    image_valid_mask: np.ndarray
    intrinsics: np.ndarray
    extrinsics_c2w: np.ndarray
    extrinsics_w2c: np.ndarray
    scene_scale: float
    reprojection_error_px: float | None = None

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def height(self) -> int:
        return int(self.images.shape[1])

    @property
    def width(self) -> int:
        return int(self.images.shape[2])

    def frame(
        self, index: int, *, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.images[index]).to(device=device, dtype=torch.float32),
            torch.from_numpy(self.image_valid_mask[index]).to(device=device, dtype=torch.bool),
            torch.from_numpy(self.intrinsics[index]).to(device=device, dtype=torch.float32),
            torch.from_numpy(self.extrinsics_w2c[index]).to(device=device, dtype=torch.float32),
        )


def load_scene_data(
    config: dict[str, Any],
    *,
    scene_root_override: str | Path | None = None,
    validate_projection: bool = False,
) -> SceneData:
    scene_cfg = config.get("scene", {})
    scene_root = Path(scene_root_override or scene_cfg.get("root", "data/scene_x"))
    images_dir = resolve_scene_path(scene_root, scene_cfg.get("images_dir", "images"))
    predictions_path = resolve_scene_path(
        scene_root, scene_cfg.get("predictions_path", "vggt/predictions.npz")
    )

    predictions = load_vggt_predictions(predictions_path)
    world_points = predictions["world_points"]
    views, height, width, _ = world_points.shape
    if "processed_images" in predictions:
        images = predictions["processed_images"]
    else:
        images = load_images(images_dir)
    if images.shape[0] != views:
        raise ValueError(
            f"Image count ({images.shape[0]}) does not match prediction views ({views})"
        )
    if images.shape[1:3] != (height, width):
        raise ValueError(
            "Scene images are not pixel-aligned with VGGT predictions: "
            f"images have {images.shape[1:3]}, predictions have {(height, width)}"
        )
    image_valid_mask = np.asarray(
        predictions.get(
            "processed_valid_mask",
            np.ones((views, height, width), dtype=bool),
        ),
        dtype=bool,
    )
    if image_valid_mask.shape != (views, height, width):
        raise ValueError(
            "processed_valid_mask must have shape "
            f"{(views, height, width)}, got {image_valid_mask.shape}"
        )

    intrinsics = predictions.get("intrinsics")
    if intrinsics is None:
        raise ValueError("VGGT predictions must contain intrinsics for gsplat rendering")
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape != (views, 3, 3):
        raise ValueError(f"intrinsics must have shape {(views, 3, 3)}, got {intrinsics.shape}")

    extrinsics_c2w = predictions.get("extrinsics_c2w", predictions.get("extrinsics"))
    extrinsics_w2c = predictions.get("extrinsics_w2c")
    if extrinsics_c2w is None and extrinsics_w2c is None:
        raise ValueError("VGGT predictions must contain c2w or w2c extrinsics")
    if extrinsics_c2w is not None:
        extrinsics_c2w = np.asarray(extrinsics_c2w, dtype=np.float32)
    if extrinsics_w2c is not None:
        extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    if extrinsics_c2w is not None and extrinsics_w2c is None:
        extrinsics_w2c = np.linalg.inv(extrinsics_c2w).astype(np.float32)
    elif extrinsics_w2c is not None and extrinsics_c2w is None:
        extrinsics_c2w = np.linalg.inv(extrinsics_w2c).astype(np.float32)
    assert extrinsics_c2w is not None and extrinsics_w2c is not None
    if extrinsics_c2w.shape != (views, 4, 4) or extrinsics_w2c.shape != (views, 4, 4):
        raise ValueError("Camera extrinsics must have shape [V, 4, 4]")

    arrays = {
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics_c2w": extrinsics_c2w,
        "extrinsics_w2c": extrinsics_w2c,
    }
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    if images.min() < 0.0 or images.max() > 1.0:
        raise ValueError("Scene images must use floating-point RGB values in [0, 1]")
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise ValueError("Camera focal lengths must be positive")
    inverse_error = float(
        np.max(
            np.abs(
                extrinsics_w2c.astype(np.float64)
                @ extrinsics_c2w.astype(np.float64)
                - np.eye(4, dtype=np.float64)
            )
        )
    )
    if inverse_error > 1.0e-3:
        raise ValueError(f"c2w/w2c matrices are inconsistent (max error {inverse_error:.6g})")

    reprojection_error = None
    if validate_projection:
        reprojection_error = validate_world_point_projection(
            world_points,
            image_valid_mask,
            intrinsics,
            extrinsics_w2c,
            max_error_px=float(config.get("training", {}).get("max_reprojection_error_px", 0.5)),
        )

    return SceneData(
        images=images.astype(np.float32),
        image_valid_mask=image_valid_mask,
        intrinsics=intrinsics,
        extrinsics_c2w=extrinsics_c2w,
        extrinsics_w2c=extrinsics_w2c,
        scene_scale=estimate_scene_scale(extrinsics_c2w),
        reprojection_error_px=reprojection_error,
    )


def split_view_indices(view_count: int, *, test_every: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    indices = np.arange(view_count, dtype=np.int64)
    if test_every <= 0 or view_count == 1:
        return indices, np.empty(0, dtype=np.int64)
    test_mask = indices % int(test_every) == 0
    if np.all(test_mask):
        test_mask[-1] = False
    return indices[~test_mask], indices[test_mask]


def estimate_scene_scale(extrinsics_c2w: np.ndarray) -> float:
    camera_centers = np.asarray(extrinsics_c2w, dtype=np.float64)[:, :3, 3]
    center = np.median(camera_centers, axis=0)
    radius = float(np.max(np.linalg.norm(camera_centers - center, axis=1)))
    return max(radius, 1.0e-6)


def validate_world_point_projection(
    world_points: np.ndarray,
    valid_mask: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    max_error_px: float,
    sample_stride: int = 16,
) -> float:
    if max_error_px <= 0.0:
        raise ValueError("max_error_px must be positive")
    views, height, width, _ = world_points.shape
    ys, xs = np.mgrid[0:height:sample_stride, 0:width:sample_stride]
    errors: list[np.ndarray] = []
    positive_depth = 0
    sample_count = 0
    for view in range(views):
        points = world_points[view, ::sample_stride, ::sample_stride]
        usable = valid_mask[view, ::sample_stride, ::sample_stride]
        usable &= np.isfinite(points).all(axis=-1)
        points = points[usable].astype(np.float64)
        if points.size == 0:
            continue
        expected = np.stack([xs[usable], ys[usable]], axis=-1).astype(np.float64)
        homogeneous = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=-1
        )
        camera = (extrinsics_w2c[view].astype(np.float64) @ homogeneous.T).T[:, :3]
        projected = (intrinsics[view].astype(np.float64) @ camera.T).T
        nonzero = np.abs(projected[:, 2]) > 1.0e-12
        projected_xy = projected[nonzero, :2] / projected[nonzero, 2:3]
        errors.append(np.linalg.norm(projected_xy - expected[nonzero], axis=-1))
        positive_depth += int(np.count_nonzero(camera[:, 2] > 0.0))
        sample_count += int(camera.shape[0])
    if not errors or sample_count == 0:
        raise ValueError("No valid world points were available for camera projection validation")
    if positive_depth != sample_count:
        fraction = positive_depth / sample_count
        raise ValueError(f"Only {fraction:.3%} of sampled world points have positive camera depth")
    error = float(np.percentile(np.concatenate(errors), 95.0))
    if not np.isfinite(error) or error > max_error_px:
        raise ValueError(
            f"VGGT world points do not match camera projection: p95 error {error:.4f}px "
            f"> {max_error_px:.4f}px"
        )
    return error
