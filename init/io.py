from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .types import GaussianProposals

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_scene_path(scene_root: str | Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(scene_root) / candidate


def load_vggt_predictions(
    path: str | Path,
    *,
    extrinsics_are_c2w: bool = True,
) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"VGGT predictions not found: {npz_path}")

    data = np.load(npz_path)
    keys = set(data.files)

    if "world_points" in keys:
        world_points = np.asarray(data["world_points"], dtype=np.float32)
    elif {"depth", "intrinsics", "extrinsics"}.issubset(keys):
        world_points = unproject_depths(
            np.asarray(data["depth"], dtype=np.float32),
            np.asarray(data["intrinsics"], dtype=np.float32),
            np.asarray(data["extrinsics"], dtype=np.float32),
            extrinsics_are_c2w=extrinsics_are_c2w,
        )
    else:
        raise ValueError(
            "predictions.npz must include world_points, or depth + intrinsics + extrinsics"
        )

    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape [V, H, W, 3]")

    views, height, width, _ = world_points.shape
    if "confidence" in keys:
        confidence = np.asarray(data["confidence"], dtype=np.float32)
    else:
        confidence = np.ones((views, height, width), dtype=np.float32)

    if confidence.shape != (views, height, width):
        raise ValueError("confidence must have shape [V, H, W]")

    output = {
        "world_points": world_points,
        "confidence": confidence,
    }
    if "processed_images" in keys:
        processed_images = np.asarray(data["processed_images"], dtype=np.float32)
        if processed_images.shape != (views, height, width, 3):
            raise ValueError("processed_images must have shape [V, H, W, 3]")
        output["processed_images"] = processed_images
    if "processed_valid_mask" in keys:
        processed_valid_mask = np.asarray(data["processed_valid_mask"], dtype=bool)
        if processed_valid_mask.shape != (views, height, width):
            raise ValueError("processed_valid_mask must have shape [V, H, W]")
        output["processed_valid_mask"] = processed_valid_mask
    for key in ("depth", "intrinsics", "extrinsics", "extrinsics_c2w", "extrinsics_w2c"):
        if key in keys:
            output[key] = np.asarray(data[key], dtype=np.float32)
    for key in (
        "model_id",
        "world_points_source",
        "aggregator_dtype",
        "head_dtype",
        "head_frames_chunk_size",
        "precision_contract",
        "confidence_semantics",
    ):
        if key in keys:
            output[key] = np.asarray(data[key])
    return output


def unproject_depths(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    extrinsics_are_c2w: bool = True,
) -> np.ndarray:
    if depth.ndim != 3:
        raise ValueError("depth must have shape [V, H, W]")

    views, height, width = depth.shape
    if intrinsics.shape != (views, 3, 3):
        raise ValueError("intrinsics must have shape [V, 3, 3]")
    if extrinsics.shape != (views, 4, 4):
        raise ValueError("extrinsics must have shape [V, 4, 4]")

    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    world_points = np.empty((views, height, width, 3), dtype=np.float32)

    for view in range(views):
        k = intrinsics[view]
        z = depth[view]
        x_cam = (xs - k[0, 2]) * z / k[0, 0]
        y_cam = (ys - k[1, 2]) * z / k[1, 1]
        cam_points = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)

        c2w = extrinsics[view] if extrinsics_are_c2w else np.linalg.inv(extrinsics[view])
        world = cam_points.reshape(-1, 4) @ c2w.T
        world_points[view] = world[:, :3].reshape(height, width, 3)

    return world_points


def load_images(images_dir: str | Path) -> np.ndarray:
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    paths = sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found in: {root}")

    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(images, axis=0)


def save_gaussians(
    path: str | Path,
    proposals: GaussianProposals,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = proposals.to_torch_dict()
    payload["metadata"] = metadata or {}
    torch.save(payload, output_path)


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors_u8 = np.full((points.shape[0], 3), 200, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    with output_path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors_u8, strict=True):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
