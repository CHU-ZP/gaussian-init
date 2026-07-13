from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from init.io import load_images, load_vggt_predictions, resolve_scene_path


@dataclass(frozen=True)
class SceneData:
    images: np.ndarray
    image_valid_mask: np.ndarray
    intrinsics: np.ndarray | None
    extrinsics_c2w: np.ndarray | None
    extrinsics_w2c: np.ndarray | None

    def __len__(self) -> int:
        return int(self.images.shape[0])


def load_scene_data(
    config: dict[str, Any], *, scene_root_override: str | Path | None = None
) -> SceneData:
    scene_cfg = config.get("scene", {})
    scene_root = Path(scene_root_override or scene_cfg.get("root", "data/scene_x"))
    images_dir = resolve_scene_path(scene_root, scene_cfg.get("images_dir", "images"))
    predictions_path = resolve_scene_path(
        scene_root, scene_cfg.get("predictions_path", "vggt/predictions.npz")
    )

    predictions = load_vggt_predictions(predictions_path)
    views, height, width, _ = predictions["world_points"].shape
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

    extrinsics_c2w = predictions.get("extrinsics_c2w", predictions.get("extrinsics"))
    extrinsics_w2c = predictions.get("extrinsics_w2c")
    if extrinsics_c2w is not None and extrinsics_w2c is None:
        extrinsics_w2c = np.linalg.inv(extrinsics_c2w).astype(np.float32)
    elif extrinsics_w2c is not None and extrinsics_c2w is None:
        extrinsics_c2w = np.linalg.inv(extrinsics_w2c).astype(np.float32)

    return SceneData(
        images=images.astype(np.float32),
        image_valid_mask=image_valid_mask,
        intrinsics=predictions.get("intrinsics"),
        extrinsics_c2w=extrinsics_c2w,
        extrinsics_w2c=extrinsics_w2c,
    )
