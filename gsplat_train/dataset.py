from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from init.io import load_images, load_vggt_predictions, resolve_scene_path


@dataclass(frozen=True)
class SceneData:
    images: np.ndarray
    intrinsics: np.ndarray | None
    extrinsics: np.ndarray | None

    def __len__(self) -> int:
        return int(self.images.shape[0])


def load_scene_data(config: dict[str, Any], *, scene_root_override: str | Path | None = None) -> SceneData:
    scene_cfg = config.get("scene", {})
    scene_root = Path(scene_root_override or scene_cfg.get("root", "data/scene_x"))
    images_dir = resolve_scene_path(scene_root, scene_cfg.get("images_dir", "images"))
    predictions_path = resolve_scene_path(scene_root, scene_cfg.get("predictions_path", "vggt/predictions.npz"))

    predictions = load_vggt_predictions(predictions_path)
    views, height, width, _ = predictions["world_points"].shape
    images = load_images(images_dir, target_size=(width, height))
    if images.shape[0] != views:
        raise ValueError(
            f"Image count ({images.shape[0]}) does not match prediction views ({views})"
        )

    return SceneData(
        images=images.astype(np.float32),
        intrinsics=predictions.get("intrinsics"),
        extrinsics=predictions.get("extrinsics"),
    )
