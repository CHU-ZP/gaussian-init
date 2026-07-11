from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from init.build_init import build_gaussian_initialization


def test_build_init_from_mock_predictions(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    images_dir = scene_root / "images"
    vggt_dir = scene_root / "vggt"
    images_dir.mkdir(parents=True)
    vggt_dir.mkdir(parents=True)

    views, height, width = 2, 24, 32
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    world_points = np.empty((views, height, width, 3), dtype=np.float32)
    confidence = np.ones((views, height, width), dtype=np.float32)
    for view in range(views):
        world_points[view, ..., 0] = (xx - width * 0.5) / width + 0.01 * view
        world_points[view, ..., 1] = (yy - height * 0.5) / width
        world_points[view, ..., 2] = 1.0

        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.clip(xx / width * 255, 0, 255).astype(np.uint8)
        image[..., 1] = np.clip(yy / height * 255, 0, 255).astype(np.uint8)
        image[..., 2] = 128
        Image.fromarray(image).save(images_dir / f"{view:03d}.png")

    np.savez_compressed(
        vggt_dir / "predictions.npz",
        world_points=world_points,
        confidence=confidence,
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None, ...], views, axis=0),
        extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None, ...], views, axis=0),
    )

    config = {
        "scene": {
            "root": str(scene_root),
            "images_dir": "images",
            "predictions_path": "vggt/predictions.npz",
            "proposals_path": "init/proposals.pt",
            "output_path": "init/fused_gaussians.pt",
        },
        "sampling": {
            "mode": "uniform",
            "stride": 8,
            "max_samples_per_view": 64,
            "confidence_threshold": 0.1,
            "salient_fraction": 0.0,
            "min_distance": 2,
        },
        "patch": {
            "radius": 2,
            "min_valid_points": 8,
            "max_center_distance": None,
        },
        "pca": {
            "eigenvalue_epsilon": 1.0e-8,
            "scale_min": 1.0e-5,
            "scale_max": 1.0,
            "condition_max": 1.0e8,
        },
        "gaussian": {"opacity": 0.1},
        "fusion": {"enabled": True, "voxel_size": 0.05},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    proposals, fused, stats = build_gaussian_initialization(config)

    assert stats["sampled_pixels"] > 0
    assert len(proposals) > 0
    assert len(fused) > 0
    assert np.isfinite(fused.means).all()
    assert np.isfinite(fused.scales).all()
    assert np.allclose(np.linalg.norm(fused.quats, axis=1), 1.0, atol=1.0e-5)

    state = torch.load(
        scene_root / "init" / "fused_gaussians.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert state["means"].shape[-1] == 3
    assert state["scales"].shape[-1] == 3
    assert state["quats"].shape[-1] == 4
