from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gsplat_train.model import GaussianModel
from init.build_init import build_gaussian_initialization, validate_initialization_config
from init.filters import PCAFilterConfig
from init.fusion import FusionConfig
from init.gaussian_params import scale_quat_to_covariance
from init.grid_supplement import GridSupplementConfig
from init.io import load_config
from init.sampling import SamplingConfig


def test_build_init_from_multiscale_ellipse_predictions(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    images_dir = scene_root / "images"
    vggt_dir = scene_root / "vggt"
    images_dir.mkdir(parents=True)
    vggt_dir.mkdir(parents=True)

    views, height, width = 2, 72, 80
    yy, xx = np.mgrid[:height, :width]
    world_points = np.empty((views, height, width, 3), dtype=np.float32)
    processed_images = np.empty((views, height, width, 3), dtype=np.float32)
    for view in range(views):
        world_points[view, ..., 0] = (xx - width * 0.5) / width + 0.01 * view
        world_points[view, ..., 1] = (yy - height * 0.5) / width
        world_points[view, ..., 2] = 1.0 + 1.0e-4 * (xx - width * 0.5) ** 2

        blob = np.exp(-((xx - 28) ** 2 + (yy - 31) ** 2) / (2.0 * 4.0**2))
        blob += 0.7 * np.exp(-((xx - 58) ** 2 + (yy - 46) ** 2) / (2.0 * 8.0**2))
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.clip((0.15 + 0.8 * blob) * 255.0, 0, 255).astype(np.uint8)
        image[..., 1] = np.clip((0.2 + 0.5 * blob) * 255.0, 0, 255).astype(np.uint8)
        image[..., 2] = np.clip((0.25 + 0.3 * blob) * 255.0, 0, 255).astype(np.uint8)
        processed_images[view] = image.astype(np.float32) / 255.0
        # The build must use the exact geometry-aligned RGB stored in the npz,
        # not independently resize/reload these deliberately different files.
        Image.fromarray(np.zeros_like(image)).save(images_dir / f"{view:03d}.png")

    np.savez_compressed(
        vggt_dir / "predictions.npz",
        world_points=world_points,
        processed_images=processed_images,
        processed_valid_mask=np.ones((views, height, width), dtype=bool),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], views, axis=0),
        extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], views, axis=0),
    )
    config = make_config(scene_root)
    proposals, fused, stats = build_gaussian_initialization(config)

    assert stats["detected_keypoints"] > 0
    assert len(proposals) > 0
    assert fused is proposals
    assert stats["total_regions"] == (
        stats["accepted_proposals"] + stats["rejected_covariance"] + stats["rejected_pca"]
    )
    assert stats["ellipse_regions"] == stats["detected_keypoints"]
    assert stats["grid_regions"] > 0
    assert stats["accepted_proposals"] == (
        stats["accepted_ellipse_proposals"] + stats["accepted_grid_proposals"]
    )
    assert np.isfinite(proposals.means).all()
    assert np.isfinite(proposals.scales).all()
    assert np.allclose(np.linalg.norm(proposals.quats, axis=1), 1.0, atol=1.0e-5)
    for covariance, scales, quat in zip(
        proposals.covariances,
        proposals.scales,
        proposals.quats,
        strict=True,
    ):
        assert np.allclose(scale_quat_to_covariance(scales, quat), covariance, atol=1.0e-6)

    output_path = scene_root / "init" / "fused_gaussians.pt"
    state = torch.load(output_path, map_location="cpu", weights_only=False)
    assert "sh_dc" in state
    assert "colors" not in state
    model = GaussianModel.from_file(output_path)
    assert torch.allclose(model.sh_dc.detach(), state["sh_dc"])
    assert torch.any(model.sh_dc < 0.0)

    no_grid_config = copy.deepcopy(config)
    no_grid_config["sampling"]["grid_supplement"] = {"enabled": False}
    no_grid_config["scene"]["proposals_path"] = "init/no_grid_proposals.pt"
    no_grid_config["scene"]["output_path"] = "init/no_grid_fused.pt"
    ellipse_only, _, no_grid_stats = build_gaussian_initialization(no_grid_config)
    assert no_grid_stats["accepted_grid_proposals"] == 0
    assert no_grid_stats["accepted_ellipse_proposals"] == stats["accepted_ellipse_proposals"]
    _assert_ellipse_proposals_unchanged(ellipse_only, proposals)

    production_config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "log_ellipse.yaml"
    )
    production_config["scene"]["root"] = str(scene_root)
    production_config["covariance"]["device"] = "cpu"
    production_config["fusion"]["enabled"] = False
    production_proposals, _, _ = build_gaussian_initialization(production_config)
    assert len(production_proposals) > 0


def make_config(scene_root: Path) -> dict:
    return {
        "scene": {
            "root": str(scene_root),
            "images_dir": "images",
            "predictions_path": "vggt/predictions.npz",
            "proposals_path": "init/proposals.pt",
            "output_path": "init/fused_gaussians.pt",
        },
        "sampling": {
            "sigmas": [1.0, 2.0, 3.0, 4.5, 7.0, 10.0],
            "response_threshold": 1.0,
            "max_keypoints_per_view": 128,
            "structure_sigma_factor": 1.5,
            "ellipse_radius_factor": 2.5,
            "min_ellipse_area": 12.0,
            "max_ellipse_area": 400.0,
            "max_axis_ratio": 4.0,
        },
        "covariance": {
            "min_valid_points": 16,
            "min_valid_fraction": 0.6,
            "continuity_neighbors": 8,
            "continuity_ratio_max": 3.0,
            "device": "cpu",
            "pixel_budget": 100000,
        },
        "pca": {
            "eigenvalue_epsilon": 1.0e-8,
            "scale_min": 1.0e-5,
            "scale_max": 1.0,
            "min_secondary_eigenvalue_ratio": 0.01,
            "normal_thickness_step_factor": 0.25,
        },
        "gaussian": {"opacity": 0.2},
        "fusion": {"enabled": False, "voxel_size": 0.05},
    }


def _assert_ellipse_proposals_unchanged(ellipse_only, combined) -> None:
    fields = (
        "means",
        "covariances",
        "scales",
        "quats",
        "sh_dc",
        "opacities",
        "view_ids",
        "scores",
    )
    for field in fields:
        expected = getattr(ellipse_only, field)
        actual_by_view = []
        for view in np.unique(ellipse_only.view_ids):
            expected_view = expected[ellipse_only.view_ids == view]
            combined_view = getattr(combined, field)[combined.view_ids == view]
            actual_by_view.append(combined_view[: len(expected_view)])
        actual = np.concatenate(actual_by_view, axis=0)
        assert np.array_equal(actual, expected), field


def test_unknown_initialization_config_key_is_rejected() -> None:
    with np.testing.assert_raises_regex(ValueError, "Unknown covariance config key"):
        validate_initialization_config({"covariance": {"unexpected_option": 0.1}})

    with np.testing.assert_raises_regex(ValueError, "Unknown sampling.ellipse_merge config key"):
        validate_initialization_config({"sampling": {"ellipse_merge": {"typo": 0.1}}})

    with np.testing.assert_raises_regex(
        ValueError, "Unknown sampling.grid_supplement config key"
    ):
        validate_initialization_config({"sampling": {"grid_supplement": {"typo": 0.1}}})


def test_sampling_defaults_match_production_config() -> None:
    production_config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "log_ellipse.yaml"
    )
    assert SamplingConfig.from_mapping(production_config["sampling"]) == SamplingConfig()
    assert GridSupplementConfig.from_mapping(
        production_config["sampling"]["grid_supplement"]
    ) == GridSupplementConfig()
    assert FusionConfig.from_mapping(production_config["fusion"]) == FusionConfig()
    assert PCAFilterConfig.from_config(production_config["pca"]) == PCAFilterConfig(
        scale_min=1.0e-5,
        scale_max=1.0,
        min_secondary_eigenvalue_ratio=0.01,
    )
