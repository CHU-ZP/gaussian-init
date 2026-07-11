from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .filters import PCAFilterConfig, valid_pca
from .fusion import voxel_fuse
from .gaussian_params import rotation_matrix_to_quaternion
from .io import (
    load_config,
    load_images,
    load_vggt_predictions,
    resolve_scene_path,
    save_gaussians,
    write_ply,
)
from .patch import extract_local_patch
from .pca import estimate_local_pca
from .sampling import sample_pixels
from .types import GaussianProposals


def build_gaussian_initialization(
    config: dict[str, Any],
    *,
    scene_root_override: str | Path | None = None,
    predictions_override: str | Path | None = None,
    output_override: str | Path | None = None,
    force_no_fusion: bool = False,
) -> tuple[GaussianProposals, GaussianProposals, dict[str, Any]]:
    scene_cfg = config.get("scene", {})
    scene_root = Path(scene_root_override or scene_cfg.get("root", "data/scene_x"))
    predictions_path = resolve_scene_path(
        scene_root,
        predictions_override or scene_cfg.get("predictions_path", "vggt/predictions.npz"),
    )
    proposals_path = resolve_scene_path(scene_root, scene_cfg.get("proposals_path", "init/proposals.pt"))
    output_path = resolve_scene_path(
        scene_root,
        output_override or scene_cfg.get("output_path", "init/fused_gaussians.pt"),
    )

    predictions = load_vggt_predictions(predictions_path)
    world_points = predictions["world_points"]
    confidence = predictions["confidence"]
    views, height, width, _ = world_points.shape

    images = load_scene_images(
        scene_root,
        scene_cfg.get("images_dir", "images"),
        views=views,
        height=height,
        width=width,
    )

    sampling_cfg = config.get("sampling", {})
    patch_cfg = config.get("patch", {})
    pca_cfg = config.get("pca", {})
    gaussian_cfg = config.get("gaussian", {})
    fusion_cfg = config.get("fusion", {})

    eigenvalue_epsilon = float(pca_cfg.get("eigenvalue_epsilon", 1.0e-8))
    pca_filter = PCAFilterConfig.from_config(pca_cfg)
    opacity = float(gaussian_cfg.get("opacity", 0.1))
    confidence_threshold = float(sampling_cfg.get("confidence_threshold", 0.2))

    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    quats: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    opacities: list[float] = []
    confidences: list[float] = []
    view_ids: list[int] = []
    scores: list[float] = []

    stats: dict[str, Any] = {
        "views": views,
        "image_height": height,
        "image_width": width,
        "sampled_pixels": 0,
        "accepted_proposals": 0,
        "rejected_patch": 0,
        "rejected_pca": 0,
    }

    for view in range(views):
        samples = sample_pixels(
            view_id=view,
            image=images[view],
            confidence=confidence[view],
            world_points=world_points[view],
            mode=str(sampling_cfg.get("mode", "uniform")),
            stride=int(sampling_cfg.get("stride", 16)),
            max_samples=int(sampling_cfg.get("max_samples_per_view", 8000)),
            confidence_threshold=confidence_threshold,
            salient_fraction=float(sampling_cfg.get("salient_fraction", 0.0)),
            min_distance=int(sampling_cfg.get("min_distance", 4)),
        )
        stats["sampled_pixels"] += len(samples)

        for u, v, score in zip(samples.us, samples.vs, samples.scores, strict=True):
            patch = extract_local_patch(
                world_points[view],
                confidence[view],
                u=int(u),
                v=int(v),
                radius=int(patch_cfg.get("radius", 3)),
                min_valid_points=int(patch_cfg.get("min_valid_points", 8)),
                confidence_threshold=confidence_threshold,
                max_center_distance=patch_cfg.get("max_center_distance"),
            )
            if patch is None:
                stats["rejected_patch"] += 1
                continue

            pca_result = estimate_local_pca(
                patch.points,
                eigenvalue_epsilon=eigenvalue_epsilon,
            )
            if not valid_pca(pca_result, pca_filter):
                stats["rejected_pca"] += 1
                continue

            means.append(patch.center)
            covariances.append(pca_result.covariance)
            scales.append(pca_result.scales)
            quats.append(rotation_matrix_to_quaternion(pca_result.basis))
            colors.append(images[view, int(v), int(u)].astype(np.float32))
            opacities.append(opacity)
            confidences.append(patch.mean_confidence)
            view_ids.append(view)
            scores.append(float(score))

    proposals = GaussianProposals.from_lists(
        means=means,
        covariances=covariances,
        scales=scales,
        quats=quats,
        colors=colors,
        opacities=opacities,
        confidences=confidences,
        view_ids=view_ids,
        scores=scores,
    )
    stats["accepted_proposals"] = len(proposals)

    save_gaussians(
        proposals_path,
        proposals,
        metadata={
            "stage": "per_view_proposals",
            "source_predictions": str(predictions_path),
            "stats": dict(stats),
        },
    )

    fusion_enabled = bool(fusion_cfg.get("enabled", False)) and not force_no_fusion
    if fusion_enabled:
        fused = voxel_fuse(
            proposals,
            voxel_size=float(fusion_cfg.get("voxel_size", 0.02)),
            eigenvalue_epsilon=eigenvalue_epsilon,
        )
    else:
        fused = proposals

    stats["fused_gaussians"] = len(fused)
    stats["fusion_enabled"] = fusion_enabled
    save_gaussians(
        output_path,
        fused,
        metadata={
            "stage": "fused_gaussians",
            "source_predictions": str(predictions_path),
            "source_proposals": str(proposals_path),
            "stats": dict(stats),
        },
    )

    debug_ply = output_path.parent / "debug.ply"
    write_ply(debug_ply, fused.means, fused.colors)
    stats["proposals_path"] = str(proposals_path)
    stats["output_path"] = str(output_path)
    stats["debug_ply"] = str(debug_ply)
    return proposals, fused, stats


def load_scene_images(
    scene_root: Path,
    images_dir: str | Path,
    *,
    views: int,
    height: int,
    width: int,
) -> np.ndarray:
    image_root = resolve_scene_path(scene_root, images_dir)
    try:
        images = load_images(image_root, target_size=(width, height))
    except FileNotFoundError:
        return np.full((views, height, width, 3), 0.5, dtype=np.float32)

    if images.shape[0] != views:
        raise ValueError(
            f"Image count ({images.shape[0]}) does not match prediction views ({views})"
        )
    return images.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PCA-initialized Gaussian parameters.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--scene-root", default=None, help="Override scene root from config.")
    parser.add_argument("--predictions", default=None, help="Override predictions npz path.")
    parser.add_argument("--output", default=None, help="Override output torch file.")
    parser.add_argument("--no-fusion", action="store_true", help="Disable fusion even if config enables it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    proposals, fused, stats = build_gaussian_initialization(
        config,
        scene_root_override=args.scene_root,
        predictions_override=args.predictions,
        output_override=args.output,
        force_no_fusion=args.no_fusion,
    )
    print(
        "Built "
        f"{len(proposals)} per-view proposals and {len(fused)} final Gaussians "
        f"from {stats['sampled_pixels']} sampled pixels."
    )
    print(f"Saved proposals: {stats['proposals_path']}")
    print(f"Saved final init: {stats['output_path']}")
    print(f"Saved debug PLY: {stats['debug_ply']}")


if __name__ == "__main__":
    main()
