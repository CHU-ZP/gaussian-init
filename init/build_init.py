from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .ellipses import compute_covariances
from .filters import PCAFilterConfig, valid_pca
from .fusion import voxel_fuse
from .gaussian_params import rgb_to_sh_dc, rotation_matrix_to_quaternion, sh_dc_to_rgb
from .io import (
    load_config,
    load_images,
    load_vggt_predictions,
    resolve_scene_path,
    save_gaussians,
    write_ply,
)
from .pca import decompose_covariance
from .sampling import detect_multiscale_keypoints
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
    proposals_path = resolve_scene_path(
        scene_root, scene_cfg.get("proposals_path", "init/proposals.pt")
    )
    output_path = resolve_scene_path(
        scene_root,
        output_override or scene_cfg.get("output_path", "init/fused_gaussians.pt"),
    )

    predictions = load_vggt_predictions(predictions_path)
    validate_vggt_precision_contract(predictions)
    world_points = predictions["world_points"]
    confidence = predictions["confidence"]
    views, height, width, _ = world_points.shape
    images = load_scene_images(
        scene_root,
        scene_cfg.get("images_dir", "images"),
        views=views,
        height=height,
        width=width,
        processed_images=predictions.get("processed_images"),
    )
    image_valid_masks = np.asarray(
        predictions.get(
            "processed_valid_mask",
            np.ones((views, height, width), dtype=bool),
        ),
        dtype=bool,
    )

    sampling_cfg = config.get("sampling", {})
    covariance_cfg = config.get("covariance", {})
    pca_cfg = config.get("pca", {})
    gaussian_cfg = config.get("gaussian", {})
    fusion_cfg = config.get("fusion", {})
    _reject_removed_config(config)

    eigenvalue_epsilon = float(pca_cfg.get("eigenvalue_epsilon", 1.0e-8))
    pca_filter = PCAFilterConfig.from_config(pca_cfg)
    opacity = float(gaussian_cfg.get("opacity", 0.1))
    confidence_percentile = float(sampling_cfg.get("confidence_percentile", 25.0))
    confidence_threshold = confidence_threshold_from_percentile(
        confidence,
        world_points,
        image_valid_masks,
        percentile=confidence_percentile,
    )

    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    quats: list[np.ndarray] = []
    sh_dc: list[np.ndarray] = []
    opacities: list[float] = []
    confidences: list[float] = []
    view_ids: list[int] = []
    scores: list[float] = []

    stats: dict[str, Any] = {
        "views": views,
        "image_height": height,
        "image_width": width,
        "detected_keypoints": 0,
        "accepted_proposals": 0,
        "rejected_covariance": 0,
        "rejected_pca": 0,
        "confidence_percentile": confidence_percentile,
        "confidence_threshold_value": confidence_threshold,
    }

    for view in range(views):
        keypoints = detect_multiscale_keypoints(
            view_id=view,
            image=images[view],
            confidence=confidence[view],
            world_points=world_points[view],
            sigmas=sampling_cfg.get("sigmas", [1.0, 1.6, 2.5, 4.0, 6.4]),
            response_threshold=float(sampling_cfg.get("response_threshold", 0.005)),
            max_keypoints=int(sampling_cfg.get("max_keypoints_per_view", 10000)),
            min_distance=int(sampling_cfg.get("min_distance", 3)),
            nms_radius_factor=float(sampling_cfg.get("nms_radius_factor", 3.0)),
            structure_sigma_factor=float(sampling_cfg.get("structure_sigma_factor", 1.5)),
            ellipse_radius_factor=float(sampling_cfg.get("ellipse_radius_factor", 2.5)),
            min_ellipse_area=float(sampling_cfg.get("min_ellipse_area", 12.0)),
            max_ellipse_area=float(sampling_cfg.get("max_ellipse_area", 800.0)),
            max_axis_ratio=float(sampling_cfg.get("max_axis_ratio", 8.0)),
            confidence_threshold=confidence_threshold,
            image_valid_mask=image_valid_masks[view],
        )
        stats["detected_keypoints"] += len(keypoints)
        covariance_results = compute_covariances(
            world_points[view],
            confidence[view],
            keypoints.us,
            keypoints.vs,
            keypoints.ellipse_matrices,
            image_valid_mask=image_valid_masks[view],
            confidence_threshold=confidence_threshold,
            min_valid_points=int(covariance_cfg.get("min_valid_points", 8)),
            min_valid_fraction=float(covariance_cfg.get("min_valid_fraction", 0.6)),
            max_center_distance=covariance_cfg.get("max_center_distance"),
            confidence_weighted=bool(covariance_cfg.get("confidence_weighted", True)),
            device=str(covariance_cfg.get("device", "auto")),
            pixel_budget=int(covariance_cfg.get("pixel_budget", 2_000_000)),
        )
        stats["rejected_covariance"] += int(np.count_nonzero(~covariance_results.valid))

        for index in np.flatnonzero(covariance_results.valid):
            pca_result = decompose_covariance(
                covariance_results.covariances[index],
                eigenvalue_epsilon=eigenvalue_epsilon,
            )
            if not valid_pca(pca_result, pca_filter):
                stats["rejected_pca"] += 1
                continue

            u = int(keypoints.us[index])
            v = int(keypoints.vs[index])
            means.append(world_points[view, v, u].astype(np.float32))
            covariances.append(pca_result.covariance)
            scales.append(pca_result.scales)
            quats.append(rotation_matrix_to_quaternion(pca_result.basis))
            sh_dc.append(rgb_to_sh_dc(images[view, v, u]))
            opacities.append(opacity)
            confidences.append(float(covariance_results.mean_confidences[index]))
            view_ids.append(view)
            scores.append(float(keypoints.scores[index]))

    proposals = GaussianProposals.from_lists(
        means=means,
        covariances=covariances,
        scales=scales,
        quats=quats,
        sh_dc=sh_dc,
        opacities=opacities,
        confidences=confidences,
        view_ids=view_ids,
        scores=scores,
    )
    stats["accepted_proposals"] = len(proposals)
    if len(proposals) == 0:
        raise RuntimeError(
            "No Gaussian proposals survived LoG detection, ellipse coverage, and PCA filtering. "
            "Check image/prediction alignment and confidence, LoG, and PCA thresholds. "
            f"Stats: {stats}"
        )
    save_gaussians(
        proposals_path,
        proposals,
        metadata={
            "stage": "per_view_log_ellipse_proposals",
            "source_predictions": str(predictions_path),
            "stats": dict(stats),
        },
    )

    fusion_enabled = bool(fusion_cfg.get("enabled", False)) and not force_no_fusion
    if fusion_enabled:
        voxel_size = float(fusion_cfg.get("voxel_size", 0.02))
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("fusion.voxel_size must be finite and positive")
        pre_filter_clusters = len(
            np.unique(np.floor(proposals.means / voxel_size).astype(np.int64), axis=0)
        )
        fused = voxel_fuse(
            proposals,
            voxel_size=voxel_size,
            eigenvalue_epsilon=eigenvalue_epsilon,
            pca_filter=pca_filter,
        )
        stats["rejected_fusion_pca"] = pre_filter_clusters - len(fused)
        if len(fused) == 0:
            raise RuntimeError("All fused Gaussians failed the configured PCA filters")
    else:
        fused = proposals
        stats["rejected_fusion_pca"] = 0

    stats["fused_gaussians"] = len(fused)
    stats["fusion_enabled"] = fusion_enabled
    save_gaussians(
        output_path,
        fused,
        metadata={
            "stage": "fused_log_ellipse_gaussians",
            "source_predictions": str(predictions_path),
            "source_proposals": str(proposals_path),
            "stats": dict(stats),
        },
    )

    debug_ply = output_path.parent / "debug.ply"
    write_ply(debug_ply, fused.means, sh_dc_to_rgb(fused.sh_dc))
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
    processed_images: np.ndarray | None = None,
) -> np.ndarray:
    if processed_images is not None:
        images = np.asarray(processed_images, dtype=np.float32)
        if images.shape != (views, height, width, 3):
            raise ValueError("processed_images are not aligned with VGGT predictions")
        return images

    image_root = resolve_scene_path(scene_root, images_dir)
    images = load_images(image_root)

    if images.shape[0] != views:
        raise ValueError(
            f"Image count ({images.shape[0]}) does not match prediction views ({views})"
        )
    if images.shape[1:3] != (height, width):
        raise ValueError(
            "Scene images are not pixel-aligned with VGGT predictions: "
            f"images have {images.shape[1:3]}, predictions have {(height, width)}"
        )
    return images.astype(np.float32)


def confidence_threshold_from_percentile(
    confidence: np.ndarray,
    world_points: np.ndarray,
    image_valid_masks: np.ndarray,
    *,
    percentile: float,
) -> float:
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("sampling.confidence_percentile must lie in [0, 100]")
    confidence_values = np.asarray(confidence, dtype=np.float32)
    points = np.asarray(world_points, dtype=np.float32)
    content_valid = np.asarray(image_valid_masks, dtype=bool)
    if confidence_values.shape != points.shape[:-1]:
        raise ValueError("confidence and world_points must share V/H/W dimensions")
    if content_valid.shape != confidence_values.shape:
        raise ValueError("processed_valid_mask must match confidence V/H/W dimensions")
    eligible = content_valid & np.isfinite(confidence_values) & np.isfinite(points).all(axis=-1)
    if not np.any(eligible):
        raise ValueError("VGGT predictions contain no finite, content-valid confidence values")
    return float(np.percentile(confidence_values[eligible], percentile))


def validate_vggt_precision_contract(predictions: dict[str, np.ndarray]) -> None:
    """Reject stale files produced by this runner with mixed-precision VGGT heads."""
    generated_by_runner = all(
        key in predictions for key in ("model_id", "world_points_source", "processed_images")
    )
    if not generated_by_runner:
        return
    contract = str(np.asarray(predictions.get("precision_contract", "")).item())
    if contract != "vggt_aggregator_amp_heads_float32_v1":
        raise RuntimeError(
            "This predictions.npz was generated by the old mixed-precision VGGT-head path. "
            "Delete it and rerun preprocess.run_vggt before building the initialization."
        )
    head_dtype = str(np.asarray(predictions.get("head_dtype", "")).item())
    if head_dtype != "float32":
        raise RuntimeError("VGGT prediction heads must use float32 outputs")


def _reject_removed_config(config: dict[str, Any]) -> None:
    sampling = config.get("sampling", {})
    removed_sampling = {"mode", "stride", "salient_fraction"}.intersection(sampling)
    if removed_sampling:
        names = ", ".join(sorted(removed_sampling))
        raise ValueError(f"Removed basic sampling options are not supported: {names}")
    if "confidence_threshold" in sampling:
        raise ValueError(
            "sampling.confidence_threshold used the wrong convention for VGGT scores; "
            "use confidence_percentile instead"
        )
    if "patch" in config:
        raise ValueError(
            "The fixed square patch configuration was removed; use covariance settings"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LoG/ellipse PCA Gaussian parameters.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--scene-root", default=None, help="Override scene root from config.")
    parser.add_argument("--predictions", default=None, help="Override predictions npz path.")
    parser.add_argument("--output", default=None, help="Override output torch file.")
    parser.add_argument(
        "--no-fusion", action="store_true", help="Disable fusion even if config enables it."
    )
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
        f"from {stats['detected_keypoints']} multi-scale keypoints."
    )
    print(f"Saved proposals: {stats['proposals_path']}")
    print(f"Saved final init: {stats['output_path']}")
    print(f"Saved debug PLY: {stats['debug_ply']}")


if __name__ == "__main__":
    main()
