from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .ellipses import compute_covariances
from .filters import PCAFilterConfig, valid_pca
from .fusion import FusionConfig, similarity_graph_fuse
from .gaussian_params import rgb_to_sh_dc, rotation_matrix_to_quaternion, sh_dc_to_rgb
from .grid_supplement import GridSupplementConfig, build_grid_supplement
from .io import (
    load_config,
    load_images,
    load_dense_predictions,
    resolve_scene_path,
    save_gaussians,
    write_ply,
)
from .pca import decompose_covariance, floor_normal_scale
from .regions import ellipse_regions, fit_regions
from .sampling import INITIALIZATION_METHOD, SamplingConfig, detect_multiscale_keypoints
from .types import GaussianProposals


def build_gaussian_initialization(
    config: dict[str, Any],
    *,
    scene_root_override: str | Path | None = None,
    predictions_override: str | Path | None = None,
    output_override: str | Path | None = None,
    force_no_fusion: bool = False,
) -> tuple[GaussianProposals, GaussianProposals, dict[str, Any]]:
    validate_initialization_config(config)
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

    predictions = load_dense_predictions(predictions_path)
    validate_prediction_precision_contract(predictions)
    world_points = predictions["world_points"]
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

    sampling = SamplingConfig.from_mapping(config.get("sampling"))
    grid_supplement = GridSupplementConfig.from_mapping(
        config.get("sampling", {}).get("grid_supplement")
    )
    covariance_cfg = config.get("covariance", {})
    pca_cfg = config.get("pca", {})
    gaussian_cfg = config.get("gaussian", {})
    fusion = FusionConfig.from_mapping(config.get("fusion"))
    eigenvalue_epsilon = float(pca_cfg.get("eigenvalue_epsilon", 1.0e-8))
    pca_filter = PCAFilterConfig.from_config(pca_cfg)
    normal_thickness_step_factor = float(
        pca_cfg.get("normal_thickness_step_factor", 0.25)
    )
    if (
        not np.isfinite(normal_thickness_step_factor)
        or normal_thickness_step_factor < 0.0
    ):
        raise ValueError("pca.normal_thickness_step_factor must be finite and non-negative")
    opacity = float(gaussian_cfg.get("opacity", 0.2))
    if not np.isfinite(opacity) or not 0.0 < opacity < 1.0:
        raise ValueError("gaussian.opacity must lie strictly between zero and one")
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    quats: list[np.ndarray] = []
    sh_dc: list[np.ndarray] = []
    opacities: list[float] = []
    view_ids: list[int] = []
    scores: list[float] = []

    stats: dict[str, Any] = {
        "views": views,
        "image_height": height,
        "image_width": width,
        "detected_keypoints": 0,
        "ellipse_regions": 0,
        "grid_candidate_cells": 0,
        "grid_merged_cells": 0,
        "grid_shape_rejected_merges": 0,
        "grid_regions": 0,
        "total_regions": 0,
        "ellipse_covered_pixels": 0,
        "grid_support_pixels": 0,
        "valid_image_pixels": 0,
        "accepted_proposals": 0,
        "accepted_ellipse_proposals": 0,
        "accepted_grid_proposals": 0,
        "rejected_covariance": 0,
        "rejected_covariance_ellipse": 0,
        "rejected_covariance_grid": 0,
        "rejected_pca": 0,
        "rejected_pca_ellipse": 0,
        "rejected_pca_grid": 0,
        "grid_normal_thickness_floored": 0,
        "normal_thickness_step_factor": normal_thickness_step_factor,
        "continuity_reference_scales": [],
    }

    for view in range(views):
        keypoints = detect_multiscale_keypoints(
            view_id=view,
            image=images[view],
            world_points=world_points[view],
            sigmas=sampling.sigmas,
            response_threshold=sampling.response_threshold,
            max_keypoints=sampling.max_keypoints_per_view,
            structure_sigma_factor=sampling.structure_sigma_factor,
            ellipse_radius_factor=sampling.ellipse_radius_factor,
            min_ellipse_area=sampling.min_ellipse_area,
            max_ellipse_area=sampling.max_ellipse_area,
            max_axis_ratio=sampling.max_axis_ratio,
            chroma_weight=sampling.chroma_weight,
            response_mad_epsilon=sampling.response_mad_epsilon,
            ellipse_merge_config=sampling.ellipse_merge,
            image_valid_mask=image_valid_masks[view],
        )
        stats["detected_keypoints"] += len(keypoints)
        view_ellipse_regions = ellipse_regions(keypoints)
        supplement = build_grid_supplement(
            images[view],
            world_points[view],
            view_ellipse_regions,
            image_valid_mask=image_valid_masks[view],
            ellipse_min_valid_points=int(covariance_cfg.get("min_valid_points", 16)),
            ellipse_min_valid_fraction=float(covariance_cfg.get("min_valid_fraction", 0.6)),
            continuity_neighbors=int(covariance_cfg.get("continuity_neighbors", 8)),
            config=grid_supplement,
        )
        stats["ellipse_regions"] += len(view_ellipse_regions)
        stats["grid_candidate_cells"] += supplement.candidate_cells
        stats["grid_merged_cells"] += supplement.merged_cells
        stats["grid_shape_rejected_merges"] += supplement.shape_rejected_merges
        stats["grid_regions"] += len(supplement.regions)
        stats["total_regions"] += len(view_ellipse_regions) + len(supplement.regions)
        stats["ellipse_covered_pixels"] += supplement.covered_pixels
        stats["grid_support_pixels"] += sum(
            region.support_count for region in supplement.regions
        )
        stats["valid_image_pixels"] += supplement.valid_pixels

        ellipse_results = compute_covariances(
            world_points[view],
            keypoints.us,
            keypoints.vs,
            keypoints.ellipse_matrices,
            image_valid_mask=image_valid_masks[view],
            min_valid_points=int(covariance_cfg.get("min_valid_points", 16)),
            min_valid_fraction=float(covariance_cfg.get("min_valid_fraction", 0.6)),
            continuity_neighbors=int(covariance_cfg.get("continuity_neighbors", 8)),
            continuity_ratio_max=float(covariance_cfg.get("continuity_ratio_max", 3.0)),
            device=str(covariance_cfg.get("device", "auto")),
            pixel_budget=int(covariance_cfg.get("pixel_budget", 2_000_000)),
        )
        stats["continuity_reference_scales"].append(
            ellipse_results.continuity_reference_scale
        )
        rejected_ellipse_covariance = int(np.count_nonzero(~ellipse_results.valid))
        stats["rejected_covariance"] += rejected_ellipse_covariance
        stats["rejected_covariance_ellipse"] += rejected_ellipse_covariance

        # Ellipse proposals use the keypoint
        # scene coordinate and RGB are the Gaussian center and SH color, and the
        # covariance is the second moment of its center-connected 3D component
        # around that fixed keypoint center.
        for index in np.flatnonzero(ellipse_results.valid):
            pca_result = decompose_covariance(
                ellipse_results.covariances[index],
                eigenvalue_epsilon=eigenvalue_epsilon,
            )
            if not valid_pca(pca_result, pca_filter):
                stats["rejected_pca"] += 1
                stats["rejected_pca_ellipse"] += 1
                continue

            u = int(keypoints.us[index])
            v = int(keypoints.vs[index])
            means.append(world_points[view, v, u].astype(np.float32))
            covariances.append(pca_result.covariance)
            scales.append(pca_result.scales)
            quats.append(rotation_matrix_to_quaternion(pca_result.basis))
            sh_dc.append(rgb_to_sh_dc(images[view, v, u]))
            opacities.append(opacity)
            view_ids.append(view)
            scores.append(float(keypoints.scores[index]))
            stats["accepted_ellipse_proposals"] += 1

        # Grid supplements have no keypoint. Fit their largest connected 3D
        # component around its own centroid, then append the resulting
        # proposals without changing any ellipse proposal above.
        grid_results = fit_regions(
            world_points[view],
            supplement.regions,
            colors=images[view],
            image_valid_mask=image_valid_masks[view],
            min_valid_points=int(covariance_cfg.get("min_valid_points", 16)),
            min_valid_fraction=float(covariance_cfg.get("min_valid_fraction", 0.6)),
            continuity_neighbors=int(covariance_cfg.get("continuity_neighbors", 8)),
            continuity_ratio_max=float(covariance_cfg.get("continuity_ratio_max", 3.0)),
            device=str(covariance_cfg.get("device", "auto")),
            pixel_budget=int(covariance_cfg.get("pixel_budget", 2_000_000)),
        )
        rejected_grid_covariance = int(np.count_nonzero(~grid_results.valid))
        stats["rejected_covariance"] += rejected_grid_covariance
        stats["rejected_covariance_grid"] += rejected_grid_covariance

        for index in np.flatnonzero(grid_results.valid):
            pca_result = decompose_covariance(
                grid_results.covariances[index],
                eigenvalue_epsilon=eigenvalue_epsilon,
            )
            if not valid_pca(pca_result, pca_filter):
                stats["rejected_pca"] += 1
                stats["rejected_pca_grid"] += 1
                continue

            minimum_normal_scale = pca_filter.scale_min
            if np.isfinite(grid_results.continuity_reference_scale):
                minimum_normal_scale = max(
                    minimum_normal_scale,
                    normal_thickness_step_factor
                    * float(grid_results.continuity_reference_scale),
                )
            raw_normal_scale = float(pca_result.scales[-1])
            pca_result = floor_normal_scale(
                pca_result,
                minimum_scale=minimum_normal_scale,
            )
            if float(pca_result.scales[-1]) > raw_normal_scale:
                stats["grid_normal_thickness_floored"] += 1

            region = supplement.regions[index]
            means.append(grid_results.means[index])
            covariances.append(pca_result.covariance)
            scales.append(pca_result.scales)
            quats.append(rotation_matrix_to_quaternion(pca_result.basis))
            sh_dc.append(rgb_to_sh_dc(grid_results.mean_colors[index]))
            opacities.append(opacity)
            view_ids.append(view)
            scores.append(region.score)
            stats["accepted_grid_proposals"] += 1

    proposals = GaussianProposals.from_lists(
        means=means,
        covariances=covariances,
        scales=scales,
        quats=quats,
        sh_dc=sh_dc,
        opacities=opacities,
        view_ids=view_ids,
        scores=scores,
    )
    stats["accepted_proposals"] = len(proposals)
    stats["ellipse_coverage_fraction"] = stats["ellipse_covered_pixels"] / max(
        stats["valid_image_pixels"], 1
    )
    stats["grid_supplement_enabled"] = grid_supplement.enabled
    if len(proposals) == 0:
        raise RuntimeError(
            "No Gaussian proposals survived region coverage, 3D connectivity, and PCA filtering. "
            "Check image/prediction alignment, finite 3D coverage, LoG, and PCA thresholds. "
            f"Stats: {stats}"
        )
    save_gaussians(
        proposals_path,
        proposals,
        metadata={
            "stage": "per_view_gaussian_proposals",
            "initialization_contract": INITIALIZATION_METHOD,
            "source_predictions": str(predictions_path),
            "stats": dict(stats),
        },
    )

    fusion_enabled = fusion.enabled and not force_no_fusion
    if fusion_enabled:
        fusion_result = similarity_graph_fuse(
            proposals,
            config=fusion,
            eigenvalue_epsilon=eigenvalue_epsilon,
            pca_filter=pca_filter,
        )
        fused = fusion_result.gaussians
        fusion_stats = fusion_result.stats
        stats.update(
            {
                "fusion_candidate_pairs": fusion_stats.candidate_pairs,
                "fusion_compatible_pairs": fusion_stats.compatible_pairs,
                "fusion_pairs_failing_overlap": fusion_stats.pairs_failing_overlap,
                "fusion_pairs_failing_normal": fusion_stats.pairs_failing_normal,
                "fusion_pairs_failing_scale": fusion_stats.pairs_failing_scale,
                "fusion_pairs_failing_color": fusion_stats.pairs_failing_color,
                "fusion_components": fusion_stats.components,
                "fusion_singleton_components": fusion_stats.singleton_components,
                "fusion_merged_components": fusion_stats.merged_components,
                "fusion_fallback_components": fusion_stats.fallback_components,
            }
        )
    else:
        fused = proposals
        stats.update(
            {
                "fusion_candidate_pairs": 0,
                "fusion_compatible_pairs": 0,
                "fusion_pairs_failing_overlap": 0,
                "fusion_pairs_failing_normal": 0,
                "fusion_pairs_failing_scale": 0,
                "fusion_pairs_failing_color": 0,
                "fusion_components": len(proposals),
                "fusion_singleton_components": len(proposals),
                "fusion_merged_components": 0,
                "fusion_fallback_components": 0,
            }
        )

    stats["fused_gaussians"] = len(fused)
    stats["fusion_enabled"] = fusion_enabled
    save_gaussians(
        output_path,
        fused,
        metadata={
            "stage": "fused_gaussian_initialization",
            "initialization_contract": INITIALIZATION_METHOD,
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
            raise ValueError("processed_images are not aligned with dense predictions")
        return images

    image_root = resolve_scene_path(scene_root, images_dir)
    images = load_images(image_root)

    if images.shape[0] != views:
        raise ValueError(
            f"Image count ({images.shape[0]}) does not match prediction views ({views})"
        )
    if images.shape[1:3] != (height, width):
        raise ValueError(
            "Scene images are not pixel-aligned with dense predictions: "
            f"images have {images.shape[1:3]}, predictions have {(height, width)}"
        )
    return images.astype(np.float32)


def validate_prediction_precision_contract(predictions: dict[str, np.ndarray]) -> None:
    """Validate the precision metadata emitted by this repository's VGGT runner."""
    generated_by_runner = all(
        key in predictions for key in ("model_id", "world_points_source", "processed_images")
    )
    if not generated_by_runner:
        return
    contract = str(np.asarray(predictions.get("precision_contract", "")).item())
    if contract != "vggt_aggregator_amp_heads_float32_v1":
        raise RuntimeError(
            "The predictions.npz precision contract is incompatible; rerun "
            "preprocess.run_vggt before building the initialization."
        )
    head_dtype = str(np.asarray(predictions.get("head_dtype", "")).item())
    if head_dtype != "float32":
        raise RuntimeError("VGGT prediction heads must use float32 outputs")


_INITIALIZATION_SECTION_KEYS = {
    "scene": {"root", "images_dir", "predictions_path", "proposals_path", "output_path"},
    "sampling": {
        "sigmas",
        "response_threshold",
        "max_keypoints_per_view",
        "structure_sigma_factor",
        "ellipse_radius_factor",
        "min_ellipse_area",
        "max_ellipse_area",
        "max_axis_ratio",
        "chroma_weight",
        "response_mad_epsilon",
        "ellipse_merge",
        "grid_supplement",
    },
    "covariance": {
        "min_valid_points",
        "min_valid_fraction",
        "continuity_neighbors",
        "continuity_ratio_max",
        "device",
        "pixel_budget",
    },
    "pca": {
        "eigenvalue_epsilon",
        "scale_min",
        "scale_max",
        "min_secondary_eigenvalue_ratio",
        "normal_thickness_step_factor",
    },
    "gaussian": {"opacity"},
    "fusion": {
        "enabled",
        "voxel_size",
        "overlap_mahalanobis_max",
        "covariance_regularization_factor",
        "normal_angle_max_degrees",
        "scale_ratio_max",
        "color_delta_e_max",
    },
}


def validate_initialization_config(config: dict[str, Any]) -> None:
    """Reject unknown keys in initialization-owned configuration sections."""
    allowed_sections = set(_INITIALIZATION_SECTION_KEYS) | {"_config_path"}
    unknown_sections = sorted(set(config) - allowed_sections)
    if unknown_sections:
        names = ", ".join(unknown_sections)
        raise ValueError(f"Unknown initialization config section(s): {names}")
    for section, allowed_keys in _INITIALIZATION_SECTION_KEYS.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
        unknown = sorted(set(values) - allowed_keys)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"Unknown {section} config key(s): {names}")
    ellipse_merge = config.get("sampling", {}).get("ellipse_merge", {})
    if not isinstance(ellipse_merge, dict):
        raise ValueError("Config section 'sampling.ellipse_merge' must be a mapping")
    allowed_merge_keys = {
        "iou_min",
        "orientation_max_degrees",
        "isotropic_axis_ratio",
        "color_delta_e_max",
        "continuity_ratio_max",
        "grid_cell_factor",
        "merged_area_factor_max",
        "merged_area_absolute_max",
    }
    unknown_merge = sorted(set(ellipse_merge) - allowed_merge_keys)
    if unknown_merge:
        names = ", ".join(unknown_merge)
        raise ValueError(f"Unknown sampling.ellipse_merge config key(s): {names}")
    grid_supplement = config.get("sampling", {}).get("grid_supplement", {})
    if not isinstance(grid_supplement, dict):
        raise ValueError("Config section 'sampling.grid_supplement' must be a mapping")
    allowed_grid_keys = {
        "enabled",
        "cell_size",
        "min_valid_fraction",
        "color_delta_e_max",
        "boundary_continuity_fraction_min",
        "continuity_ratio_max",
        "max_component_pixels",
        "convex_fill_ratio_min",
        "require_hole_free",
    }
    unknown_grid = sorted(set(grid_supplement) - allowed_grid_keys)
    if unknown_grid:
        names = ", ".join(unknown_grid)
        raise ValueError(f"Unknown sampling.grid_supplement config key(s): {names}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Lab LoG/ellipse plus uncovered-grid PCA Gaussian parameters."
    )
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
        f"from {stats['ellipse_regions']} ellipse and {stats['grid_regions']} grid regions."
    )
    print(f"Saved proposals: {stats['proposals_path']}")
    print(f"Saved final init: {stats['output_path']}")
    print(f"Saved debug PLY: {stats['debug_ply']}")


if __name__ == "__main__":
    main()
