from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gsplat_train.dataset import estimate_scene_scale
from preprocess.run_colmap import load_aligned_frames, reconstruction_stats

from .gaussian_params import rgb_to_sh_dc
from .io import resolve_scene_path, save_gaussians, write_ply
from .knn import compute_knn_isotropic_scales
from .types import GaussianProposals


def build_colmap_initialization(
    *,
    predictions_path: str | Path,
    model_path: str | Path,
    scene_output_path: str | Path,
    gaussian_output_path: str | Path,
    opacity: float = 0.2,
    device: str = "auto",
    knn_chunk_size: int = 512,
    scale_min: float = 1.0e-5,
    require_all_views: bool = True,
) -> tuple[GaussianProposals, dict[str, Any]]:
    if not np.isfinite(opacity) or not 0.0 < opacity < 1.0:
        raise ValueError("opacity must lie strictly between zero and one")
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("Install PyCOLMAP with: uv sync --extra colmap") from exc

    predictions_path = Path(predictions_path)
    model_path = Path(model_path)
    scene_output_path = Path(scene_output_path)
    gaussian_output_path = Path(gaussian_output_path)
    images, valid_mask = load_aligned_frames(predictions_path)
    expected_names = [f"view_{view_id:03d}.png" for view_id in range(images.shape[0])]
    reconstruction = pycolmap.Reconstruction(model_path)
    extracted = extract_reconstruction_arrays(
        reconstruction,
        expected_image_names=expected_names,
        expected_shape=images.shape[1:3],
        require_all_views=require_all_views,
    )

    target_c2w, source_image_names = load_vggt_camera_gauge(predictions_path, images.shape[0])
    normalized_points, normalized_c2w, normalization = normalize_to_camera_gauge(
        extracted["points"],
        extracted["extrinsics_c2w"],
        target_c2w,
    )
    normalized_w2c = np.linalg.inv(normalized_c2w.astype(np.float64)).astype(np.float32)

    scales_1d = compute_knn_isotropic_scales(
        normalized_points,
        neighbors=3,
        device=device,
        chunk_size=knn_chunk_size,
        minimum_scale=scale_min,
    )
    count = normalized_points.shape[0]
    scales = np.repeat(scales_1d[:, None], 3, axis=1).astype(np.float32)
    covariances = np.zeros((count, 3, 3), dtype=np.float32)
    covariances[:, np.arange(3), np.arange(3)] = scales**2
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    colors = extracted["colors"]
    gaussians = GaussianProposals(
        means=normalized_points,
        covariances=covariances,
        scales=scales,
        quats=quats,
        sh_dc=rgb_to_sh_dc(colors),
        opacities=np.full(count, opacity, dtype=np.float32),
        view_ids=np.full(count, -1, dtype=np.int64),
        scores=np.ones(count, dtype=np.float32),
    )

    colmap_stats = reconstruction_stats(reconstruction)
    reprojection_error = float(colmap_stats["mean_reprojection_error_px"])
    scene_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scene_output_path,
        processed_images=images.astype(np.float32),
        processed_valid_mask=valid_mask.astype(bool),
        intrinsics=extracted["intrinsics"].astype(np.float32),
        extrinsics=normalized_c2w,
        extrinsics_c2w=normalized_c2w,
        extrinsics_w2c=normalized_w2c,
        image_names=np.asarray(expected_names),
        source_image_names=np.asarray(source_image_names),
        camera_source=np.asarray("colmap_incremental_sfm"),
        camera_model=np.asarray("PINHOLE"),
        reprojection_error_px=np.asarray(reprojection_error, dtype=np.float32),
        mean_reprojection_error_px=np.asarray(reprojection_error, dtype=np.float32),
    )

    stats: dict[str, Any] = {
        "method": "colmap_sparse_3nn_isotropic",
        "source_vggt_alignment_archive": str(predictions_path.resolve()),
        "source_colmap_model": str(model_path.resolve()),
        "scene_archive": str(scene_output_path.resolve()),
        "output_path": str(gaussian_output_path.resolve()),
        "views": int(images.shape[0]),
        "registered_images": int(extracted["registered_images"]),
        "sparse_points": int(count),
        "camera_model": "PINHOLE",
        "undistortion": False,
        "opacity": float(opacity),
        "knn_neighbors": 3,
        "scale_formula": "sqrt(mean(d_1^2, d_2^2, d_3^2))",
        "scale_minimum_floor": float(scale_min),
        "scale_floor_count": int(np.count_nonzero(scales_1d <= scale_min)),
        "scale_min": float(scales_1d.min()),
        "scale_median": float(np.median(scales_1d)),
        "scale_p95": float(np.percentile(scales_1d, 95.0)),
        "scale_max": float(scales_1d.max()),
        "mean_reprojection_error_px": reprojection_error,
        "reprojection_error_p95_px": float(colmap_stats["reprojection_error_p95_px"]),
        "mean_track_length": float(colmap_stats["mean_track_length"]),
        "normalization": normalization,
    }
    debug_ply = gaussian_output_path.with_name(f"{gaussian_output_path.stem}_debug.ply")
    stats["debug_ply"] = str(debug_ply.resolve())
    save_gaussians(
        gaussian_output_path,
        gaussians,
        metadata={
            "stage": "colmap_sparse_initialization",
            "initialization_contract": "activated_scales_opacities_wxyz_sh_dc",
            "stats": stats,
        },
    )
    write_ply(debug_ply, normalized_points, colors)
    return gaussians, stats


def extract_reconstruction_arrays(
    reconstruction,
    *,
    expected_image_names: list[str],
    expected_shape: tuple[int, int],
    require_all_views: bool,
) -> dict[str, np.ndarray | int]:
    images_by_name = {
        image.name: image
        for image in reconstruction.images.values()
        if bool(image.has_pose)
    }
    missing = [name for name in expected_image_names if name not in images_by_name]
    if require_all_views and missing:
        raise RuntimeError(
            f"COLMAP is missing {len(missing)} aligned views: {', '.join(missing[:8])}"
        )
    selected_names = [name for name in expected_image_names if name in images_by_name]
    if len(selected_names) != len(expected_image_names):
        raise RuntimeError(
            "Partial reconstructions cannot produce the fixed all-view benchmark scene archive"
        )

    intrinsics = np.empty((len(selected_names), 3, 3), dtype=np.float32)
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], len(selected_names), axis=0)
    height, width = (int(expected_shape[0]), int(expected_shape[1]))
    for view_id, name in enumerate(selected_names):
        image = images_by_name[name]
        camera = reconstruction.cameras[image.camera_id]
        if camera.model_name != "PINHOLE":
            raise RuntimeError(f"Expected PINHOLE camera for {name}, got {camera.model_name}")
        if (int(camera.height), int(camera.width)) != (height, width):
            raise RuntimeError(f"Camera dimensions for {name} do not match aligned images")
        intrinsics[view_id] = np.asarray(camera.calibration_matrix(), dtype=np.float32)
        w2c[view_id, :3, :4] = np.asarray(image.cam_from_world().matrix(), dtype=np.float32)

    c2w = np.linalg.inv(w2c.astype(np.float64)).astype(np.float32)
    point_items = sorted(reconstruction.points3D.items())
    if len(point_items) < 4:
        raise RuntimeError("COLMAP reconstruction needs at least four sparse points")
    points = np.asarray([point.xyz for _, point in point_items], dtype=np.float32)
    colors = np.asarray([point.color for _, point in point_items], dtype=np.float32) / 255.0
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("COLMAP sparse points or colors contain non-finite values")
    return {
        "intrinsics": intrinsics,
        "extrinsics_c2w": c2w,
        "points": points,
        "colors": colors,
        "registered_images": len(selected_names),
    }


def load_vggt_camera_gauge(
    predictions_path: Path,
    expected_views: int,
) -> tuple[np.ndarray, list[str]]:
    with np.load(predictions_path) as archive:
        if "extrinsics_c2w" in archive.files:
            target_c2w = np.asarray(archive["extrinsics_c2w"], dtype=np.float32)
        elif "extrinsics" in archive.files:
            target_c2w = np.asarray(archive["extrinsics"], dtype=np.float32)
        elif "extrinsics_w2c" in archive.files:
            target_c2w = np.linalg.inv(
                np.asarray(archive["extrinsics_w2c"], dtype=np.float64)
            ).astype(np.float32)
        else:
            raise ValueError("VGGT archive needs camera extrinsics for gauge normalization")
        if "image_paths" in archive.files:
            source_names = [Path(str(value)).name for value in archive["image_paths"].tolist()]
        else:
            source_names = [f"view_{view_id:03d}" for view_id in range(expected_views)]
    if target_c2w.shape != (expected_views, 4, 4):
        raise ValueError("VGGT camera count does not match processed_images")
    if len(source_names) != expected_views:
        raise ValueError("VGGT image path count does not match processed_images")
    return target_c2w, source_names


def normalize_to_camera_gauge(
    points: np.ndarray,
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_centers = np.asarray(source_c2w, dtype=np.float64)[:, :3, 3]
    target_centers = np.asarray(target_c2w, dtype=np.float64)[:, :3, 3]
    source_center = np.median(source_centers, axis=0)
    target_center = np.median(target_centers, axis=0)
    source_radius = estimate_scene_scale(np.asarray(source_c2w, dtype=np.float32))
    target_radius = estimate_scene_scale(np.asarray(target_c2w, dtype=np.float32))
    if not np.isfinite(source_radius) or source_radius <= 0.0:
        raise ValueError("COLMAP camera gauge has a non-positive radius")
    scale = float(target_radius / source_radius)

    normalized_points = (
        (np.asarray(points, dtype=np.float64) - source_center) * scale + target_center
    ).astype(np.float32)
    normalized_c2w = np.asarray(source_c2w, dtype=np.float32).copy()
    normalized_c2w[:, :3, 3] = (
        (source_centers - source_center) * scale + target_center
    ).astype(np.float32)
    return normalized_points, normalized_c2w, {
        "type": "camera_center_translation_and_uniform_scale",
        "source_camera_center": source_center.tolist(),
        "target_camera_center": target_center.tolist(),
        "source_camera_radius": float(source_radius),
        "target_camera_radius": float(target_radius),
        "uniform_scale": scale,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an aligned COLMAP sparse model to scene and Gaussian archives."
    )
    parser.add_argument("--scene-root", type=Path, default=Path("data/tnt_truck_48"))
    parser.add_argument("--predictions", default="vggt/predictions.npz")
    parser.add_argument("--model", default="colmap/sparse/0")
    parser.add_argument("--scene-output", default="colmap/scene.npz")
    parser.add_argument("--output", default="init/colmap_sparse_gaussians.pt")
    parser.add_argument("--opacity", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    parser.add_argument("--scale-min", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.expanduser().resolve()
    gaussians, stats = build_colmap_initialization(
        predictions_path=resolve_scene_path(scene_root, args.predictions),
        model_path=resolve_scene_path(scene_root, args.model),
        scene_output_path=resolve_scene_path(scene_root, args.scene_output),
        gaussian_output_path=resolve_scene_path(scene_root, args.output),
        opacity=args.opacity,
        device=args.device,
        knn_chunk_size=args.knn_chunk_size,
        scale_min=args.scale_min,
    )
    print(json.dumps(stats, indent=2))
    print(f"Built {len(gaussians):,} COLMAP sparse Gaussians")


if __name__ == "__main__":
    main()
