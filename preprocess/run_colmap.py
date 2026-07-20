from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from init.io import resolve_scene_path


def load_aligned_frames(predictions_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"VGGT predictions not found: {path}")
    with np.load(path) as archive:
        if "processed_images" not in archive.files:
            raise ValueError(
                "Aligned COLMAP reconstruction requires processed_images in the VGGT archive"
            )
        images = np.asarray(archive["processed_images"], dtype=np.float32)
        valid_mask = np.asarray(
            archive["processed_valid_mask"]
            if "processed_valid_mask" in archive.files
            else np.ones(images.shape[:3], dtype=bool),
            dtype=bool,
        )
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("processed_images must have shape [V, H, W, 3]")
    if valid_mask.shape != images.shape[:3]:
        raise ValueError("processed_valid_mask must match processed_images")
    if not np.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
        raise ValueError("processed_images must contain finite RGB values in [0, 1]")
    return images, valid_mask


def export_aligned_images(
    predictions_path: str | Path,
    *,
    images_dir: str | Path,
    masks_dir: str | Path,
) -> dict[str, Any]:
    images, valid_mask = load_aligned_frames(predictions_path)
    image_root = Path(images_dir)
    mask_root = Path(masks_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    expected_names = [f"view_{view_id:03d}.png" for view_id in range(images.shape[0])]
    remove_stale_files(image_root, expected_names)
    remove_stale_files(mask_root, [f"{name}.png" for name in expected_names])

    quantized = np.clip(np.rint(images * 255.0), 0.0, 255.0).astype(np.uint8)
    for view_id, name in enumerate(expected_names):
        Image.fromarray(quantized[view_id], mode="RGB").save(image_root / name)
        Image.fromarray(valid_mask[view_id].astype(np.uint8) * 255, mode="L").save(
            mask_root / f"{name}.png"
        )

    reconstructed = quantized.astype(np.float32) / 255.0
    fingerprint = hashlib.sha256()
    fingerprint.update(np.asarray(images.shape, dtype=np.int64).tobytes())
    fingerprint.update(quantized.tobytes())
    fingerprint.update(valid_mask.tobytes())
    return {
        "image_names": expected_names,
        "views": int(images.shape[0]),
        "height": int(images.shape[1]),
        "width": int(images.shape[2]),
        "uses_masks": bool(np.any(~valid_mask)),
        "max_png_quantization_error": float(np.max(np.abs(reconstructed - images))),
        "aligned_content_sha256": fingerprint.hexdigest(),
    }


def remove_stale_files(root: Path, expected_names: list[str]) -> None:
    expected = set(expected_names)
    for path in root.iterdir():
        if path.is_file() and path.name not in expected:
            path.unlink()


def run_sparse_reconstruction(
    *,
    predictions_path: str | Path,
    output_root: str | Path,
    device: str = "cuda",
    gpu_index: str = "0",
    camera_model: str = "PINHOLE",
    max_num_features: int = 8192,
    seed: int = 42,
    require_all_views: bool = True,
    restart: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if camera_model != "PINHOLE":
        raise ValueError("The aligned benchmark currently requires camera_model=PINHOLE")
    if max_num_features <= 0:
        raise ValueError("max_num_features must be positive")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")

    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("Install PyCOLMAP with: uv sync --extra colmap") from exc

    root = Path(output_root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    database_path = root / "database.db"
    candidate_models_dir = root / "models"
    selected_model_dir = root / "sparse" / "0"
    manifest_path = root / "reconstruction.json"

    root.mkdir(parents=True, exist_ok=True)
    if restart:
        for path in (database_path, candidate_models_dir, root / "sparse", manifest_path):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    export_stats = export_aligned_images(
        predictions_path,
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
    expected_views = int(export_stats["views"])

    if selected_model_dir.exists():
        if not manifest_path.exists():
            raise RuntimeError(
                f"Existing COLMAP model has no compatibility manifest: {selected_model_dir}. "
                "Rerun with --restart."
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("aligned_content_sha256") != export_stats["aligned_content_sha256"]:
            raise RuntimeError(
                "Aligned benchmark images changed after COLMAP reconstruction. "
                "Rerun with --restart."
            )
        reconstruction = pycolmap.Reconstruction(selected_model_dir)
        stats = reconstruction_stats(reconstruction)
        validate_reconstruction(
            reconstruction,
            expected_views=expected_views,
            expected_shape=(int(export_stats["height"]), int(export_stats["width"])),
            require_all_views=require_all_views,
        )
        stats.update(export_stats)
        stats.update(
            {
                "status": "reused",
                "camera_model": "PINHOLE",
                "predictions_path": str(Path(predictions_path).resolve()),
                "database_path": str(database_path.resolve()),
                "model_path": str(selected_model_dir.resolve()),
                "undistortion": False,
            }
        )
        return selected_model_dir, stats

    if database_path.exists() or candidate_models_dir.exists():
        raise RuntimeError(
            f"Incomplete COLMAP workspace exists in {root}. Rerun with --restart to rebuild it."
        )

    candidate_models_dir.mkdir(parents=True)
    started = time.perf_counter()
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.use_gpu = device != "cpu"
    extraction_options.gpu_index = str(gpu_index)
    extraction_options.sift.max_num_features = int(max_num_features)
    # The aligned 518px frames deliberately contain less detail than the raw
    # source images. Keep the same pixels, but admit weaker repeatable SIFT
    # extrema so low-texture views are not silently lost by the SfM front-end.
    extraction_options.sift.peak_threshold = 0.003
    reader_options = pycolmap.ImageReaderOptions(camera_model=camera_model)
    if bool(export_stats["uses_masks"]):
        reader_options.mask_path = masks_dir

    print(f"Extracting COLMAP features from {expected_views} aligned images", flush=True)
    pycolmap.extract_features(
        database_path,
        images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=resolve_pycolmap_device(pycolmap, device),
    )

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = device != "cpu"
    matching_options.gpu_index = str(gpu_index)
    matching_options.guided_matching = True
    matching_options.max_num_matches = max(32768, int(max_num_features) * 4)
    matching_options.sift.max_ratio = 0.9
    matching_options.sift.max_distance = 0.8
    print("Running exhaustive COLMAP matching", flush=True)
    pycolmap.match_exhaustive(
        database_path,
        matching_options=matching_options,
        device=resolve_pycolmap_device(pycolmap, device),
    )

    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.random_seed = int(seed)
    mapping_options.mapper.random_seed = int(seed)
    mapping_options.triangulation.random_seed = int(seed)
    mapping_options.extract_colors = True
    mapping_options.multiple_models = True
    mapping_options.ba_global_function_tolerance = 1.0e-6
    mapping_options.min_num_matches = 10
    mapping_options.mapper.init_min_num_inliers = 50
    mapping_options.mapper.init_min_tri_angle = 8.0
    mapping_options.mapper.abs_pose_min_num_inliers = 15
    mapping_options.mapper.abs_pose_min_inlier_ratio = 0.15
    mapping_options.mapper.max_reg_trials = 10
    mapping_options.triangulation.ignore_two_view_tracks = False
    print("Running incremental COLMAP mapping", flush=True)
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        images_dir,
        candidate_models_dir,
        options=mapping_options,
    )
    if not reconstructions:
        raise RuntimeError("COLMAP incremental mapping produced no reconstruction")
    _, reconstruction = max(
        reconstructions.items(),
        key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()),
    )
    validate_reconstruction(
        reconstruction,
        expected_views=expected_views,
        expected_shape=(int(export_stats["height"]), int(export_stats["width"])),
        require_all_views=require_all_views,
    )
    selected_model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(selected_model_dir)

    stats = reconstruction_stats(reconstruction)
    stats.update(export_stats)
    stats.update(
        {
            "status": "built",
            "camera_model": camera_model,
            "device": device,
            "gpu_index": str(gpu_index),
            "max_num_features": int(max_num_features),
            "seed": int(seed),
            "elapsed_seconds": float(time.perf_counter() - started),
            "predictions_path": str(Path(predictions_path).resolve()),
            "database_path": str(database_path.resolve()),
            "model_path": str(selected_model_dir.resolve()),
            "undistortion": False,
            "sift_peak_threshold": 0.003,
            "guided_matching": True,
            "sift_matching_max_ratio": 0.9,
            "sift_matching_max_distance": 0.8,
        }
    )
    manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return selected_model_dir, stats


def resolve_pycolmap_device(pycolmap, value: str):
    if value == "cuda":
        if not pycolmap.has_cuda:
            raise RuntimeError("CUDA was requested but the installed PyCOLMAP has no CUDA support")
        return pycolmap.Device.cuda
    if value == "cpu":
        return pycolmap.Device.cpu
    return pycolmap.Device.auto


def reconstruction_stats(reconstruction) -> dict[str, Any]:
    point_errors = np.asarray(
        [float(point.error) for point in reconstruction.points3D.values() if point.has_error()],
        dtype=np.float64,
    )
    return {
        "registered_images": int(reconstruction.num_reg_images()),
        "sparse_points": int(reconstruction.num_points3D()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
        "mean_observations_per_image": float(
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "mean_reprojection_error_px": float(reconstruction.compute_mean_reprojection_error()),
        "reprojection_error_p95_px": (
            float(np.percentile(point_errors, 95.0)) if point_errors.size else float("nan")
        ),
    }


def validate_reconstruction(
    reconstruction,
    *,
    expected_views: int,
    expected_shape: tuple[int, int],
    require_all_views: bool,
) -> None:
    registered = int(reconstruction.num_reg_images())
    if require_all_views and registered != expected_views:
        raise RuntimeError(
            f"COLMAP registered {registered}/{expected_views} aligned views; the controlled "
            "benchmark requires every view"
        )
    if registered < 2 or reconstruction.num_points3D() < 4:
        raise RuntimeError("COLMAP reconstruction is too small for Gaussian initialization")
    height, width = expected_shape
    for camera in reconstruction.cameras.values():
        if camera.model_name != "PINHOLE":
            raise RuntimeError(f"Expected PINHOLE camera, got {camera.model_name}")
        if (int(camera.height), int(camera.width)) != (height, width):
            raise RuntimeError(
                "COLMAP camera dimensions do not match aligned images: "
                f"{camera.height}x{camera.width} vs {height}x{width}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run aligned PINHOLE PyCOLMAP sparse reconstruction without undistortion."
    )
    parser.add_argument("--scene-root", type=Path, default=Path("data/tnt_truck_48"))
    parser.add_argument("--predictions", default="vggt/predictions.npz")
    parser.add_argument("--output-root", default="colmap")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--max-num-features", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.expanduser().resolve()
    predictions_path = resolve_scene_path(scene_root, args.predictions)
    output_root = resolve_scene_path(scene_root, args.output_root)
    model_path, stats = run_sparse_reconstruction(
        predictions_path=predictions_path,
        output_root=output_root,
        device=args.device,
        gpu_index=args.gpu_index,
        max_num_features=args.max_num_features,
        seed=args.seed,
        require_all_views=not args.allow_partial,
        restart=args.restart,
    )
    print(json.dumps(stats, indent=2))
    print(f"Selected sparse model: {model_path}")


if __name__ == "__main__":
    main()
