from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from init.io import IMAGE_EXTENSIONS, load_images


def list_image_paths(images_dir: str | Path) -> list[Path]:
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found in: {root}")
    return paths


def create_mock_plane_predictions(images_dir: str | Path, output: str | Path) -> None:
    images = load_images(images_dir)
    views, height, width, _ = images.shape

    focal = float(max(height, width))
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    depth = np.ones((views, height, width), dtype=np.float32)
    confidence = np.ones((views, height, width), dtype=np.float32)
    world_points = np.empty((views, height, width, 3), dtype=np.float32)
    intrinsics = np.empty((views, 3, 3), dtype=np.float32)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], views, axis=0)

    for view in range(views):
        intrinsics[view] = np.asarray(
            [
                [focal, 0.0, cx],
                [0.0, focal, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        world_points[view, ..., 0] = (xs - cx) / focal + 0.02 * view
        world_points[view, ..., 1] = (ys - cy) / focal
        world_points[view, ..., 2] = 1.0
        extrinsics[view, 0, 3] = 0.02 * view

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        depth=depth,
        confidence=confidence,
        world_points=world_points,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )
    print(f"Saved mock VGGT predictions: {output_path}")


def run_real_vggt(
    images_dir: str | Path,
    output: str | Path,
    *,
    model_id: str,
    device_name: str,
    preprocess_mode: str,
    max_resolution: int,
    use_point_map: bool,
) -> None:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_paths = list_image_paths(images_dir)
    device = resolve_device(device_name)
    dtype = resolve_inference_dtype(device)

    print(f"Loading VGGT model: {model_id}")
    model = VGGT.from_pretrained(model_id).to(device)
    model.eval()

    print(f"Loading {len(image_paths)} images from {images_dir}")
    images = load_and_preprocess_images([str(path) for path in image_paths], mode=preprocess_mode)
    images = resize_for_memory(images, max_resolution=max_resolution)
    images = images.to(device)
    print(f"Preprocessed tensor shape: {tuple(images.shape)}")

    print(f"Running VGGT on {device} with dtype {dtype}")
    autocast_context = (
        torch.cuda.amp.autocast(dtype=dtype) if device.type == "cuda" else nullcontext()
    )
    with torch.inference_mode():
        with autocast_context:
            images_batched = images[None]
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_batched)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            depth_tensor, depth_conf_tensor = model.depth_head(
                aggregated_tokens_list,
                images_batched,
                patch_start_idx,
            )
            if use_point_map:
                point_map_tensor, point_conf_tensor = model.point_head(
                    aggregated_tokens_list,
                    images_batched,
                    patch_start_idx,
                )
            else:
                point_map_tensor = None
                point_conf_tensor = None

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        pose_enc,
        images.shape[-2:],
    )

    depth = to_numpy(depth_tensor).squeeze(0)
    depth_conf = to_numpy(depth_conf_tensor).squeeze(0)
    point_map = to_numpy(point_map_tensor).squeeze(0) if point_map_tensor is not None else None
    point_conf = to_numpy(point_conf_tensor).squeeze(0) if point_conf_tensor is not None else None
    extrinsic_w2c = to_numpy(extrinsic).squeeze(0)
    intrinsic_np = to_numpy(intrinsic).squeeze(0)

    world_points_from_depth = unproject_depth_map_to_point_map(
        depth,
        extrinsic_w2c,
        intrinsic_np,
    ).astype(np.float32)

    if use_point_map:
        if point_map is None or point_conf is None:
            raise RuntimeError("Point map was requested but VGGT did not produce it")
        world_points = point_map.astype(np.float32)
        confidence = point_conf.astype(np.float32)
        world_points_source = "point_map"
    else:
        world_points = world_points_from_depth
        confidence = depth_conf.astype(np.float32)
        world_points_source = "depth_unprojection"

    extrinsics_w2c_4x4 = extrinsics_3x4_to_4x4(extrinsic_w2c)
    extrinsics_c2w_4x4 = np.linalg.inv(extrinsics_w2c_4x4).astype(np.float32)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "depth": depth.squeeze(-1).astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "world_points": world_points.astype(np.float32),
        "world_points_from_depth": world_points_from_depth.astype(np.float32),
        "depth_confidence": depth_conf.astype(np.float32),
        "intrinsics": intrinsic_np.astype(np.float32),
        "extrinsics": extrinsics_c2w_4x4,
        "extrinsics_c2w": extrinsics_c2w_4x4,
        "extrinsics_w2c": extrinsics_w2c_4x4,
        "image_paths": np.asarray([str(path) for path in image_paths]),
        "preprocess_mode": np.asarray(preprocess_mode),
        "model_id": np.asarray(model_id),
        "world_points_source": np.asarray(world_points_source),
    }
    if point_map is not None and point_conf is not None:
        payload["point_map"] = point_map.astype(np.float32)
        payload["point_confidence"] = point_conf.astype(np.float32)
    np.savez_compressed(output_path, **payload)
    print(f"Saved VGGT predictions: {output_path}")
    print(f"world_points source: {world_points_source}")
    print(f"depth shape: {tuple(depth.squeeze(-1).shape)}")
    print(f"world_points shape: {tuple(world_points.shape)}")


def resolve_device(device_name: str):
    import torch

    requested = device_name.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def resolve_inference_dtype(device):
    import torch

    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().numpy()
    return np.asarray(tensor)


def resize_for_memory(images, *, max_resolution: int):
    import torch.nn.functional as F

    if max_resolution <= 0:
        raise ValueError("--max-resolution must be positive")

    _, _, height, width = images.shape
    current_max = max(height, width)
    if current_max <= max_resolution:
        return images

    scale = max_resolution / current_max
    new_height = max(14, round(height * scale / 14) * 14)
    new_width = max(14, round(width * scale / 14) * 14)
    return F.interpolate(images, size=(new_height, new_width), mode="bilinear", align_corners=False)


def extrinsics_3x4_to_4x4(extrinsics: np.ndarray) -> np.ndarray:
    matrices = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], extrinsics.shape[0], axis=0)
    matrices[:, :3, :4] = extrinsics.astype(np.float32)
    return matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT preprocessing.")
    parser.add_argument("--images", required=True, help="Input image directory.")
    parser.add_argument("--output", required=True, help="Output predictions.npz path.")
    parser.add_argument("--model-id", default="facebook/VGGT-1B", help="Hugging Face model id.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device.",
    )
    parser.add_argument(
        "--preprocess-mode",
        default="crop",
        choices=("crop", "pad"),
        help="VGGT image preprocessing mode.",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=518,
        help="Maximum preprocessed image side. Lower this for small GPUs.",
    )
    parser.add_argument(
        "--use-point-map",
        action="store_true",
        help="Use VGGT point-head world points instead of depth unprojection.",
    )
    parser.add_argument(
        "--mock-plane",
        action="store_true",
        help="Create deterministic plane geometry for smoke tests instead of running VGGT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock_plane:
        create_mock_plane_predictions(args.images, args.output)
        return

    run_real_vggt(
        args.images,
        args.output,
        model_id=args.model_id,
        device_name=args.device,
        preprocess_mode=args.preprocess_mode,
        max_resolution=args.max_resolution,
        use_point_map=args.use_point_map,
    )


if __name__ == "__main__":
    main()
