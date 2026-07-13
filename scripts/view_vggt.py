from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import viser

from init.build_init import (
    confidence_threshold_from_percentile,
    validate_vggt_precision_contract,
)
from init.gaussian_params import rotation_matrix_to_quaternion
from init.io import load_vggt_predictions
from preprocess.export_vggt_geometry import normalize_confidence


def load_vggt_view_data(
    path: str | Path,
    *,
    confidence_percentile: float,
    stride: int,
    max_points: int | None,
) -> dict[str, np.ndarray | float]:
    if stride < 1:
        raise ValueError("stride must be at least one")
    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be positive or None")

    predictions = load_vggt_predictions(path)
    validate_vggt_precision_contract(predictions)

    world_points = predictions["world_points"]
    confidence = predictions["confidence"]
    views, height, width, _ = world_points.shape
    content_mask = np.asarray(
        predictions.get(
            "processed_valid_mask",
            np.ones((views, height, width), dtype=bool),
        ),
        dtype=bool,
    )
    confidence_threshold = confidence_threshold_from_percentile(
        confidence,
        world_points,
        content_mask,
        percentile=confidence_percentile,
    )

    points_map = world_points[:, ::stride, ::stride]
    confidence_map = confidence[:, ::stride, ::stride]
    valid = content_mask[:, ::stride, ::stride]
    valid &= np.isfinite(points_map).all(axis=-1)
    valid &= np.isfinite(confidence_map)
    valid &= confidence_map >= confidence_threshold

    images = predictions.get("processed_images")
    if images is not None:
        color_map = np.asarray(images[:, ::stride, ::stride], dtype=np.float32)
        valid &= np.isfinite(color_map).all(axis=-1)
        colors = np.clip(color_map[valid] * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        selected_confidence = confidence_map[valid]
        colors = np.repeat(
            (normalize_confidence(selected_confidence) * 255.0).astype(np.uint8)[:, None],
            3,
            axis=1,
        )

    points = points_map[valid].astype(np.float32)
    if points.shape[0] == 0:
        raise ValueError(
            "No VGGT points remain after applying the content mask, confidence filter, "
            "and spatial stride"
        )

    if max_points is not None and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
        colors = colors[indices]

    lower, upper = np.percentile(points, [5.0, 95.0], axis=0)
    scene_center = ((lower + upper) * 0.5).astype(np.float32)
    scene_scale = float(np.linalg.norm(upper - lower))
    if not np.isfinite(scene_scale) or scene_scale <= 1.0e-8:
        scene_scale = 1.0

    output: dict[str, np.ndarray | float] = {
        "points": points - scene_center,
        "colors": colors,
        "scene_center": scene_center,
        "scene_scale": scene_scale,
        "confidence_threshold": confidence_threshold,
    }
    if images is not None:
        output["images"] = np.asarray(images, dtype=np.float32)
    if "intrinsics" in predictions:
        output["intrinsics"] = predictions["intrinsics"]

    extrinsics_c2w = predictions.get("extrinsics_c2w", predictions.get("extrinsics"))
    if extrinsics_c2w is None and "extrinsics_w2c" in predictions:
        extrinsics_c2w = np.linalg.inv(predictions["extrinsics_w2c"]).astype(np.float32)
    if extrinsics_c2w is not None:
        output["extrinsics_c2w"] = _as_4x4_extrinsics(extrinsics_c2w, views=views)
    return output


def _as_4x4_extrinsics(extrinsics: np.ndarray, *, views: int) -> np.ndarray:
    matrices = np.asarray(extrinsics, dtype=np.float32)
    if matrices.shape == (views, 4, 4):
        return matrices
    if matrices.shape == (views, 3, 4):
        output = np.repeat(np.eye(4, dtype=np.float32)[None], views, axis=0)
        output[:, :3] = matrices
        return output
    raise ValueError(
        f"camera extrinsics must have shape {(views, 3, 4)} or {(views, 4, 4)}, "
        f"got {matrices.shape}"
    )


def add_cameras(
    server: viser.ViserServer,
    view_data: dict[str, np.ndarray | float],
) -> int:
    if "extrinsics_c2w" not in view_data:
        return 0

    extrinsics_c2w = np.asarray(view_data["extrinsics_c2w"], dtype=np.float32)
    scene_center = np.asarray(view_data["scene_center"], dtype=np.float32)
    scene_scale = float(view_data["scene_scale"])
    images = view_data.get("images")
    intrinsics = view_data.get("intrinsics")
    if images is not None:
        images = np.asarray(images, dtype=np.float32)
    if intrinsics is not None:
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.shape != (extrinsics_c2w.shape[0], 3, 3):
            raise ValueError(
                "intrinsics must have shape "
                f"{(extrinsics_c2w.shape[0], 3, 3)}, got {intrinsics.shape}"
            )

    for view_id, c2w in enumerate(extrinsics_c2w):
        if not np.isfinite(c2w).all():
            raise ValueError(f"camera {view_id} contains non-finite extrinsics")
        server.scene.add_frame(
            f"/cameras/{view_id}",
            wxyz=rotation_matrix_to_quaternion(c2w[:3, :3]),
            position=c2w[:3, 3] - scene_center,
            axes_length=0.03 * scene_scale,
            axes_radius=0.0015 * scene_scale,
            origin_radius=0.002 * scene_scale,
        )

        if images is None or intrinsics is None:
            continue
        height, width = images.shape[1:3]
        fy = float(intrinsics[view_id, 1, 1])
        if not np.isfinite(fy) or fy <= 0.0:
            raise ValueError(f"camera {view_id} has an invalid vertical focal length")
        image = np.clip(images[view_id] * 255.0, 0.0, 255.0).astype(np.uint8)
        server.scene.add_camera_frustum(
            f"/cameras/{view_id}/frustum",
            fov=2.0 * np.arctan2(height * 0.5, fy),
            aspect=width / height,
            scale=0.04 * scene_scale,
            line_width=1.0,
            image=image,
        )
    return int(extrinsics_c2w.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View VGGT predictions in a web browser.")
    parser.add_argument("--input", type=Path, required=True, help="Input predictions.npz path.")
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=25.0,
        help="Discard this bottom percentage of valid confidence scores.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Keep one point every N pixels in each image dimension.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500_000,
        help="Maximum number of points sent to the browser.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Viewer bind address.")
    parser.add_argument("--port", type=int, default=8080, help="Viewer port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    view_data = load_vggt_view_data(
        args.input,
        confidence_percentile=args.confidence_percentile,
        stride=args.stride,
        max_points=args.max_points,
    )

    server = viser.ViserServer(host=args.host, port=args.port)
    point_size = max(float(view_data["scene_scale"]) * 0.0005, 1.0e-5)
    points = np.asarray(view_data["points"], dtype=np.float32)
    server.scene.add_point_cloud(
        "/vggt/points",
        points=points,
        colors=np.asarray(view_data["colors"], dtype=np.uint8),
        point_size=point_size,
        point_shape="circle",
        precision="float32",
    )
    camera_count = add_cameras(server, view_data)

    print(f"Loaded {points.shape[0]} VGGT points from {args.input}")
    print(
        "Confidence threshold: "
        f"{float(view_data['confidence_threshold']):.6g} "
        f"(bottom {args.confidence_percentile:g}% removed)"
    )
    print(f"Displayed {camera_count} camera poses")
    print(f"Open http://127.0.0.1:{server.get_port()} in a browser")
    print("Drag to rotate; scroll to zoom; press Ctrl+C to stop")
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
