from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from init.build_init import (
    load_scene_images,
    validate_initialization_config,
    validate_prediction_precision_contract,
)
from init.io import load_config, load_dense_predictions, resolve_scene_path, resolve_scene_root
from init.sampling import SamplingConfig, detect_multiscale_keypoints


def ellipse_axes(ellipse_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return descending semi-axis lengths and their image-space directions."""
    matrix = np.asarray(ellipse_matrix, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("ellipse_matrix must be a finite 2x2 matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.any(eigenvalues <= 0.0):
        raise ValueError("ellipse_matrix must be positive definite")
    return np.sqrt(eigenvalues).astype(np.float32), eigenvectors.astype(np.float32)


def ellipse_outline(
    *,
    u: float,
    v: float,
    ellipse_matrix: np.ndarray,
    samples: int = 96,
) -> list[tuple[float, float]]:
    if samples < 8:
        raise ValueError("ellipse outline requires at least eight samples")
    axes, directions = ellipse_axes(ellipse_matrix)
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False, dtype=np.float32)
    unit_circle = np.stack([np.cos(angles), np.sin(angles)], axis=0)
    offsets = directions @ (axes[:, None] * unit_circle)
    points = offsets.T + np.asarray([u, v], dtype=np.float32)
    return [(float(x), float(y)) for x, y in points]


def render_ellipse_overlays(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    scene_root_override: str | Path | None = None,
    predictions_override: str | Path | None = None,
    view_selection: str | None = None,
    max_ellipses_per_view: int = 0,
    highlight_ratio: float = 4.0,
    line_width: int = 2,
) -> dict[str, Any]:
    validate_initialization_config(config)
    if max_ellipses_per_view < 0:
        raise ValueError("max_ellipses_per_view must be non-negative")
    if not np.isfinite(highlight_ratio) or highlight_ratio < 1.0:
        raise ValueError("highlight_ratio must be finite and at least one")
    if line_width < 1:
        raise ValueError("line_width must be positive")

    scene_cfg = config.get("scene", {})
    scene_root = resolve_scene_root(config, scene_root_override)
    predictions_path = resolve_scene_path(
        scene_root,
        predictions_override or scene_cfg.get("predictions_path", "vggt/predictions.npz"),
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
    selected_views = parse_view_selection(view_selection, views=views)

    sampling = SamplingConfig.from_mapping(config.get("sampling"))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, int | float | bool]] = []
    view_summaries: list[dict[str, int | float | str]] = []

    for view_id in selected_views:
        keypoints = detect_multiscale_keypoints(
            view_id=view_id,
            image=images[view_id],
            world_points=world_points[view_id],
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
            image_valid_mask=image_valid_masks[view_id],
        )

        axes = np.empty((len(keypoints), 2), dtype=np.float32)
        directions = np.empty((len(keypoints), 2, 2), dtype=np.float32)
        for index, matrix in enumerate(keypoints.ellipse_matrices):
            axes[index], directions[index] = ellipse_axes(matrix)
        ratios = (
            axes[:, 0] / np.maximum(axes[:, 1], 1.0e-20)
            if len(keypoints)
            else np.empty((0,), dtype=np.float32)
        )

        draw_indices = np.arange(len(keypoints), dtype=np.int64)
        if max_ellipses_per_view and len(draw_indices) > max_ellipses_per_view:
            draw_indices = np.argsort(-ratios, kind="stable")[:max_ellipses_per_view]
        drawn = np.zeros((len(keypoints),), dtype=bool)
        drawn[draw_indices] = True

        base = Image.fromarray(
            np.clip(images[view_id] * 255.0, 0.0, 255.0).astype(np.uint8),
            mode="RGB",
        ).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for index in draw_indices:
            ratio = float(ratios[index])
            color = ellipse_color(ratio, highlight_ratio=highlight_ratio)
            outline = ellipse_outline(
                u=float(keypoints.us[index]),
                v=float(keypoints.vs[index]),
                ellipse_matrix=keypoints.ellipse_matrices[index],
            )
            draw.line([*outline, outline[0]], fill=(*color, 230), width=line_width, joint="curve")
            u = int(keypoints.us[index])
            v = int(keypoints.vs[index])
            draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill=(255, 255, 255, 230))

        ratio_percentiles = (
            np.percentile(ratios, [50.0, 95.0, 100.0])
            if len(ratios)
            else np.zeros((3,), dtype=np.float32)
        )
        label = (
            f"view {view_id:03d} | ellipses {len(keypoints)} | "
            f"axis ratio p50/p95/max "
            f"{ratio_percentiles[0]:.2f}/{ratio_percentiles[1]:.2f}/{ratio_percentiles[2]:.2f}"
        )
        text_box = draw.textbbox((6, 5), label)
        draw.rectangle(
            (text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2),
            fill=(0, 0, 0, 180),
        )
        draw.text((6, 5), label, fill=(255, 255, 255, 255))
        output_name = f"view_{view_id:03d}.png"
        Image.alpha_composite(base, overlay).convert("RGB").save(destination / output_name)

        for index in range(len(keypoints)):
            major_direction = directions[index, :, 0]
            csv_rows.append(
                {
                    "view_id": view_id,
                    "u": int(keypoints.us[index]),
                    "v": int(keypoints.vs[index]),
                    "sigma": float(keypoints.sigmas[index]),
                    "level": int(keypoints.levels[index]),
                    "score": float(keypoints.scores[index]),
                    "area": float(keypoints.ellipse_areas[index]),
                    "major_radius": float(axes[index, 0]),
                    "minor_radius": float(axes[index, 1]),
                    "axis_ratio": float(ratios[index]),
                    "major_angle_degrees": math.degrees(
                        math.atan2(float(major_direction[1]), float(major_direction[0]))
                    ),
                    "drawn": bool(drawn[index]),
                }
            )
        view_summaries.append(
            {
                "view_id": view_id,
                "ellipse_count": len(keypoints),
                "drawn_count": int(np.count_nonzero(drawn)),
                "axis_ratio_p50": float(ratio_percentiles[0]),
                "axis_ratio_p95": float(ratio_percentiles[1]),
                "axis_ratio_max": float(ratio_percentiles[2]),
                "overlay": output_name,
            }
        )
        print(
            f"View {view_id:03d}: {len(keypoints)} ellipses, "
            f"axis ratio p50/p95/max="
            f"{ratio_percentiles[0]:.2f}/{ratio_percentiles[1]:.2f}/{ratio_percentiles[2]:.2f}"
        )

    csv_path = destination / "ellipses.csv"
    fieldnames = [
        "view_id",
        "u",
        "v",
        "sigma",
        "level",
        "score",
        "area",
        "major_radius",
        "minor_radius",
        "axis_ratio",
        "major_angle_degrees",
        "drawn",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary: dict[str, Any] = {
        "config": str(config.get("_config_path", "")),
        "predictions": str(predictions_path),
        "highlight_ratio": highlight_ratio,
        "view_count": len(selected_views),
        "ellipse_count": len(csv_rows),
        "views": view_summaries,
        "csv": str(csv_path),
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved overlays and ellipse statistics to {destination}")
    return summary


def ellipse_color(ratio: float, *, highlight_ratio: float) -> tuple[int, int, int]:
    if ratio >= highlight_ratio:
        return (255, 48, 48)
    if ratio >= 2.0:
        return (255, 196, 32)
    return (48, 224, 96)


def parse_view_selection(selection: str | None, *, views: int) -> list[int]:
    if selection is None or selection.strip().lower() in {"", "all"}:
        return list(range(views))
    selected: set[int] = set()
    for item in selection.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending view range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    if not selected:
        raise ValueError("No views were selected")
    if min(selected) < 0 or max(selected) >= views:
        raise ValueError(f"Selected views must lie in [0, {views - 1}]")
    return sorted(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw detected Lab LoG/structure-tensor ellipses over aligned images."
    )
    parser.add_argument("--config", type=Path, required=True, help="Initialization YAML config.")
    parser.add_argument("--scene-root", type=Path, default=None, help="Override scene root.")
    parser.add_argument("--predictions", type=Path, default=None, help="Override predictions NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="Output overlay directory.")
    parser.add_argument(
        "--views",
        default=None,
        help="Comma-separated view IDs/ranges or 'all'; default is all views.",
    )
    parser.add_argument(
        "--max-ellipses-per-view",
        type=int,
        default=0,
        help="Draw only the most elongated N ellipses per view; zero draws all.",
    )
    parser.add_argument(
        "--highlight-ratio",
        type=float,
        default=4.0,
        help="Draw ellipses at or above this axis ratio in red.",
    )
    parser.add_argument("--line-width", type=int, default=2, help="Ellipse outline width.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    render_ellipse_overlays(
        config,
        output_dir=args.output,
        scene_root_override=args.scene_root,
        predictions_override=args.predictions,
        view_selection=args.views,
        max_ellipses_per_view=args.max_ellipses_per_view,
        highlight_ratio=args.highlight_ratio,
        line_width=args.line_width,
    )


if __name__ == "__main__":
    main()
