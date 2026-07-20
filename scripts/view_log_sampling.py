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
from init.ellipses import ellipse_mask
from init.io import load_config, load_dense_predictions, resolve_scene_path, resolve_scene_root
from init.sampling import (
    INITIALIZATION_METHOD,
    SamplingConfig,
    build_log_scale_space,
    detect_multiscale_keypoints,
    normalized_lab_to_rgb,
    scale_space_maxima,
    valid_pixel_mask,
)
from init.types import EllipseKeypoints

if __package__:
    from scripts.view_ellipses import ellipse_outline, parse_view_selection
else:
    from view_ellipses import ellipse_outline, parse_view_selection


HEADER_HEIGHT = 28
MAGNITUDE_COLOR = (255, 216, 32)
CHANNEL_COLORS = np.asarray(
    [
        [255, 210, 48],  # L: yellow
        [255, 72, 192],  # a: magenta
        [48, 192, 255],  # b: cyan
    ],
    dtype=np.uint8,
)
CHANNEL_NAMES = ("L", "a", "b")


def render_log_sampling(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    scene_root_override: str | Path | None = None,
    predictions_override: str | Path | None = None,
    view_selection: str | None = "0",
) -> dict[str, Any]:
    """Render the exact blur/LoG/keypoint stages for every configured scale."""
    validate_initialization_config(config)
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
        predictions.get("processed_valid_mask", np.ones((views, height, width), dtype=bool)),
        dtype=bool,
    )
    selected_views = (
        list(range(views))
        if view_selection is None or view_selection.strip().lower() == "all"
        else parse_view_selection(view_selection, views=views)
    )

    sampling = SamplingConfig.from_mapping(config.get("sampling"))
    sigma_values = sampling.sigmas
    response_threshold = sampling.response_threshold

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, int | float | str]] = []
    view_summaries: list[dict[str, Any]] = []

    for view_id in selected_views:
        detector_image = images[view_id]
        valid = valid_pixel_mask(world_points[view_id])
        valid &= image_valid_masks[view_id]
        scale_space = build_log_scale_space(
            detector_image,
            sigmas=sigma_values,
            valid_mask=valid,
            chroma_weight=sampling.chroma_weight,
            response_mad_epsilon=sampling.response_mad_epsilon,
        )
        responses_with_guards = scale_space.responses
        extrema_with_guards = scale_space_maxima(responses_with_guards)
        extrema_with_guards &= valid[None, :, :]
        extrema_with_guards &= np.abs(responses_with_guards) >= response_threshold
        keypoints = detect_keypoints(
            view_id=view_id,
            image=detector_image,
            world_points=world_points[view_id],
            image_valid_mask=image_valid_masks[view_id],
            sampling=sampling,
        )

        view_dir = destination / f"view_{view_id:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        scale_rows: list[Image.Image] = []
        for level, sigma in enumerate(sigma_values):
            blurred_channels = scale_space.blurred_channels[:, level + 1]
            response = responses_with_guards[level + 1]
            dominant_channels = dominant_channel_map(
                scale_space.channel_responses[:, level + 1],
                response_scales=scale_space.response_scales,
                channel_weights=scale_space.structure_weights,
            )
            raw_extrema = extrema_with_guards[level + 1]
            indices = np.flatnonzero(keypoints.levels == level)
            support_counts = build_support_counts(
                valid.shape,
                keypoints=keypoints,
                indices=indices,
                valid_mask=valid,
            )
            support_pixels = int(np.count_nonzero(support_counts))
            valid_pixels = int(np.count_nonzero(valid))
            coverage_fraction = support_pixels / max(valid_pixels, 1)
            selected_scores = keypoints.scores[indices]
            score_p50 = (
                float(np.percentile(selected_scores, 50.0)) if len(indices) else float("nan")
            )
            score_p95 = (
                float(np.percentile(selected_scores, 95.0)) if len(indices) else float("nan")
            )
            channel_counts = keypoint_channel_counts(
                dominant_channels,
                us=keypoints.us[indices],
                vs=keypoints.vs[indices],
            )

            detector_rgb = float_rgb_to_image(detector_image)
            blurred_rgb = lab_blur_to_image(blurred_channels, valid_mask=valid)
            contribution_image = render_channel_contributions(
                response,
                dominant_channels=dominant_channels,
                valid_mask=valid,
                selected_us=keypoints.us[indices],
                selected_vs=keypoints.vs[indices],
            )
            response_image = render_log_response(
                response,
                valid_mask=valid,
                raw_extrema=raw_extrema,
                selected_us=keypoints.us[indices],
                selected_vs=keypoints.vs[indices],
            )
            original_samples = render_sampling_overlay(
                detector_rgb,
                keypoints=keypoints,
                indices=indices,
                support_counts=support_counts,
                dominant_channels=dominant_channels,
            )
            kernel_radius = max(1, int(math.ceil(3.0 * sigma)))
            raw_count = int(np.count_nonzero(raw_extrema))
            selected_count = int(len(indices))
            panels = [
                add_header(
                    blurred_rgb,
                    "Lab blur reconstructed to RGB: "
                    f"sigma={sigma:g}, kernel radius={kernel_radius}px",
                ),
                add_header(
                    contribution_image,
                    "Dominant LoG: "
                    f"L={channel_counts[0]}, a={channel_counts[1]}, b={channel_counts[2]}",
                ),
                add_header(
                    response_image,
                    f"Fused LoG magnitude: raw={raw_count}, selected={selected_count}",
                ),
                add_header(
                    original_samples,
                    f"Supports on RGB: union coverage={coverage_fraction:.1%}",
                ),
            ]
            row = concatenate_horizontally(panels)
            scale_rows.append(row)
            output_name = f"level_{level:02d}_sigma_{format_sigma(sigma)}.png"
            row.save(view_dir / output_name)
            csv_rows.append(
                {
                    "view_id": view_id,
                    "level": level,
                    "sigma": sigma,
                    "kernel_radius": kernel_radius,
                    "raw_extrema": raw_count,
                    "selected_keypoints": selected_count,
                    "selected_L": int(channel_counts[0]),
                    "selected_a": int(channel_counts[1]),
                    "selected_b": int(channel_counts[2]),
                    "support_pixels": support_pixels,
                    "support_fraction": coverage_fraction,
                    "score_p50": score_p50,
                    "score_p95": score_p95,
                    "image": str(Path(f"view_{view_id:03d}") / output_name),
                }
            )

        contact_sheet = concatenate_vertically(scale_rows)
        contact_sheet_path = view_dir / "contact_sheet.png"
        contact_sheet.save(contact_sheet_path)
        view_summaries.append(
            {
                "view_id": view_id,
                "keypoint_count": len(keypoints),
                "channel_response_scales": {
                    name: float(value)
                    for name, value in zip(
                        scale_space.channel_names,
                        scale_space.response_scales,
                        strict=True,
                    )
                },
                "contact_sheet": str(contact_sheet_path.relative_to(destination)),
            }
        )
        print(
            f"View {view_id:03d}: {len(keypoints)} selected keypoints; saved {contact_sheet_path}"
        )

    csv_path = destination / "scales.csv"
    fieldnames = [
        "view_id",
        "level",
        "sigma",
        "kernel_radius",
        "raw_extrema",
        "selected_keypoints",
        "selected_L",
        "selected_a",
        "selected_b",
        "support_pixels",
        "support_fraction",
        "score_p50",
        "score_p95",
        "image",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary: dict[str, Any] = {
        "config": str(config.get("_config_path", "")),
        "predictions": str(predictions_path),
        "detector_input": "dense-geometry-aligned processed RGB before Gaussian blur",
        "detector": INITIALIZATION_METHOD,
        "chroma_weight": sampling.chroma_weight,
        "response_mad_epsilon": sampling.response_mad_epsilon,
        "sigmas": list(sigma_values),
        "response_threshold": response_threshold,
        "legend": {
            "support_fill": "green: discrete 2D ellipse support before 3D continuity",
            "channel_colors": "L=yellow, a=magenta, b=cyan",
            "ellipse_outline": "dominant normalized Lab LoG channel at the keypoint",
            "keypoint_center": "white",
            "log_raw_extrema": "yellow",
            "log_selected_after_ellipse_merge": "white",
            "invalid_pixels": "purple in blur panels; excluded by content/finite-geometry mask",
        },
        "views": view_summaries,
        "csv": str(csv_path),
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def detect_keypoints(
    *,
    view_id: int,
    image: np.ndarray,
    world_points: np.ndarray,
    image_valid_mask: np.ndarray,
    sampling: SamplingConfig,
) -> EllipseKeypoints:
    return detect_multiscale_keypoints(
        view_id=view_id,
        image=image,
        world_points=world_points,
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
        image_valid_mask=image_valid_mask,
    )


def build_support_counts(
    image_shape: tuple[int, int],
    *,
    keypoints: EllipseKeypoints,
    indices: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != image_shape:
        raise ValueError("valid_mask must match image_shape")
    counts = np.zeros(image_shape, dtype=np.uint16)
    for index in np.asarray(indices, dtype=np.int64):
        support = ellipse_mask(
            image_shape,
            u=int(keypoints.us[index]),
            v=int(keypoints.vs[index]),
            ellipse_matrix=keypoints.ellipse_matrices[index],
        )
        counts += (support & valid).astype(np.uint16)
    return counts


def dominant_channel_map(
    channel_responses: np.ndarray,
    *,
    response_scales: np.ndarray,
    channel_weights: np.ndarray,
) -> np.ndarray:
    """Return the strongest normalized Lab LoG channel at every pixel."""
    responses = np.asarray(channel_responses, dtype=np.float32)
    scales = np.asarray(response_scales, dtype=np.float32)
    weights = np.asarray(channel_weights, dtype=np.float32)
    if responses.ndim != 3 or responses.shape[0] != len(CHANNEL_NAMES):
        raise ValueError("channel_responses must have shape [3, H, W]")
    if scales.shape != (len(CHANNEL_NAMES),) or weights.shape != (len(CHANNEL_NAMES),):
        raise ValueError("response_scales and channel_weights must have shape [3]")
    normalized_energy = weights[:, None, None] * (responses / scales[:, None, None]) ** 2
    return np.argmax(normalized_energy, axis=0).astype(np.uint8)


def keypoint_channel_counts(
    dominant_channels: np.ndarray,
    *,
    us: np.ndarray,
    vs: np.ndarray,
) -> np.ndarray:
    channels = np.asarray(dominant_channels, dtype=np.uint8)
    u_values = np.asarray(us, dtype=np.int64)
    v_values = np.asarray(vs, dtype=np.int64)
    if channels.ndim != 2 or u_values.shape != v_values.shape:
        raise ValueError("dominant channel map and keypoint coordinates are inconsistent")
    if not len(u_values):
        return np.zeros((len(CHANNEL_NAMES),), dtype=np.int64)
    return np.bincount(channels[v_values, u_values], minlength=len(CHANNEL_NAMES))


def lab_blur_to_image(
    blurred_channels: np.ndarray,
    *,
    valid_mask: np.ndarray,
) -> Image.Image:
    channels = np.asarray(blurred_channels, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if channels.shape != (len(CHANNEL_NAMES), *valid.shape):
        raise ValueError("blurred_channels must have shape [3, H, W] matching valid_mask")
    rgb = normalized_lab_to_rgb(np.moveaxis(channels, 0, -1))
    rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb_u8[~valid] = np.asarray([80, 0, 80], dtype=np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB")


def render_channel_contributions(
    fused_response: np.ndarray,
    *,
    dominant_channels: np.ndarray,
    valid_mask: np.ndarray,
    selected_us: np.ndarray,
    selected_vs: np.ndarray,
) -> Image.Image:
    response = np.asarray(fused_response, dtype=np.float32)
    channels = np.asarray(dominant_channels, dtype=np.uint8)
    valid = np.asarray(valid_mask, dtype=bool)
    if response.shape != valid.shape or channels.shape != valid.shape:
        raise ValueError("response, dominant_channels, and valid_mask must share shape")
    finite_valid = valid & np.isfinite(response)
    scale = float(np.percentile(response[finite_valid], 99.0)) if np.any(finite_valid) else 1.0
    strength = np.sqrt(np.clip(response / max(scale, 1.0e-12), 0.0, 1.0))
    rgb = CHANNEL_COLORS[channels].astype(np.float32) * strength[..., None]
    rgb[~valid] = np.asarray([45.0, 45.0, 45.0], dtype=np.float32)
    output = Image.fromarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(output)
    for u, v in zip(selected_us, selected_vs, strict=True):
        u_int = int(u)
        v_int = int(v)
        draw.ellipse((u_int - 2, v_int - 2, u_int + 2, v_int + 2), outline=(255, 255, 255))
    return output


def render_sampling_overlay(
    image: Image.Image,
    *,
    keypoints: EllipseKeypoints,
    indices: np.ndarray,
    support_counts: np.ndarray,
    dominant_channels: np.ndarray,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    counts = np.asarray(support_counts)
    if counts.shape != base.shape[:2]:
        raise ValueError("support_counts must match image dimensions")
    channels = np.asarray(dominant_channels, dtype=np.uint8)
    if channels.shape != counts.shape:
        raise ValueError("dominant_channels must match image dimensions")
    covered = counts > 0
    alpha = np.clip(0.20 + 0.08 * np.maximum(counts.astype(np.float32) - 1.0, 0.0), 0.0, 0.52)
    green = np.asarray([40.0, 230.0, 90.0], dtype=np.float32)
    covered_alpha = alpha[covered][:, None]
    base[covered] = base[covered] * (1.0 - covered_alpha) + green[None, :] * covered_alpha
    output = Image.fromarray(np.clip(base, 0.0, 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(output)
    for index in np.asarray(indices, dtype=np.int64):
        u = int(keypoints.us[index])
        v = int(keypoints.vs[index])
        outline = ellipse_outline(
            u=float(u),
            v=float(v),
            ellipse_matrix=keypoints.ellipse_matrices[index],
        )
        color = tuple(int(value) for value in CHANNEL_COLORS[channels[v, u]])
        draw.line([*outline, outline[0]], fill=color, width=1, joint="curve")
        draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill=(255, 255, 255))
    return output


def render_log_response(
    response: np.ndarray,
    *,
    valid_mask: np.ndarray,
    raw_extrema: np.ndarray,
    selected_us: np.ndarray,
    selected_vs: np.ndarray,
) -> Image.Image:
    values = np.asarray(response, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    raw = np.asarray(raw_extrema, dtype=bool)
    if valid.shape != values.shape or raw.shape != values.shape:
        raise ValueError("response, valid_mask, and raw_extrema must share shape")
    finite_valid = valid & np.isfinite(values)
    scale = (
        float(np.percentile(np.abs(values[finite_valid]), 99.0)) if np.any(finite_valid) else 1.0
    )
    scale = max(scale, 1.0e-12)
    normalized = np.clip(values / scale, 0.0, 1.0)
    strength = np.sqrt(normalized)
    rgb = np.zeros((*values.shape, 3), dtype=np.float32)
    rgb[..., 0] = 255.0 * strength
    rgb[..., 1] = 64.0 * strength
    rgb[~valid] = np.asarray([45.0, 45.0, 45.0], dtype=np.float32)
    output = Image.fromarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(output)
    raw_vs, raw_us = np.nonzero(raw)
    for u, v in zip(raw_us, raw_vs, strict=True):
        draw.point((int(u), int(v)), fill=(255, 232, 32))
    for u, v in zip(selected_us, selected_vs, strict=True):
        u_int = int(u)
        v_int = int(v)
        draw.ellipse((u_int - 2, v_int - 2, u_int + 2, v_int + 2), outline=(255, 255, 255))
    return output


def float_rgb_to_image(image: np.ndarray) -> Image.Image:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    return Image.fromarray(np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")


def add_header(image: Image.Image, title: str) -> Image.Image:
    rgb = image.convert("RGB")
    output = Image.new("RGB", (rgb.width, rgb.height + HEADER_HEIGHT), (16, 16, 16))
    output.paste(rgb, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(output)
    draw.text((6, 7), title, fill=(255, 255, 255))
    return output


def concatenate_horizontally(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required")
    height = max(image.height for image in images)
    output = Image.new("RGB", (sum(image.width for image in images), height), (0, 0, 0))
    x = 0
    for image in images:
        output.paste(image.convert("RGB"), (x, 0))
        x += image.width
    return output


def concatenate_vertically(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required")
    width = max(image.width for image in images)
    output = Image.new("RGB", (width, sum(image.height for image in images)), (0, 0, 0))
    y = 0
    for image in images:
        output.paste(image.convert("RGB"), (0, y))
        y += image.height
    return output


def format_sigma(sigma: float) -> str:
    return f"{sigma:.3f}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Gaussian blur, LoG extrema, and selected ellipse supports per scale."
    )
    parser.add_argument("--config", type=Path, required=True, help="Initialization YAML config.")
    parser.add_argument("--scene-root", type=Path, default=None, help="Override scene root.")
    parser.add_argument("--predictions", type=Path, default=None, help="Override predictions NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="Output diagnostic directory.")
    parser.add_argument(
        "--views",
        default="0",
        help="View IDs/ranges such as 0,3,6-10, or 'all'; default is view 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    render_log_sampling(
        config,
        output_dir=args.output,
        scene_root_override=args.scene_root,
        predictions_override=args.predictions,
        view_selection=args.views,
    )


if __name__ == "__main__":
    main()
