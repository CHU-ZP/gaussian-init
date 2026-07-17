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
    confidence_threshold_from_percentile,
    load_scene_images,
    validate_initialization_config,
    validate_vggt_precision_contract,
)
from init.io import load_config, load_vggt_predictions, resolve_scene_path
from init.sampling import (
    INITIALIZATION_METHOD,
    SamplingConfig,
    build_log_scale_space,
    multichannel_structure_tensor,
    scale_space_maxima,
    tensor_to_ellipse,
    valid_pixel_mask,
)
from init.types import EllipseKeypoints

if __package__:
    from scripts.view_ellipses import ellipse_outline, parse_view_selection
    from scripts.view_log_sampling import (
        CHANNEL_COLORS,
        add_header,
        build_support_counts,
        concatenate_horizontally,
        concatenate_vertically,
        detect_keypoints,
        dominant_channel_map,
        float_rgb_to_image,
        format_sigma,
        keypoint_channel_counts,
        lab_blur_to_image,
        render_channel_contributions,
        render_sampling_overlay,
    )
else:
    from view_ellipses import ellipse_outline, parse_view_selection
    from view_log_sampling import (
        CHANNEL_COLORS,
        add_header,
        build_support_counts,
        concatenate_horizontally,
        concatenate_vertically,
        detect_keypoints,
        dominant_channel_map,
        float_rgb_to_image,
        format_sigma,
        keypoint_channel_counts,
        lab_blur_to_image,
        render_channel_contributions,
        render_sampling_overlay,
    )


def render_ellipse_merging(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    scene_root_override: str | Path | None = None,
    predictions_override: str | Path | None = None,
    view_selection: str | None = "0",
) -> dict[str, Any]:
    """Visualize raw candidates and the result of same-scale ellipse merging."""
    validate_initialization_config(config)
    scene_cfg = config.get("scene", {})
    scene_root = Path(scene_root_override or scene_cfg.get("root", "data/scene_x"))
    predictions_path = resolve_scene_path(
        scene_root,
        predictions_override or scene_cfg.get("predictions_path", "vggt/predictions.npz"),
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
    confidence_percentile = sampling.confidence_percentile
    confidence_threshold = confidence_threshold_from_percentile(
        confidence,
        world_points,
        image_valid_masks,
        percentile=confidence_percentile,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, int | float | str]] = []
    view_summaries: list[dict[str, Any]] = []

    for view_id in selected_views:
        detector_image = images[view_id]
        detector_rgb = float_rgb_to_image(detector_image)
        valid = valid_pixel_mask(confidence[view_id], world_points[view_id], confidence_threshold)
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
        selected = detect_keypoints(
            view_id=view_id,
            image=detector_image,
            confidence=confidence[view_id],
            world_points=world_points[view_id],
            image_valid_mask=image_valid_masks[view_id],
            confidence_threshold=confidence_threshold,
            sampling=sampling,
        )

        view_dir = destination / f"view_{view_id:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        scale_rows: list[Image.Image] = []
        raw_total = 0
        for level, sigma in enumerate(sigma_values):
            blurred_channels = scale_space.blurred_channels[:, level + 1]
            response = responses_with_guards[level + 1]
            dominant_channels = dominant_channel_map(
                scale_space.channel_responses[:, level + 1],
                response_scales=scale_space.response_scales,
                channel_weights=scale_space.structure_weights,
            )
            raw = build_raw_ellipse_candidates(
                view_id=view_id,
                level=level,
                sigma=sigma,
                blurred_channels=blurred_channels,
                response=response,
                extrema=extrema_with_guards[level + 1],
                valid_mask=valid,
                structure_weights=scale_space.structure_weights,
                sampling=sampling,
            )
            raw_total += len(raw)
            selected_indices = np.flatnonzero(selected.levels == level)
            raw_indices = np.arange(len(raw), dtype=np.int64)
            raw_support = build_support_counts(
                valid.shape,
                keypoints=raw,
                indices=raw_indices,
                valid_mask=valid,
            )
            selected_support = build_support_counts(
                valid.shape,
                keypoints=selected,
                indices=selected_indices,
                valid_mask=valid,
            )
            raw_support_pixels = int(np.count_nonzero(raw_support))
            selected_support_pixels = int(np.count_nonzero(selected_support))
            valid_pixels = int(np.count_nonzero(valid))
            raw_coverage = raw_support_pixels / max(valid_pixels, 1)
            selected_coverage = selected_support_pixels / max(valid_pixels, 1)
            retained_fraction = len(selected_indices) / max(len(raw), 1)
            raw_channel_counts = keypoint_channel_counts(
                dominant_channels,
                us=raw.us,
                vs=raw.vs,
            )
            selected_channel_counts = keypoint_channel_counts(
                dominant_channels,
                us=selected.us[selected_indices],
                vs=selected.vs[selected_indices],
            )

            blurred_rgb = lab_blur_to_image(blurred_channels, valid_mask=valid)
            contribution_image = render_channel_contributions(
                response,
                dominant_channels=dominant_channels,
                valid_mask=valid,
                selected_us=selected.us[selected_indices],
                selected_vs=selected.vs[selected_indices],
            )
            raw_ellipses_on_rgb = render_candidate_ellipses(
                detector_rgb,
                keypoints=raw,
                dominant_channels=dominant_channels,
            )
            selected_overlay = render_sampling_overlay(
                detector_rgb,
                keypoints=selected,
                indices=selected_indices,
                support_counts=selected_support,
                dominant_channels=dominant_channels,
            )
            kernel_radius = max(1, int(math.ceil(3.0 * sigma)))
            panels = [
                add_header(
                    blurred_rgb,
                    "Lab blur reconstructed to RGB: "
                    f"sigma={sigma:g}, kernel radius={kernel_radius}px",
                ),
                add_header(
                    contribution_image,
                    "Raw dominant LoG: "
                    f"L={raw_channel_counts[0]}, a={raw_channel_counts[1]}, "
                    f"b={raw_channel_counts[2]}",
                ),
                add_header(
                    raw_ellipses_on_rgb,
                    f"Raw ellipses on RGB: coverage={raw_coverage:.1%}",
                ),
                add_header(
                    selected_overlay,
                    f"After ellipse merge: {len(selected_indices)} ({retained_fraction:.1%})",
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
                    "raw_candidates": len(raw),
                    "merged_keypoints": len(selected_indices),
                    "raw_L": int(raw_channel_counts[0]),
                    "raw_a": int(raw_channel_counts[1]),
                    "raw_b": int(raw_channel_counts[2]),
                    "merged_L": int(selected_channel_counts[0]),
                    "merged_a": int(selected_channel_counts[1]),
                    "merged_b": int(selected_channel_counts[2]),
                    "retained_fraction": retained_fraction,
                    "raw_support_pixels": raw_support_pixels,
                    "raw_support_fraction": raw_coverage,
                    "raw_max_overlap": int(raw_support.max(initial=0)),
                    "merged_support_pixels": selected_support_pixels,
                    "merged_support_fraction": selected_coverage,
                    "image": str(Path(f"view_{view_id:03d}") / output_name),
                }
            )

        contact_sheet = concatenate_vertically(scale_rows)
        contact_sheet_path = view_dir / "contact_sheet.png"
        contact_sheet.save(contact_sheet_path)
        view_summaries.append(
            {
                "view_id": view_id,
                "raw_candidates": raw_total,
                "merged_keypoints": len(selected),
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
            f"View {view_id:03d}: {raw_total} raw candidates -> "
            f"{len(selected)} merged outputs; saved {contact_sheet_path}"
        )

    csv_path = destination / "ellipse_merge_scales.csv"
    fieldnames = [
        "view_id",
        "level",
        "sigma",
        "kernel_radius",
        "raw_candidates",
        "merged_keypoints",
        "raw_L",
        "raw_a",
        "raw_b",
        "merged_L",
        "merged_a",
        "merged_b",
        "retained_fraction",
        "raw_support_pixels",
        "raw_support_fraction",
        "raw_max_overlap",
        "merged_support_pixels",
        "merged_support_fraction",
        "image",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary: dict[str, Any] = {
        "config": str(config.get("_config_path", "")),
        "predictions": str(predictions_path),
        "candidate_stage": (
            "after strict 3x3x3 x/y/scale extrema and response threshold; "
            "before same-scale ellipse similarity merging"
        ),
        "sigmas": list(sigma_values),
        "detector": INITIALIZATION_METHOD,
        "chroma_weight": sampling.chroma_weight,
        "response_mad_epsilon": sampling.response_mad_epsilon,
        "response_threshold": response_threshold,
        "confidence_threshold": confidence_threshold,
        "ellipse_merge": {key: value for key, value in sampling.ellipse_merge.__dict__.items()},
        "legend": {
            "channel_colors": "L=yellow, a=magenta, b=cyan",
            "raw_ellipses": "outline is the dominant normalized Lab LoG channel",
            "merged_support": "green fill, channel-colored outline, white center",
            "invalid_pixels": "purple in blur panels",
        },
        "views": view_summaries,
        "csv": str(csv_path),
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def build_raw_ellipse_candidates(
    *,
    view_id: int,
    level: int,
    sigma: float,
    blurred_channels: np.ndarray,
    response: np.ndarray,
    extrema: np.ndarray,
    valid_mask: np.ndarray,
    structure_weights: np.ndarray,
    sampling: SamplingConfig,
) -> EllipseKeypoints:
    extrema_mask = np.asarray(extrema, dtype=bool)
    vs, us = np.nonzero(extrema_mask)
    count = len(us)
    if count == 0:
        return EllipseKeypoints(
            view_ids=np.empty((0,), dtype=np.int64),
            us=np.empty((0,), dtype=np.int64),
            vs=np.empty((0,), dtype=np.int64),
            scores=np.empty((0,), dtype=np.float32),
            sigmas=np.empty((0,), dtype=np.float32),
            levels=np.empty((0,), dtype=np.int64),
            ellipse_matrices=np.empty((0, 2, 2), dtype=np.float32),
            ellipse_areas=np.empty((0,), dtype=np.float32),
        )
    tensor_level = multichannel_structure_tensor(
        blurred_channels,
        integration_sigma=sampling.structure_sigma_factor * sigma,
        valid_mask=valid_mask,
        channel_weights=structure_weights,
    )
    matrices = np.empty((count, 2, 2), dtype=np.float32)
    areas = np.empty((count,), dtype=np.float32)
    for index, (u, v) in enumerate(zip(us, vs, strict=True)):
        matrices[index], areas[index] = tensor_to_ellipse(
            tensor_level[int(v), int(u)],
            sigma=sigma,
            radius_factor=sampling.ellipse_radius_factor,
            min_area=sampling.min_ellipse_area,
            max_area=sampling.max_ellipse_area,
            max_axis_ratio=sampling.max_axis_ratio,
        )
    return EllipseKeypoints(
        view_ids=np.full((count,), view_id, dtype=np.int64),
        us=us.astype(np.int64),
        vs=vs.astype(np.int64),
        scores=np.abs(response[vs, us]).astype(np.float32),
        sigmas=np.full((count,), sigma, dtype=np.float32),
        levels=np.full((count,), level, dtype=np.int64),
        ellipse_matrices=matrices,
        ellipse_areas=areas,
    )


def render_candidate_ellipses(
    image: Image.Image,
    *,
    keypoints: EllipseKeypoints,
    dominant_channels: np.ndarray,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    channels = np.asarray(dominant_channels, dtype=np.uint8)
    if channels.shape != (image.height, image.width):
        raise ValueError("dominant_channels must match image dimensions")
    # Draw weak candidates first so the strongest responses remain visible.
    order = np.argsort(keypoints.scores, kind="stable")
    for index in order:
        u_int = int(keypoints.us[index])
        v_int = int(keypoints.vs[index])
        outline = ellipse_outline(
            u=float(u_int),
            v=float(v_int),
            ellipse_matrix=keypoints.ellipse_matrices[index],
        )
        color = tuple(int(value) for value in CHANNEL_COLORS[channels[v_int, u_int]])
        draw.line([*outline, outline[0]], fill=color, width=1, joint="curve")
        draw.point((u_int, v_int), fill=(255, 255, 255))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Visualize raw per-scale candidates and same-scale ellipse merging.")
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
    render_ellipse_merging(
        config,
        output_dir=args.output,
        scene_root_override=args.scene_root,
        predictions_override=args.predictions,
        view_selection=args.views,
    )


if __name__ == "__main__":
    main()
