from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from .continuity import global_scene_step_scale
from .types import EllipseKeypoints

INITIALIZATION_METHOD = "dense_lab_log_ellipse_grid_region_pca"


@dataclass(frozen=True)
class EllipseMergeConfig:
    iou_min: float = 0.35
    orientation_max_degrees: float = 30.0
    isotropic_axis_ratio: float = 1.2
    color_delta_e_max: float = 30.0
    continuity_ratio_max: float = 3.0
    grid_cell_factor: float = 5.0
    merged_area_factor_max: float = 1.5
    merged_area_absolute_max: float = 1500.0

    def __post_init__(self) -> None:
        values = (
            self.iou_min,
            self.orientation_max_degrees,
            self.isotropic_axis_ratio,
            self.color_delta_e_max,
            self.continuity_ratio_max,
            self.grid_cell_factor,
            self.merged_area_factor_max,
            self.merged_area_absolute_max,
        )
        if not np.isfinite(values).all():
            raise ValueError("ellipse merge parameters must be finite")
        if not 0.0 <= self.iou_min <= 1.0:
            raise ValueError("sampling.ellipse_merge.iou_min must lie in [0, 1]")
        if not 0.0 <= self.orientation_max_degrees <= 90.0:
            raise ValueError("ellipse merge orientation_max_degrees must lie in [0, 90]")
        if self.isotropic_axis_ratio < 1.0:
            raise ValueError("ellipse merge isotropic_axis_ratio must be at least one")
        if self.color_delta_e_max < 0.0:
            raise ValueError("ellipse merge color_delta_e_max must be non-negative")
        if self.continuity_ratio_max <= 0.0:
            raise ValueError("ellipse merge continuity_ratio_max must be positive")
        if self.grid_cell_factor <= 0.0:
            raise ValueError("ellipse merge grid_cell_factor must be positive")
        if self.merged_area_factor_max < 1.0:
            raise ValueError("ellipse merge merged_area_factor_max must be at least one")
        if self.merged_area_absolute_max <= 0.0:
            raise ValueError("ellipse merge merged_area_absolute_max must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "EllipseMergeConfig":
        config = values or {}
        return cls(
            iou_min=float(config.get("iou_min", cls.iou_min)),
            orientation_max_degrees=float(
                config.get("orientation_max_degrees", cls.orientation_max_degrees)
            ),
            isotropic_axis_ratio=float(
                config.get("isotropic_axis_ratio", cls.isotropic_axis_ratio)
            ),
            color_delta_e_max=float(config.get("color_delta_e_max", cls.color_delta_e_max)),
            continuity_ratio_max=float(
                config.get("continuity_ratio_max", cls.continuity_ratio_max)
            ),
            grid_cell_factor=float(config.get("grid_cell_factor", cls.grid_cell_factor)),
            merged_area_factor_max=float(
                config.get("merged_area_factor_max", cls.merged_area_factor_max)
            ),
            merged_area_absolute_max=float(
                config.get("merged_area_absolute_max", cls.merged_area_absolute_max)
            ),
        )


@dataclass(frozen=True)
class SamplingConfig:
    sigmas: tuple[float, ...] = (1.0, 1.6, 2.5, 4.0, 6.4, 10.0)
    response_threshold: float = 1.0
    max_keypoints_per_view: int = 12000
    structure_sigma_factor: float = 1.5
    ellipse_radius_factor: float = 2.5
    min_ellipse_area: float = 12.0
    max_ellipse_area: float = 800.0
    max_axis_ratio: float = 4.0
    chroma_weight: float = 1.0
    response_mad_epsilon: float = 0.01
    ellipse_merge: EllipseMergeConfig = field(default_factory=EllipseMergeConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "SamplingConfig":
        config = values or {}
        return cls(
            sigmas=tuple(float(value) for value in config.get("sigmas", cls.sigmas)),
            response_threshold=float(config.get("response_threshold", cls.response_threshold)),
            max_keypoints_per_view=int(
                config.get("max_keypoints_per_view", cls.max_keypoints_per_view)
            ),
            structure_sigma_factor=float(
                config.get("structure_sigma_factor", cls.structure_sigma_factor)
            ),
            ellipse_radius_factor=float(
                config.get("ellipse_radius_factor", cls.ellipse_radius_factor)
            ),
            min_ellipse_area=float(config.get("min_ellipse_area", cls.min_ellipse_area)),
            max_ellipse_area=float(config.get("max_ellipse_area", cls.max_ellipse_area)),
            max_axis_ratio=float(config.get("max_axis_ratio", cls.max_axis_ratio)),
            chroma_weight=float(config.get("chroma_weight", cls.chroma_weight)),
            response_mad_epsilon=float(
                config.get("response_mad_epsilon", cls.response_mad_epsilon)
            ),
            ellipse_merge=EllipseMergeConfig.from_mapping(config.get("ellipse_merge")),
        )


@dataclass(frozen=True)
class LoGScaleSpace:
    channel_names: tuple[str, ...]
    guard_sigmas: tuple[float, ...]
    channel_images: np.ndarray
    blurred_channels: np.ndarray
    channel_responses: np.ndarray
    response_scales: np.ndarray
    structure_weights: np.ndarray
    responses: np.ndarray


def detect_multiscale_keypoints(
    *,
    view_id: int,
    image: np.ndarray,
    world_points: np.ndarray,
    sigmas: Sequence[float],
    response_threshold: float,
    max_keypoints: int,
    structure_sigma_factor: float,
    ellipse_radius_factor: float,
    min_ellipse_area: float,
    max_ellipse_area: float,
    max_axis_ratio: float,
    chroma_weight: float = 1.0,
    response_mad_epsilon: float = 0.01,
    ellipse_merge_config: EllipseMergeConfig | None = None,
    image_valid_mask: np.ndarray | None = None,
) -> EllipseKeypoints:
    """Detect multiscale LoG keypoints and attach a structure-tensor ellipse.

    Coordinates use ``(x, y) == (u, v)`` throughout.  Each ellipse is represented
    by a positive-definite matrix E and contains offsets d satisfying
    ``d.T @ inv(E) @ d <= 1``. Keypoints are maxima of the MAD-normalized Lab
    response magnitude.
    """
    sigma_values = _validate_detector_parameters(
        sigmas=sigmas,
        response_threshold=response_threshold,
        max_keypoints=max_keypoints,
        structure_sigma_factor=structure_sigma_factor,
        ellipse_radius_factor=ellipse_radius_factor,
        min_ellipse_area=min_ellipse_area,
        max_ellipse_area=max_ellipse_area,
        max_axis_ratio=max_axis_ratio,
        chroma_weight=chroma_weight,
        response_mad_epsilon=response_mad_epsilon,
    )
    if max_keypoints == 0:
        return _empty_keypoints()

    image_values = np.asarray(image)
    if image_values.ndim not in {2, 3}:
        raise ValueError("image must have shape [H, W] or [H, W, C]")
    image_shape = image_values.shape[:2]
    if world_points.shape != (*image_shape, 3):
        raise ValueError("image and world_points must share H/W dimensions")
    valid = valid_pixel_mask(world_points)
    if image_valid_mask is not None:
        content_valid = np.asarray(image_valid_mask, dtype=bool)
        if content_valid.shape != image_shape:
            raise ValueError("image_valid_mask must have shape [H, W]")
        valid &= content_valid
    scale_space = build_log_scale_space(
        image_values,
        sigmas=sigma_values,
        valid_mask=valid,
        chroma_weight=chroma_weight,
        response_mad_epsilon=response_mad_epsilon,
    )
    responses = scale_space.responses
    extrema = scale_space_maxima(responses)
    extrema &= valid[None, :, :]
    extrema &= np.abs(responses) >= float(response_threshold)

    level_ids, vs, us = np.nonzero(extrema)
    if len(us) == 0:
        return _empty_keypoints()

    scores = np.abs(responses[level_ids, vs, us]).astype(np.float32)
    configured_level_ids = level_ids - 1
    required_levels = np.unique(configured_level_ids)
    tensor_levels = {
        int(level): multichannel_structure_tensor(
            scale_space.blurred_channels[:, int(level) + 1],
            integration_sigma=structure_sigma_factor * sigma_values[int(level)],
            valid_mask=valid,
            channel_weights=scale_space.structure_weights,
        )
        for level in required_levels
    }
    ellipse_matrices = np.empty((len(us), 2, 2), dtype=np.float32)
    ellipse_areas = np.empty((len(us),), dtype=np.float32)
    for index, (u, v, level_id) in enumerate(zip(us, vs, configured_level_ids, strict=True)):
        tensor = tensor_levels[int(level_id)][int(v), int(u)]
        matrix, area = tensor_to_ellipse(
            tensor,
            sigma=sigma_values[int(level_id)],
            radius_factor=ellipse_radius_factor,
            min_area=min_ellipse_area,
            max_area=max_ellipse_area,
            max_axis_ratio=max_axis_ratio,
        )
        ellipse_matrices[index] = matrix
        ellipse_areas[index] = area

    candidates = EllipseKeypoints(
        view_ids=np.full((len(us),), view_id, dtype=np.int64),
        us=us.astype(np.int64),
        vs=vs.astype(np.int64),
        scores=scores,
        sigmas=np.asarray(
            [sigma_values[level] for level in configured_level_ids], dtype=np.float32
        ),
        levels=configured_level_ids,
        ellipse_matrices=ellipse_matrices,
        ellipse_areas=ellipse_areas,
    )
    return merge_same_scale_ellipses(
        candidates,
        normalized_lab=np.moveaxis(scale_space.channel_images, 0, -1),
        world_points=world_points,
        valid_mask=valid,
        config=ellipse_merge_config or EllipseMergeConfig(),
        max_keypoints=max_keypoints,
        max_axis_ratio=max_axis_ratio,
    )


def valid_pixel_mask(world_points: np.ndarray) -> np.ndarray:
    return np.isfinite(world_points).all(axis=-1)


def add_guard_scales(sigmas: Sequence[float]) -> tuple[float, ...]:
    """Add geometric guard levels around the configured usable LoD range."""
    values = tuple(float(value) for value in sigmas)
    lower_ratio = values[1] / values[0]
    upper_ratio = values[-1] / values[-2]
    return (values[0] / lower_ratio, *values, values[-1] * upper_ratio)


def build_log_scale_space(
    image: np.ndarray,
    *,
    sigmas: Sequence[float],
    valid_mask: np.ndarray,
    chroma_weight: float,
    response_mad_epsilon: float,
) -> LoGScaleSpace:
    """Build the MAD-normalized multichannel Lab LoG scale space."""
    sigma_values = tuple(float(value) for value in sigmas)
    if (
        len(sigma_values) < 3
        or not np.isfinite(sigma_values).all()
        or any(value <= 0.0 for value in sigma_values)
        or any(left >= right for left, right in zip(sigma_values, sigma_values[1:]))
    ):
        raise ValueError("sigmas must contain at least three strictly increasing values")
    if not np.isfinite(chroma_weight) or chroma_weight < 0.0:
        raise ValueError("chroma_weight must be finite and non-negative")
    if not np.isfinite(response_mad_epsilon) or response_mad_epsilon <= 0.0:
        raise ValueError("response_mad_epsilon must be finite and positive")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("valid_mask must have shape [H, W]")
    lab = rgb_to_normalized_lab(image)
    channel_images = np.moveaxis(lab, -1, 0)
    channel_names = ("L", "a", "b")
    structure_weights = np.asarray([1.0, chroma_weight, chroma_weight], dtype=np.float32)
    if channel_images.shape[1:] != valid.shape:
        raise ValueError("valid_mask must match image H/W dimensions")

    guard_sigmas = add_guard_scales(sigma_values)
    blurred = np.stack(
        [
            np.stack(
                [masked_gaussian_blur(channel, sigma, valid) for sigma in guard_sigmas],
                axis=0,
            )
            for channel in channel_images
        ],
        axis=0,
    ).astype(np.float32)
    channel_responses = np.stack(
        [
            np.stack(
                [
                    scale_normalized_laplacian(level, sigma)
                    for level, sigma in zip(channel_levels, guard_sigmas, strict=True)
                ],
                axis=0,
            )
            for channel_levels in blurred
        ],
        axis=0,
    ).astype(np.float32)

    response_scales = robust_channel_response_scales(
        channel_responses[:, 1:-1],
        valid_mask=valid,
        epsilon=response_mad_epsilon,
    )
    normalized = channel_responses / response_scales[:, None, None, None]
    responses = np.sqrt(
        np.sum(
            structure_weights[:, None, None, None] * normalized * normalized,
            axis=0,
        )
    ).astype(np.float32)

    return LoGScaleSpace(
        channel_names=channel_names,
        guard_sigmas=guard_sigmas,
        channel_images=channel_images.astype(np.float32),
        blurred_channels=blurred,
        channel_responses=channel_responses,
        response_scales=response_scales,
        structure_weights=structure_weights,
        responses=responses.astype(np.float32),
    )


def robust_channel_response_scales(
    responses: np.ndarray,
    *,
    valid_mask: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Return one robust scale per channel without disturbing cross-scale ratios."""
    values = np.asarray(responses, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if values.ndim != 4 or values.shape[2:] != valid.shape:
        raise ValueError("responses must have shape [C, S, H, W] matching valid_mask")
    scales = np.empty((values.shape[0],), dtype=np.float32)
    for channel in range(values.shape[0]):
        samples = values[channel, :, valid]
        samples = samples[np.isfinite(samples)]
        if not len(samples):
            scales[channel] = float(epsilon)
            continue
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        scales[channel] = mad + float(epsilon)
    return scales


def scale_space_maxima(responses: np.ndarray) -> np.ndarray:
    """Return strict maxima in a 3x3x3 neighborhood for nonnegative responses."""
    values = np.asarray(responses, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("responses must have shape [S, H, W]")
    scales, height, width = values.shape
    if scales < 3:
        raise ValueError("At least three LoG scales are required")
    neighbor_max = np.full_like(values, -np.inf)
    padded = np.pad(values, ((1, 1), (1, 1), (1, 1)), mode="edge")
    for ds in range(3):
        for dy in range(3):
            for dx in range(3):
                if ds == 1 and dy == 1 and dx == 1:
                    continue
                neighbor = padded[ds : ds + scales, dy : dy + height, dx : dx + width]
                neighbor_max = np.maximum(neighbor_max, neighbor)
    maxima = values > neighbor_max
    maxima[0] = False
    maxima[-1] = False
    if height:
        maxima[:, 0] = False
        maxima[:, -1] = False
    if width:
        maxima[:, :, 0] = False
        maxima[:, :, -1] = False
    return maxima


def merge_same_scale_ellipses(
    candidates: EllipseKeypoints,
    *,
    normalized_lab: np.ndarray,
    world_points: np.ndarray,
    valid_mask: np.ndarray,
    config: EllipseMergeConfig,
    max_keypoints: int,
    max_axis_ratio: float,
) -> EllipseKeypoints:
    """Merge only geometrically and photometrically compatible same-scale ellipses."""
    if len(candidates) == 0:
        return candidates
    lab_image = np.asarray(normalized_lab, dtype=np.float32)
    points = np.asarray(world_points, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    image_shape = valid.shape
    if lab_image.shape != (*image_shape, 3):
        raise ValueError("normalized_lab must have shape [H, W, 3]")
    if points.shape != (*image_shape, 3):
        raise ValueError("world_points must match the merge image shape")
    if max_keypoints < 0:
        raise ValueError("max_keypoints must be non-negative")
    if not np.isfinite(max_axis_ratio):
        raise ValueError("ellipse merge bounds must be finite")
    if max_keypoints == 0:
        return _empty_keypoints()

    centers = np.stack([candidates.us, candidates.vs], axis=1).astype(np.float64)
    matrices = np.asarray(candidates.ellipse_matrices, dtype=np.float64)
    inverses = np.linalg.inv(matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    axis_ratios = np.sqrt(
        np.maximum(eigenvalues[:, 1], 0.0) / np.maximum(eigenvalues[:, 0], 1.0e-20)
    )
    long_directions = eigenvectors[:, :, 1]
    extents = np.sqrt(np.maximum(np.diagonal(matrices, axis1=1, axis2=2), 0.0))
    bounds = np.stack(
        [
            centers[:, 0] - extents[:, 0],
            centers[:, 1] - extents[:, 1],
            centers[:, 0] + extents[:, 0],
            centers[:, 1] + extents[:, 1],
        ],
        axis=1,
    )
    lab = lab_image[candidates.vs, candidates.us].astype(np.float64)
    lab *= np.asarray([100.0, 128.0, 128.0], dtype=np.float64)
    continuity_reference_scale = global_scene_step_scale(points, valid, neighbors=8)

    union_find = _EllipseUnionFind(len(candidates))
    component_members = {index: [index] for index in range(len(candidates))}
    orientation_cosine_min = float(np.cos(np.deg2rad(config.orientation_max_degrees)))
    compatible_edges: list[tuple[float, int, int]] = []
    for level in np.unique(candidates.levels):
        indices = np.flatnonzero(candidates.levels == level)
        sigma = float(candidates.sigmas[indices[0]])
        cell_size = max(config.grid_cell_factor * sigma, 1.0)
        for left, right in _ellipse_spatial_pairs(indices, bounds, cell_size=cell_size):
            similarity = _ellipse_pair_similarity(
                left,
                right,
                centers=centers,
                inverses=inverses,
                axis_ratios=axis_ratios,
                long_directions=long_directions,
                bounds=bounds,
                lab=lab,
                world_points=points,
                valid_mask=valid,
                continuity_reference_scale=continuity_reference_scale,
                image_shape=image_shape,
                orientation_cosine_min=orientation_cosine_min,
                config=config,
            )
            if similarity is None:
                continue
            compatible_edges.append((similarity, left, right))

    for _similarity, left, right in sorted(
        compatible_edges,
        key=lambda edge: (-edge[0], edge[1], edge[2]),
    ):
        left_root = union_find.find(left)
        right_root = union_find.find(right)
        if left_root == right_root:
            continue
        combined = component_members[left_root] + component_members[right_root]
        if (
            _merge_ellipse_component(
                candidates,
                combined,
                config=config,
                max_axis_ratio=max_axis_ratio,
            )
            is None
        ):
            continue
        new_root = union_find.union(left_root, right_root)
        removed_root = right_root if new_root == left_root else left_root
        component_members[new_root] = combined
        del component_members[removed_root]

    output = _EllipseKeypointBuilder(candidates)
    for component in sorted(component_members.values(), key=min):
        if len(component) == 1:
            output.append_original(candidates, component[0])
            continue
        merged = _merge_ellipse_component(
            candidates,
            component,
            config=config,
            max_axis_ratio=max_axis_ratio,
        )
        if merged is None:
            for index in component:
                output.append_original(candidates, index)
            continue
        output.append_merged(*merged)

    merged_keypoints = output.build()
    order = np.argsort(-merged_keypoints.scores, kind="stable")
    order = order[:max_keypoints]
    return _select_keypoints(merged_keypoints, order)


def scene_path_is_continuous(
    left_center: np.ndarray,
    right_center: np.ndarray,
    *,
    world_points: np.ndarray,
    valid_mask: np.ndarray,
    reference_scale: float,
    ratio_max: float,
) -> bool:
    """Check a pixel path against one fixed per-view 3D step limit.

    The line between the two ellipse centers is sampled as an 8-connected pixel
    path. Every pixel must be valid, and every adjacent 3D step must be no larger
    than ``ratio_max * reference_scale``. The reference is the robust median over
    all valid neighboring pixels in the view, so a local discontinuity cannot
    inflate its own tolerance.
    """
    points = np.asarray(world_points, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if points.shape != (*valid.shape, 3):
        raise ValueError("path continuity inputs must share H/W dimensions")
    if not np.isfinite(reference_scale) or reference_scale < 0.0:
        return False
    if not np.isfinite(ratio_max) or ratio_max <= 0.0:
        raise ValueError("ratio_max must be finite and positive")

    start = np.rint(np.asarray(left_center, dtype=np.float64)).astype(np.int64)
    end = np.rint(np.asarray(right_center, dtype=np.float64)).astype(np.int64)
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("ellipse centers must have shape [2]")
    delta = end - start
    segment_count = int(np.max(np.abs(delta)))
    if segment_count == 0:
        path_x = np.asarray([start[0]], dtype=np.int64)
        path_y = np.asarray([start[1]], dtype=np.int64)
    else:
        path_x = np.rint(np.linspace(start[0], end[0], segment_count + 1)).astype(np.int64)
        path_y = np.rint(np.linspace(start[1], end[1], segment_count + 1)).astype(np.int64)
        keep = np.ones((len(path_x),), dtype=bool)
        keep[1:] = (path_x[1:] != path_x[:-1]) | (path_y[1:] != path_y[:-1])
        path_x = path_x[keep]
        path_y = path_y[keep]

    height, width = valid.shape
    if (
        np.any(path_x < 0)
        or np.any(path_x >= width)
        or np.any(path_y < 0)
        or np.any(path_y >= height)
    ):
        return False
    if not np.all(valid[path_y, path_x]):
        return False
    path_points = points[path_y, path_x]
    if not np.isfinite(path_points).all():
        return False
    if len(path_x) == 1:
        return True

    pixel_steps = np.hypot(np.diff(path_x), np.diff(path_y))
    scene_steps = np.linalg.norm(np.diff(path_points, axis=0), axis=1) / pixel_steps
    tolerance = float(ratio_max) * max(float(reference_scale), 1.0e-12)
    return bool(np.all(np.isfinite(scene_steps)) and np.all(scene_steps <= tolerance))


def ellipse_intersection_over_union(
    left_center: np.ndarray,
    left_inverse: np.ndarray,
    right_center: np.ndarray,
    right_inverse: np.ndarray,
    *,
    image_shape: tuple[int, int],
    bounds: np.ndarray,
) -> float:
    """Return the discrete IoU of two ellipse supports inside the image."""
    height, width = image_shape
    x0 = max(0, int(math.floor(min(float(bounds[0, 0]), float(bounds[1, 0])))))
    y0 = max(0, int(math.floor(min(float(bounds[0, 1]), float(bounds[1, 1])))))
    x1 = min(width, int(math.ceil(max(float(bounds[0, 2]), float(bounds[1, 2])))) + 1)
    y1 = min(height, int(math.ceil(max(float(bounds[0, 3]), float(bounds[1, 3])))) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    ys, xs = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64),
        np.arange(x0, x1, dtype=np.float64),
        indexing="ij",
    )
    left_x = xs - float(left_center[0])
    left_y = ys - float(left_center[1])
    right_x = xs - float(right_center[0])
    right_y = ys - float(right_center[1])
    left_quadratic = (
        left_inverse[0, 0] * left_x * left_x
        + 2.0 * left_inverse[0, 1] * left_x * left_y
        + left_inverse[1, 1] * left_y * left_y
    )
    right_quadratic = (
        right_inverse[0, 0] * right_x * right_x
        + 2.0 * right_inverse[0, 1] * right_x * right_y
        + right_inverse[1, 1] * right_y * right_y
    )
    left_mask = left_quadratic <= 1.0 + 1.0e-6
    right_mask = right_quadratic <= 1.0 + 1.0e-6
    union = int(np.count_nonzero(left_mask | right_mask))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(left_mask & right_mask)) / union


def _ellipse_pair_similarity(
    left: int,
    right: int,
    *,
    centers: np.ndarray,
    inverses: np.ndarray,
    axis_ratios: np.ndarray,
    long_directions: np.ndarray,
    bounds: np.ndarray,
    lab: np.ndarray,
    world_points: np.ndarray,
    valid_mask: np.ndarray,
    continuity_reference_scale: float,
    image_shape: tuple[int, int],
    orientation_cosine_min: float,
    config: EllipseMergeConfig,
) -> float | None:
    if (
        bounds[left, 2] < bounds[right, 0]
        or bounds[right, 2] < bounds[left, 0]
        or bounds[left, 3] < bounds[right, 1]
        or bounds[right, 3] < bounds[left, 1]
    ):
        return None
    if (
        axis_ratios[left] >= config.isotropic_axis_ratio
        and axis_ratios[right] >= config.isotropic_axis_ratio
        and abs(float(np.dot(long_directions[left], long_directions[right])))
        < orientation_cosine_min
    ):
        return None
    if float(np.linalg.norm(lab[left] - lab[right])) > config.color_delta_e_max:
        return None

    iou = ellipse_intersection_over_union(
        centers[left],
        inverses[left],
        centers[right],
        inverses[right],
        image_shape=image_shape,
        bounds=bounds[[left, right]],
    )
    if iou < config.iou_min:
        return None
    if not scene_path_is_continuous(
        centers[left],
        centers[right],
        world_points=world_points,
        valid_mask=valid_mask,
        reference_scale=continuity_reference_scale,
        ratio_max=config.continuity_ratio_max,
    ):
        return None
    return iou


def _ellipse_spatial_pairs(
    indices: np.ndarray,
    bounds: np.ndarray,
    *,
    cell_size: float,
) -> list[tuple[int, int]]:
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in np.asarray(indices, dtype=np.int64):
        x0 = int(math.floor(float(bounds[index, 0]) / cell_size))
        y0 = int(math.floor(float(bounds[index, 1]) / cell_size))
        x1 = int(math.floor(float(bounds[index, 2]) / cell_size))
        y1 = int(math.floor(float(bounds[index, 3]) / cell_size))
        for cell_y in range(y0, y1 + 1):
            for cell_x in range(x0, x1 + 1):
                cells[(cell_x, cell_y)].append(int(index))
    pairs: set[tuple[int, int]] = set()
    for members in cells.values():
        for left, right in combinations(members, 2):
            pairs.add((min(left, right), max(left, right)))
    return sorted(pairs)


def _merge_ellipse_component(
    candidates: EllipseKeypoints,
    component: list[int],
    *,
    config: EllipseMergeConfig,
    max_axis_ratio: float,
) -> tuple[int, np.ndarray, float, float] | None:
    indices = np.asarray(component, dtype=np.int64)
    raw_weights = np.maximum(candidates.scores[indices], 1.0e-6).astype(np.float64)
    weights = raw_weights / np.sum(raw_weights)
    centers = np.stack([candidates.us[indices], candidates.vs[indices]], axis=1).astype(np.float64)
    centroid = np.sum(centers * weights[:, None], axis=0)
    anchor_position = int(np.argmin(np.sum((centers - centroid[None, :]) ** 2, axis=1)))
    anchor = int(indices[anchor_position])
    anchor_center = centers[anchor_position]
    offsets = centers - anchor_center[None, :]
    ellipse_support_covariances = candidates.ellipse_matrices[indices].astype(np.float64) / 4.0
    second_moments = ellipse_support_covariances + offsets[:, :, None] @ offsets[:, None, :]
    merged_covariance = np.sum(second_moments * weights[:, None, None], axis=0)
    merged_matrix = 4.0 * merged_covariance
    merged_matrix = 0.5 * (merged_matrix + merged_matrix.T)
    eigenvalues = np.linalg.eigvalsh(merged_matrix)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0):
        return None
    area = math.pi * math.sqrt(float(np.prod(eigenvalues)))
    axis_ratio = math.sqrt(float(eigenvalues[-1] / eigenvalues[0]))
    member_area = float(np.max(candidates.ellipse_areas[indices]))
    area_limit = min(
        config.merged_area_factor_max * member_area,
        config.merged_area_absolute_max,
    )
    if area > area_limit + 1.0e-6 or axis_ratio > max_axis_ratio + 1.0e-6:
        return None
    return anchor, merged_matrix.astype(np.float32), area, float(np.max(candidates.scores[indices]))


class _EllipseUnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros((size,), dtype=np.uint8)

    def find(self, value: int) -> int:
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while value != root:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root


class _EllipseKeypointBuilder:
    def __init__(self, candidates: EllipseKeypoints) -> None:
        self.candidates = candidates
        self.view_ids: list[int] = []
        self.us: list[int] = []
        self.vs: list[int] = []
        self.scores: list[float] = []
        self.sigmas: list[float] = []
        self.levels: list[int] = []
        self.matrices: list[np.ndarray] = []
        self.areas: list[float] = []

    def append_original(self, candidates: EllipseKeypoints, index: int) -> None:
        self.append_merged(
            index,
            candidates.ellipse_matrices[index],
            float(candidates.ellipse_areas[index]),
            float(candidates.scores[index]),
        )

    def append_merged(
        self,
        anchor: int,
        matrix: np.ndarray,
        area: float,
        score: float,
    ) -> None:
        candidates = self.candidates
        self.view_ids.append(int(candidates.view_ids[anchor]))
        self.us.append(int(candidates.us[anchor]))
        self.vs.append(int(candidates.vs[anchor]))
        self.scores.append(score)
        self.sigmas.append(float(candidates.sigmas[anchor]))
        self.levels.append(int(candidates.levels[anchor]))
        self.matrices.append(np.asarray(matrix, dtype=np.float32))
        self.areas.append(area)

    def build(self) -> EllipseKeypoints:
        if not self.us:
            return _empty_keypoints()
        return EllipseKeypoints(
            view_ids=np.asarray(self.view_ids, dtype=np.int64),
            us=np.asarray(self.us, dtype=np.int64),
            vs=np.asarray(self.vs, dtype=np.int64),
            scores=np.asarray(self.scores, dtype=np.float32),
            sigmas=np.asarray(self.sigmas, dtype=np.float32),
            levels=np.asarray(self.levels, dtype=np.int64),
            ellipse_matrices=np.asarray(self.matrices, dtype=np.float32),
            ellipse_areas=np.asarray(self.areas, dtype=np.float32),
        )


def _select_keypoints(keypoints: EllipseKeypoints, indices: np.ndarray) -> EllipseKeypoints:
    selected = np.asarray(indices, dtype=np.int64)
    return EllipseKeypoints(
        view_ids=keypoints.view_ids[selected],
        us=keypoints.us[selected],
        vs=keypoints.vs[selected],
        scores=keypoints.scores[selected],
        sigmas=keypoints.sigmas[selected],
        levels=keypoints.levels[selected],
        ellipse_matrices=keypoints.ellipse_matrices[selected],
        ellipse_areas=keypoints.ellipse_areas[selected],
    )


def multichannel_structure_tensor(
    image: np.ndarray,
    *,
    integration_sigma: float,
    valid_mask: np.ndarray | None = None,
    channel_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a color structure tensor by summing weighted channel gradients."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError("multichannel image must have shape [C, H, W]")
    weights = (
        np.ones((values.shape[0],), dtype=np.float32)
        if channel_weights is None
        else np.asarray(channel_weights, dtype=np.float32)
    )
    if weights.shape != (values.shape[0],):
        raise ValueError("channel_weights must have shape [C]")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("channel_weights must be finite and non-negative")
    dx = np.empty_like(values)
    dy = np.empty_like(values)
    for channel in range(values.shape[0]):
        dx[channel], dy[channel] = image_gradients(values[channel])
    j_xx_raw = np.sum(weights[:, None, None] * dx * dx, axis=0)
    j_xy_raw = np.sum(weights[:, None, None] * dx * dy, axis=0)
    j_yy_raw = np.sum(weights[:, None, None] * dy * dy, axis=0)
    j_xx = masked_gaussian_blur(j_xx_raw, integration_sigma, valid_mask)
    j_xy = masked_gaussian_blur(j_xy_raw, integration_sigma, valid_mask)
    j_yy = masked_gaussian_blur(j_yy_raw, integration_sigma, valid_mask)
    output = np.empty((*values.shape[1:], 2, 2), dtype=np.float32)
    output[..., 0, 0] = j_xx
    output[..., 0, 1] = j_xy
    output[..., 1, 0] = j_xy
    output[..., 1, 1] = j_yy
    return output


def tensor_to_ellipse(
    tensor: np.ndarray,
    *,
    sigma: float,
    radius_factor: float,
    min_area: float,
    max_area: float,
    max_axis_ratio: float,
) -> tuple[np.ndarray, float]:
    """Map structure eigenvectors to an area-bounded anisotropic ellipse."""
    matrix = np.asarray(tensor, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    regularizer = max(float(np.trace(matrix)) * 1.0e-3, 1.0e-12)
    ratio = math.sqrt(float(eigenvalues[1] + regularizer) / float(eigenvalues[0] + regularizer))
    ratio = float(np.clip(ratio, 1.0, max_axis_ratio))

    base_radius = max(float(radius_factor) * float(sigma), 0.5)
    # eigh is ascending: the long axis follows the weak-gradient eigenvector.
    axes = np.asarray(
        [base_radius * math.sqrt(ratio), base_radius / math.sqrt(ratio)],
        dtype=np.float64,
    )
    area = math.pi * float(axes[0] * axes[1])
    target_area = float(np.clip(area, min_area, max_area))
    axes *= math.sqrt(target_area / max(area, 1.0e-20))

    ellipse = eigenvectors @ np.diag(axes * axes) @ eigenvectors.T
    ellipse = 0.5 * (ellipse + ellipse.T)
    final_area = math.pi * math.sqrt(max(float(np.linalg.det(ellipse)), 0.0))
    return ellipse.astype(np.float32), float(final_area)


def scale_normalized_laplacian(image: np.ndarray, sigma: float) -> np.ndarray:
    padded = _pad_spatial(np.asarray(image, dtype=np.float32), 1)
    center = padded[1:-1, 1:-1]
    laplacian = (
        padded[1:-1, 2:] + padded[1:-1, :-2] + padded[2:, 1:-1] + padded[:-2, 1:-1] - 4.0 * center
    )
    return (float(sigma) ** 2 * laplacian).astype(np.float32)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("gaussian_blur expects a 2D image")
    if sigma <= 0.0:
        raise ValueError("Gaussian sigma must be positive")
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel = (kernel / np.sum(kernel)).astype(np.float32)

    padded_x = _pad_axis(values, radius, axis=1)
    horizontal = np.zeros_like(values)
    for index, weight in enumerate(kernel):
        horizontal += weight * padded_x[:, index : index + values.shape[1]]

    padded_y = _pad_axis(horizontal, radius, axis=0)
    output = np.zeros_like(values)
    for index, weight in enumerate(kernel):
        output += weight * padded_y[index : index + values.shape[0], :]
    return output


def masked_gaussian_blur(
    image: np.ndarray,
    sigma: float,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    if valid_mask is None:
        return gaussian_blur(image, sigma)
    mask = np.asarray(valid_mask, dtype=np.float32)
    if mask.shape != np.asarray(image).shape:
        raise ValueError("valid_mask must match the image shape")
    if np.all(mask):
        return gaussian_blur(image, sigma)
    numerator = gaussian_blur(np.asarray(image, dtype=np.float32) * mask, sigma)
    denominator = gaussian_blur(mask, sigma)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-6,
    )


def image_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(image, dtype=np.float32)
    padded = _pad_spatial(values, 1)
    dx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    dy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return dx.astype(np.float32), dy.astype(np.float32)


def rgb_to_normalized_lab(image: np.ndarray) -> np.ndarray:
    """Convert sRGB in [0, 1] to D65 CIELAB normalized to comparable ranges."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 3:
        raise ValueError("Lab conversion expects image shape [H, W, 3]")
    if not np.isfinite(values[..., :3]).all():
        raise ValueError("Lab conversion expects finite RGB values")
    srgb = np.clip(values[..., :3], 0.0, 1.0).astype(np.float64)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    transform = np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ transform.T
    xyz /= np.asarray([0.95047, 1.0, 1.08883], dtype=np.float64)
    delta = 6.0 / 29.0
    f_xyz = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lightness = 116.0 * f_xyz[..., 1] - 16.0
    a_channel = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b_channel = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return np.stack(
        [lightness / 100.0, a_channel / 128.0, b_channel / 128.0],
        axis=-1,
    ).astype(np.float32)


def normalized_lab_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert normalized D65 CIELAB back to clipped sRGB in [0, 1]."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("Lab image must have shape [H, W, 3]")
    lightness = values[..., 0].astype(np.float64) * 100.0
    a_channel = values[..., 1].astype(np.float64) * 128.0
    b_channel = values[..., 2].astype(np.float64) * 128.0
    f_y = (lightness + 16.0) / 116.0
    f_x = f_y + a_channel / 500.0
    f_z = f_y - b_channel / 200.0
    f_xyz = np.stack([f_x, f_y, f_z], axis=-1)
    delta = 6.0 / 29.0
    xyz = np.where(
        f_xyz > delta,
        f_xyz**3,
        3.0 * delta**2 * (f_xyz - 4.0 / 29.0),
    )
    xyz *= np.asarray([0.95047, 1.0, 1.08883], dtype=np.float64)
    transform = np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    linear = xyz @ np.linalg.inv(transform).T
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def _validate_detector_parameters(
    *,
    sigmas: Sequence[float],
    response_threshold: float,
    max_keypoints: int,
    structure_sigma_factor: float,
    ellipse_radius_factor: float,
    min_ellipse_area: float,
    max_ellipse_area: float,
    max_axis_ratio: float,
    chroma_weight: float,
    response_mad_epsilon: float,
) -> tuple[float, ...]:
    sigma_values = tuple(float(value) for value in sigmas)
    if (
        len(sigma_values) < 3
        or not np.isfinite(sigma_values).all()
        or any(value <= 0.0 for value in sigma_values)
    ):
        raise ValueError("sampling.sigmas must contain at least three positive values")
    if any(left >= right for left, right in zip(sigma_values, sigma_values[1:])):
        raise ValueError("sampling.sigmas must be strictly increasing")
    floating_parameters = (
        response_threshold,
        structure_sigma_factor,
        ellipse_radius_factor,
        min_ellipse_area,
        max_ellipse_area,
        max_axis_ratio,
        chroma_weight,
        response_mad_epsilon,
    )
    if not np.isfinite(floating_parameters).all():
        raise ValueError("sampling parameters must be finite")
    if response_threshold < 0.0:
        raise ValueError("response_threshold must be non-negative")
    if max_keypoints < 0:
        raise ValueError("max_keypoints must be non-negative")
    if structure_sigma_factor <= 0.0 or ellipse_radius_factor <= 0.0:
        raise ValueError("scale factors must be positive")
    if min_ellipse_area <= 0.0 or max_ellipse_area < min_ellipse_area:
        raise ValueError("ellipse area bounds are invalid")
    if max_axis_ratio < 1.0:
        raise ValueError("max_axis_ratio must be at least one")
    if chroma_weight < 0.0:
        raise ValueError("sampling.chroma_weight must be non-negative")
    if response_mad_epsilon <= 0.0:
        raise ValueError("sampling.response_mad_epsilon must be positive")
    return sigma_values


def _empty_keypoints() -> EllipseKeypoints:
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


def _pad_spatial(values: np.ndarray, radius: int) -> np.ndarray:
    padded = _pad_axis(values, radius, axis=1)
    return _pad_axis(padded, radius, axis=0)


def _pad_axis(values: np.ndarray, radius: int, *, axis: int) -> np.ndarray:
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    mode = "reflect" if values.shape[axis] > 1 else "edge"
    return np.pad(values, padding, mode=mode)
