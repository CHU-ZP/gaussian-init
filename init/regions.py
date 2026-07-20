from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .continuity import global_scene_step_scale, neighbor_offsets, paired_slices
from .types import EllipseKeypoints


@dataclass(frozen=True)
class PixelRegion:
    """One image-space support region in local bounding-box coordinates."""

    x0: int
    y0: int
    mask: np.ndarray
    source: str
    score: float

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or mask.size == 0 or not np.any(mask):
            raise ValueError("region mask must be a non-empty two-dimensional mask")
        if not self.source:
            raise ValueError("region source must be non-empty")
        if not np.isfinite(self.score):
            raise ValueError("region score must be finite")
        object.__setattr__(self, "mask", mask)

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    @property
    def support_count(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class RegionFitResults:
    means: np.ndarray
    covariances: np.ndarray
    mean_colors: np.ndarray
    valid_counts: np.ndarray
    support_counts: np.ndarray
    valid: np.ndarray
    continuity_reference_scale: float


def ellipse_regions(keypoints: EllipseKeypoints) -> list[PixelRegion]:
    """Rasterize final keypoint ellipses as generic image-space regions."""
    regions: list[PixelRegion] = []
    for u, v, matrix, score in zip(
        keypoints.us,
        keypoints.vs,
        keypoints.ellipse_matrices,
        keypoints.scores,
        strict=True,
    ):
        ellipse = np.asarray(matrix, dtype=np.float64)
        extent_x = int(np.ceil(np.sqrt(max(float(ellipse[0, 0]), 0.0))))
        extent_y = int(np.ceil(np.sqrt(max(float(ellipse[1, 1]), 0.0))))
        ys, xs = np.meshgrid(
            np.arange(-extent_y, extent_y + 1, dtype=np.float64),
            np.arange(-extent_x, extent_x + 1, dtype=np.float64),
            indexing="ij",
        )
        inverse = np.linalg.inv(ellipse)
        quadratic = (
            inverse[0, 0] * xs * xs
            + 2.0 * inverse[0, 1] * xs * ys
            + inverse[1, 1] * ys * ys
        )
        regions.append(
            PixelRegion(
                x0=int(u) - extent_x,
                y0=int(v) - extent_y,
                mask=quadratic <= 1.0 + 1.0e-6,
                source="ellipse",
                score=float(score),
            )
        )
    return regions


def fit_regions(
    world_points: np.ndarray,
    regions: Sequence[PixelRegion],
    *,
    colors: np.ndarray | None = None,
    image_valid_mask: np.ndarray | None = None,
    min_valid_points: int,
    min_valid_fraction: float,
    continuity_neighbors: int,
    continuity_ratio_max: float,
    device: str,
    pixel_budget: int,
) -> RegionFitResults:
    """Fit one 3D mean and covariance to each arbitrary image-space region.

    The region is intersected with content-valid pixels having finite 3D
    coordinates. Image-neighbor edges are accepted using one fixed per-view 3D
    step threshold, and only the largest resulting 3D-connected component is
    retained. Mean, RGB and covariance are then estimated from that component.
    """
    import torch

    points_np = np.asarray(world_points, dtype=np.float32)
    color_np = None if colors is None else np.asarray(colors, dtype=np.float32)
    image_valid_np = (
        np.ones(points_np.shape[:2], dtype=bool)
        if image_valid_mask is None
        else np.asarray(image_valid_mask, dtype=bool)
    )
    _validate_fit_inputs(
        points_np,
        regions,
        colors=color_np,
        image_valid_mask=image_valid_np,
        min_valid_points=min_valid_points,
        min_valid_fraction=min_valid_fraction,
        continuity_neighbors=continuity_neighbors,
        continuity_ratio_max=continuity_ratio_max,
        pixel_budget=pixel_budget,
    )

    continuity_valid = image_valid_np.copy()
    continuity_valid &= np.isfinite(points_np).all(axis=-1)
    continuity_reference_scale = global_scene_step_scale(
        points_np,
        continuity_valid,
        neighbors=continuity_neighbors,
    )

    count = len(regions)
    output_means = np.full((count, 3), np.nan, dtype=np.float32)
    output_covariances = np.full((count, 3, 3), np.nan, dtype=np.float32)
    output_colors = np.full((count, 3), np.nan, dtype=np.float32)
    output_valid_counts = np.zeros((count,), dtype=np.int64)
    output_support_counts = np.asarray(
        [region.support_count for region in regions], dtype=np.int64
    )
    output_valid = np.zeros((count,), dtype=bool)
    if count == 0:
        return RegionFitResults(
            means=output_means,
            covariances=output_covariances,
            mean_colors=output_colors,
            valid_counts=output_valid_counts,
            support_counts=output_support_counts,
            valid=output_valid,
            continuity_reference_scale=continuity_reference_scale,
        )

    torch_device = _resolve_device(device)
    points = torch.as_tensor(points_np, device=torch_device)
    color_values = None if color_np is None else torch.as_tensor(color_np, device=torch_device)
    image_valid = torch.as_tensor(image_valid_np, device=torch_device)
    height, width = points_np.shape[:2]

    shape_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, region in enumerate(regions):
        shape_groups[region.mask.shape].append(index)

    for (region_height, region_width), group in shape_groups.items():
        pixels_per_region = region_height * region_width
        if pixels_per_region > pixel_budget:
            raise ValueError(
                "A region bounding window exceeds covariance.pixel_budget; "
                "increase the budget or reduce the region area limit"
            )
        chunk_size = max(1, int(pixel_budget) // pixels_per_region)
        local_y, local_x = torch.meshgrid(
            torch.arange(region_height, dtype=torch.int64, device=torch_device),
            torch.arange(region_width, dtype=torch.int64, device=torch_device),
            indexing="ij",
        )
        for start in range(0, len(group), chunk_size):
            indices_np = np.asarray(group[start : start + chunk_size], dtype=np.int64)
            masks = torch.as_tensor(
                np.stack([regions[index].mask for index in indices_np]),
                device=torch_device,
            )
            x0 = torch.as_tensor(
                [regions[index].x0 for index in indices_np],
                dtype=torch.int64,
                device=torch_device,
            )
            y0 = torch.as_tensor(
                [regions[index].y0 for index in indices_np],
                dtype=torch.int64,
                device=torch_device,
            )
            xs = x0[:, None, None] + local_x[None]
            ys = y0[:, None, None] + local_y[None]
            in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            flat_indices = torch.clamp(ys, 0, height - 1) * width + torch.clamp(
                xs, 0, width - 1
            )
            gathered_points = points.reshape(-1, 3)[flat_indices]
            gathered_image_valid = image_valid.reshape(-1)[flat_indices]
            candidate = masks & in_bounds & gathered_image_valid
            candidate &= torch.isfinite(gathered_points).all(dim=-1)

            connected = _largest_connected_component(
                gathered_points,
                candidate,
                continuity_neighbors=continuity_neighbors,
                continuity_ratio_max=continuity_ratio_max,
                continuity_reference_scale=continuity_reference_scale,
            )
            valid_counts = connected.sum(dim=(1, 2))
            weights = connected.to(points.dtype)
            weight_sum = weights.sum(dim=(1, 2))
            clean_points = torch.where(
                connected[..., None], gathered_points, torch.zeros_like(gathered_points)
            )
            means = (clean_points * weights[..., None]).sum(dim=(1, 2))
            means /= torch.clamp(weight_sum, min=1.0e-20)[:, None]
            centered = torch.where(
                connected[..., None],
                gathered_points - means[:, None, None, :],
                torch.zeros_like(gathered_points),
            )
            flat_centered = centered.reshape(len(indices_np), -1, 3)
            flat_weights = weights.reshape(len(indices_np), -1)
            scatter = torch.bmm(
                (flat_centered * flat_weights[..., None]).transpose(1, 2),
                flat_centered,
            )
            covariances = scatter / torch.clamp(weight_sum, min=1.0e-20)[:, None, None]
            if color_values is not None:
                gathered_colors = color_values.reshape(-1, 3)[flat_indices]
                clean_colors = torch.where(
                    connected[..., None], gathered_colors, torch.zeros_like(gathered_colors)
                )
                mean_colors = (clean_colors * weights[..., None]).sum(dim=(1, 2))
                mean_colors /= torch.clamp(weight_sum, min=1.0e-20)[:, None]
            else:
                mean_colors = torch.full_like(means, float("nan"))

            support_counts = masks.sum(dim=(1, 2))
            valid_fraction = valid_counts.to(points.dtype) / torch.clamp(
                support_counts, min=1
            ).to(points.dtype)
            accepted = valid_counts >= int(min_valid_points)
            accepted &= valid_fraction >= float(min_valid_fraction)
            accepted &= weight_sum > 0.0
            accepted &= torch.isfinite(means).all(dim=1)
            accepted &= torch.isfinite(covariances).all(dim=(1, 2))
            if color_values is not None:
                accepted &= torch.isfinite(mean_colors).all(dim=1)

            output_means[indices_np] = means.detach().cpu().numpy()
            output_covariances[indices_np] = covariances.detach().cpu().numpy()
            output_colors[indices_np] = mean_colors.detach().cpu().numpy()
            output_valid_counts[indices_np] = valid_counts.detach().cpu().numpy()
            output_valid[indices_np] = accepted.detach().cpu().numpy()

    output_means[~output_valid] = np.nan
    output_covariances[~output_valid] = np.nan
    output_colors[~output_valid] = np.nan
    return RegionFitResults(
        means=output_means,
        covariances=output_covariances,
        mean_colors=output_colors,
        valid_counts=output_valid_counts,
        support_counts=output_support_counts,
        valid=output_valid,
        continuity_reference_scale=continuity_reference_scale,
    )


def region_image_slices(
    region: PixelRegion,
    image_shape: tuple[int, int],
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    """Return aligned image and local-region slices, clipped to the image."""
    height, width = image_shape
    image_x0 = max(0, region.x0)
    image_y0 = max(0, region.y0)
    image_x1 = min(width, region.x0 + region.width)
    image_y1 = min(height, region.y0 + region.height)
    if image_x0 >= image_x1 or image_y0 >= image_y1:
        return None
    local_x0 = image_x0 - region.x0
    local_y0 = image_y0 - region.y0
    local_x1 = local_x0 + image_x1 - image_x0
    local_y1 = local_y0 + image_y1 - image_y0
    return (
        (slice(image_y0, image_y1), slice(image_x0, image_x1)),
        (slice(local_y0, local_y1), slice(local_x0, local_x1)),
    )


def _largest_connected_component(
    points,
    candidate,
    *,
    continuity_neighbors: int,
    continuity_ratio_max: float,
    continuity_reference_scale: float,
):
    """Return the largest fixed-step 3D component in each batched 2D mask."""
    import torch

    batch, height, width = candidate.shape
    pixel_count = height * width
    offsets = neighbor_offsets(continuity_neighbors)
    reference = float(continuity_reference_scale)
    tolerance = (
        float(continuity_ratio_max) * max(reference, 1.0e-12)
        if np.isfinite(reference)
        else 0.0
    )
    edge_masks = []
    for delta_y, delta_x in offsets:
        target_y, source_y = paired_slices(height, delta_y)
        target_x, source_x = paired_slices(width, delta_x)
        target_valid = candidate[:, target_y, target_x]
        source_valid = candidate[:, source_y, source_x]
        delta = points[:, source_y, source_x] - points[:, target_y, target_x]
        step = torch.linalg.norm(delta, dim=-1) / float(np.hypot(delta_x, delta_y))
        edge = target_valid & source_valid & torch.isfinite(step) & (step <= tolerance)
        edge_masks.append((target_y, target_x, source_y, source_x, edge))

    base = torch.arange(batch, device=points.device, dtype=torch.int64)[:, None] * pixel_count
    labels = base + torch.arange(pixel_count, device=points.device, dtype=torch.int64)[None]
    labels = labels.reshape(batch, height, width)
    invalid_label = batch * pixel_count
    labels = torch.where(candidate, labels, torch.full_like(labels, invalid_label))
    for _ in range(pixel_count):
        updated = labels.clone()
        for target_y, target_x, source_y, source_x, edge in edge_masks:
            current = updated[:, target_y, target_x]
            neighbor = labels[:, source_y, source_x]
            updated[:, target_y, target_x] = torch.where(
                edge, torch.minimum(current, neighbor), current
            )
        updated = torch.where(candidate, updated, torch.full_like(updated, invalid_label))
        if torch.equal(updated, labels):
            break
        labels = updated

    selected_labels = labels[candidate]
    counts = torch.bincount(selected_labels, minlength=batch * pixel_count + 1)
    counts = counts[: batch * pixel_count].reshape(batch, pixel_count)
    largest_local_label = torch.argmax(counts, dim=1)
    largest_label = base[:, 0] + largest_local_label
    return candidate & (labels == largest_label[:, None, None])


def _resolve_device(device: str):
    import torch

    normalized = str(device).lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("covariance.device must be 'auto', 'cpu', or 'cuda'")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA covariance computation was requested but CUDA is unavailable")
    return torch.device(normalized)


def _validate_fit_inputs(
    world_points: np.ndarray,
    regions: Sequence[PixelRegion],
    *,
    colors: np.ndarray | None,
    image_valid_mask: np.ndarray,
    min_valid_points: int,
    min_valid_fraction: float,
    continuity_neighbors: int,
    continuity_ratio_max: float,
    pixel_budget: int,
) -> None:
    if world_points.ndim != 3 or world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape [H, W, 3]")
    if colors is not None and colors.shape != world_points.shape:
        raise ValueError("colors must have shape [H, W, 3]")
    if image_valid_mask.shape != world_points.shape[:2]:
        raise ValueError("image_valid_mask must have shape [H, W]")
    if any(not isinstance(region, PixelRegion) for region in regions):
        raise TypeError("regions must contain PixelRegion values")
    if min_valid_points < 2:
        raise ValueError("min_valid_points must be at least two")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must lie in [0, 1]")
    if continuity_neighbors not in {4, 8}:
        raise ValueError("continuity_neighbors must be 4 or 8")
    if not np.isfinite(continuity_ratio_max) or continuity_ratio_max < 1.0:
        raise ValueError("continuity_ratio_max must be finite and at least one")
    if pixel_budget <= 0:
        raise ValueError("pixel_budget must be positive")
