from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .continuity import global_scene_step_scale


@dataclass(frozen=True)
class EllipseCovarianceResults:
    covariances: np.ndarray
    mean_confidences: np.ndarray
    valid_counts: np.ndarray
    support_counts: np.ndarray
    valid: np.ndarray
    continuity_reference_scale: float


def compute_covariances(
    world_points: np.ndarray,
    confidence: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    ellipse_matrices: np.ndarray,
    *,
    image_valid_mask: np.ndarray | None = None,
    confidence_threshold: float,
    min_valid_points: int,
    min_valid_fraction: float,
    continuity_neighbors: int,
    continuity_ratio_max: float,
    confidence_weighted: bool,
    device: str,
    pixel_budget: int,
) -> EllipseCovarianceResults:
    """Compute local 3D covariances inside image-space ellipses.

    The implementation is a chunked tensor kernel.  With ``device='cuda'`` (or
    ``'auto'`` on a CUDA host), all masks, gathers, and covariance reductions run
    on the GPU.  The same implementation is used as the CPU fallback.
    """
    import torch

    points_np = np.asarray(world_points, dtype=np.float32)
    confidence_np = np.asarray(confidence, dtype=np.float32)
    us_np = np.asarray(us, dtype=np.int64)
    vs_np = np.asarray(vs, dtype=np.int64)
    matrices_np = np.asarray(ellipse_matrices, dtype=np.float32)
    image_valid_np = (
        np.ones(confidence_np.shape, dtype=bool)
        if image_valid_mask is None
        else np.asarray(image_valid_mask, dtype=bool)
    )
    _validate_inputs(
        points_np,
        confidence_np,
        us_np,
        vs_np,
        matrices_np,
        image_valid_mask=image_valid_np,
        confidence_threshold=confidence_threshold,
        min_valid_points=min_valid_points,
        min_valid_fraction=min_valid_fraction,
        continuity_neighbors=continuity_neighbors,
        continuity_ratio_max=continuity_ratio_max,
        pixel_budget=pixel_budget,
    )
    continuity_valid_np = image_valid_np.copy()
    continuity_valid_np &= np.isfinite(points_np).all(axis=-1)
    continuity_valid_np &= np.isfinite(confidence_np)
    continuity_valid_np &= confidence_np >= float(confidence_threshold)
    continuity_reference_scale = global_scene_step_scale(
        points_np,
        continuity_valid_np,
        neighbors=continuity_neighbors,
    )

    count = len(us_np)
    if count == 0:
        return EllipseCovarianceResults(
            covariances=np.empty((0, 3, 3), dtype=np.float32),
            mean_confidences=np.empty((0,), dtype=np.float32),
            valid_counts=np.empty((0,), dtype=np.int64),
            support_counts=np.empty((0,), dtype=np.int64),
            valid=np.empty((0,), dtype=bool),
            continuity_reference_scale=continuity_reference_scale,
        )

    torch_device = resolve_device(device)
    height, width, _ = points_np.shape
    points = torch.as_tensor(points_np, device=torch_device)
    confidences = torch.as_tensor(confidence_np, device=torch_device)
    us_tensor = torch.as_tensor(us_np, device=torch_device)
    vs_tensor = torch.as_tensor(vs_np, device=torch_device)
    matrices = torch.as_tensor(matrices_np, device=torch_device)
    image_valid = torch.as_tensor(image_valid_np, device=torch_device)

    output_covariances = np.full((count, 3, 3), np.nan, dtype=np.float32)
    output_confidences = np.full((count,), np.nan, dtype=np.float32)
    output_valid_counts = np.zeros((count,), dtype=np.int64)
    output_support_counts = np.zeros((count,), dtype=np.int64)
    output_valid = np.zeros((count,), dtype=bool)

    extents = torch.ceil(
        torch.sqrt(torch.clamp(torch.diagonal(matrices, dim1=-2, dim2=-1), min=0.0))
    ).to(torch.int64)
    bounding_extents = extents.detach().cpu().numpy()

    for extent_x, extent_y in np.unique(bounding_extents, axis=0):
        bucket = np.nonzero(
            (bounding_extents[:, 0] == extent_x) & (bounding_extents[:, 1] == extent_y)
        )[0]
        pixels_per_ellipse = (2 * int(extent_x) + 1) * (2 * int(extent_y) + 1)
        if pixels_per_ellipse > pixel_budget:
            raise ValueError(
                "An ellipse bounding window exceeds covariance.pixel_budget; "
                "increase the budget or tighten ellipse bounds"
            )
        chunk_size = max(1, int(pixel_budget) // max(pixels_per_ellipse, 1))
        offsets = _rectangle_offsets(int(extent_x), int(extent_y), torch_device)
        for start in range(0, len(bucket), chunk_size):
            indices_np = bucket[start : start + chunk_size]
            indices = torch.as_tensor(indices_np, device=torch_device)
            result = _compute_chunk(
                points=points,
                confidences=confidences,
                us=us_tensor[indices],
                vs=vs_tensor[indices],
                matrices=matrices[indices],
                image_valid=image_valid,
                offsets=offsets,
                height=height,
                width=width,
                confidence_threshold=confidence_threshold,
                min_valid_points=min_valid_points,
                min_valid_fraction=min_valid_fraction,
                continuity_neighbors=continuity_neighbors,
                continuity_ratio_max=continuity_ratio_max,
                continuity_reference_scale=continuity_reference_scale,
                confidence_weighted=confidence_weighted,
                extent_x=int(extent_x),
                extent_y=int(extent_y),
            )
            output_covariances[indices_np] = result[0].detach().cpu().numpy()
            output_confidences[indices_np] = result[1].detach().cpu().numpy()
            output_valid_counts[indices_np] = result[2].detach().cpu().numpy()
            output_support_counts[indices_np] = result[3].detach().cpu().numpy()
            output_valid[indices_np] = result[4].detach().cpu().numpy()

    return EllipseCovarianceResults(
        covariances=output_covariances,
        mean_confidences=output_confidences,
        valid_counts=output_valid_counts,
        support_counts=output_support_counts,
        valid=output_valid,
        continuity_reference_scale=continuity_reference_scale,
    )


def ellipse_mask(
    image_shape: tuple[int, int],
    *,
    u: int,
    v: int,
    ellipse_matrix: np.ndarray,
) -> np.ndarray:
    """Rasterize one ellipse; primarily useful for visualization and tests."""
    height, width = image_shape
    matrix = np.asarray(ellipse_matrix, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    extent_x = int(np.ceil(np.sqrt(max(float(matrix[0, 0]), 0.0))))
    extent_y = int(np.ceil(np.sqrt(max(float(matrix[1, 1]), 0.0))))
    x0 = max(0, int(u) - extent_x)
    x1 = min(width, int(u) + extent_x + 1)
    y0 = max(0, int(v) - extent_y)
    y1 = min(height, int(v) + extent_y + 1)
    mask = np.zeros((height, width), dtype=bool)
    if x0 >= x1 or y0 >= y1:
        return mask
    ys, xs = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64) - float(v),
        np.arange(x0, x1, dtype=np.float64) - float(u),
        indexing="ij",
    )
    quadratic = inverse[0, 0] * xs * xs + 2.0 * inverse[0, 1] * xs * ys + inverse[1, 1] * ys * ys
    mask[y0:y1, x0:x1] = quadratic <= 1.0 + 1.0e-6
    return mask


def resolve_device(device: str):
    import torch

    normalized = str(device).lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("covariance.device must be 'auto', 'cpu', or 'cuda'")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA covariance computation was requested but CUDA is unavailable")
    return torch.device(normalized)


def _compute_chunk(
    *,
    points,
    confidences,
    us,
    vs,
    matrices,
    image_valid,
    offsets,
    height: int,
    width: int,
    confidence_threshold: float,
    min_valid_points: int,
    min_valid_fraction: float,
    continuity_neighbors: int,
    continuity_ratio_max: float,
    continuity_reference_scale: float,
    confidence_weighted: bool,
    extent_x: int,
    extent_y: int,
):
    import torch

    dx = offsets[:, 0]
    dy = offsets[:, 1]
    xs = us[:, None] + dx[None, :]
    ys = vs[:, None] + dy[None, :]
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    inverse = torch.linalg.inv(matrices)
    quadratic = (
        inverse[:, 0, 0, None] * dx[None, :] ** 2
        + 2.0 * inverse[:, 0, 1, None] * dx[None, :] * dy[None, :]
        + inverse[:, 1, 1, None] * dy[None, :] ** 2
    )
    full_support = quadratic <= 1.0 + 1.0e-6
    support = in_bounds & full_support

    flat_indices = torch.clamp(ys, 0, height - 1) * width + torch.clamp(xs, 0, width - 1)
    gathered_points = points.reshape(-1, 3)[flat_indices]
    gathered_confidence = confidences.reshape(-1)[flat_indices]
    gathered_image_valid = image_valid.reshape(-1)[flat_indices]
    candidate = support & torch.isfinite(gathered_points).all(dim=-1)
    candidate &= gathered_image_valid
    candidate &= torch.isfinite(gathered_confidence)
    candidate &= gathered_confidence >= float(confidence_threshold)
    center_points = points[vs, us]
    center_confidences = confidences[vs, us]
    center_image_valid = image_valid[vs, us]
    center_valid = torch.isfinite(center_points).all(dim=-1)
    center_valid &= torch.isfinite(center_confidences)
    center_valid &= center_confidences >= float(confidence_threshold)
    center_valid &= center_image_valid

    valid = _center_connected_component(
        gathered_points,
        candidate,
        extent_x=extent_x,
        extent_y=extent_y,
        continuity_neighbors=continuity_neighbors,
        continuity_ratio_max=continuity_ratio_max,
        continuity_reference_scale=continuity_reference_scale,
    )

    valid_counts = valid.sum(dim=1)
    support_counts = full_support.sum(dim=1)
    clean_points = torch.where(valid[..., None], gathered_points, torch.zeros_like(gathered_points))
    if confidence_weighted:
        raw_weights = torch.clamp(gathered_confidence, min=0.0)
    else:
        raw_weights = torch.ones_like(gathered_confidence)
    weights = torch.where(valid, raw_weights, torch.zeros_like(raw_weights))
    weight_sum = weights.sum(dim=1)
    # The Gaussian center is the keypoint scene coordinate, so estimate the
    # second moment around that same fixed center instead of a patch centroid.
    centered = torch.where(
        valid[..., None],
        clean_points - center_points[:, None, :],
        torch.zeros_like(clean_points),
    )
    scatter = torch.bmm((centered * weights[..., None]).transpose(1, 2), centered)
    denominator = weight_sum
    covariances = scatter / torch.clamp(denominator, min=1.0e-20)[:, None, None]
    mean_confidences = torch.where(
        valid_counts > 0,
        torch.where(valid, gathered_confidence, torch.zeros_like(gathered_confidence)).sum(dim=1)
        / torch.clamp(valid_counts, min=1).to(points.dtype),
        torch.full_like(weight_sum, float("nan")),
    )
    valid_fraction = valid_counts.to(points.dtype) / torch.clamp(support_counts, min=1).to(
        points.dtype
    )
    accepted = valid_counts >= int(min_valid_points)
    accepted &= valid_fraction >= float(min_valid_fraction)
    accepted &= denominator > 0.0
    accepted &= center_valid
    accepted &= torch.isfinite(covariances).all(dim=(1, 2))
    covariances = torch.where(
        accepted[:, None, None],
        covariances,
        torch.full_like(covariances, float("nan")),
    )
    return covariances, mean_confidences, valid_counts, support_counts, accepted


def _center_connected_component(
    points,
    candidate,
    *,
    extent_x: int,
    extent_y: int,
    continuity_neighbors: int,
    continuity_ratio_max: float,
    continuity_reference_scale: float,
):
    """Keep the globally step-bounded 3D component containing the keypoint.

    Candidate edges follow the image's 4- or 8-neighbor topology. An edge is
    traversable when its 3D step per image pixel is no larger than
    ``continuity_ratio_max * continuity_reference_scale``. The reference is one
    robust median for the whole view, so an anomalous endpoint cannot relax its
    own threshold.
    """
    import torch

    batch = points.shape[0]
    grid_height = 2 * extent_y + 1
    grid_width = 2 * extent_x + 1
    grid_points = points.reshape(batch, grid_height, grid_width, 3)
    grid_candidate = candidate.reshape(batch, grid_height, grid_width)
    offsets = _neighbor_offsets(continuity_neighbors)

    tolerance = float(continuity_ratio_max) * max(
        float(continuity_reference_scale),
        1.0e-12,
    )
    edge_masks = []
    for delta_y, delta_x in offsets:
        target_y, source_y = _paired_slices(grid_height, delta_y)
        target_x, source_x = _paired_slices(grid_width, delta_x)
        target_valid = grid_candidate[:, target_y, target_x]
        source_valid = grid_candidate[:, source_y, source_x]
        point_delta = grid_points[:, source_y, source_x] - grid_points[:, target_y, target_x]
        pixel_step = float(np.hypot(delta_x, delta_y))
        step = torch.linalg.norm(point_delta, dim=-1) / pixel_step
        edge = target_valid & source_valid & torch.isfinite(step)
        edge &= step <= tolerance
        edge_grid = torch.zeros_like(grid_candidate)
        edge_grid[:, target_y, target_x] = edge
        edge_masks.append(edge_grid)

    connected = torch.zeros_like(grid_candidate)
    connected[:, extent_y, extent_x] = grid_candidate[:, extent_y, extent_x]
    # The usual ellipse converges in roughly its radius.  The full grid-size
    # bound also handles winding valid regions exactly.
    for _ in range(grid_height * grid_width):
        expanded = connected.clone()
        for (delta_y, delta_x), edge_grid in zip(offsets, edge_masks, strict=True):
            target_y, source_y = _paired_slices(grid_height, delta_y)
            target_x, source_x = _paired_slices(grid_width, delta_x)
            expanded[:, target_y, target_x] |= (
                connected[:, source_y, source_x] & edge_grid[:, target_y, target_x]
            )
        updated = expanded & grid_candidate
        if torch.equal(updated, connected):
            break
        connected = updated
    return connected.reshape_as(candidate)


def _neighbor_offsets(neighbors: int) -> tuple[tuple[int, int], ...]:
    cardinal = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if neighbors == 4:
        return cardinal
    return cardinal + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _paired_slices(length: int, delta: int) -> tuple[slice, slice]:
    """Return aligned target/source slices with source = target + delta."""
    if delta < 0:
        return slice(-delta, length), slice(0, length + delta)
    if delta > 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(0, length), slice(0, length)


def _rectangle_offsets(extent_x: int, extent_y: int, device):
    import torch

    xs = torch.arange(-extent_x, extent_x + 1, dtype=torch.int64, device=device)
    ys = torch.arange(-extent_y, extent_y + 1, dtype=torch.int64, device=device)
    dy, dx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)


def _validate_inputs(
    world_points: np.ndarray,
    confidence: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    ellipse_matrices: np.ndarray,
    *,
    image_valid_mask: np.ndarray,
    confidence_threshold: float,
    min_valid_points: int,
    min_valid_fraction: float,
    continuity_neighbors: int,
    continuity_ratio_max: float,
    pixel_budget: int,
) -> None:
    if world_points.ndim != 3 or world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape [H, W, 3]")
    if confidence.shape != world_points.shape[:2]:
        raise ValueError("confidence must have shape [H, W]")
    if image_valid_mask.shape != confidence.shape:
        raise ValueError("image_valid_mask must have shape [H, W]")
    count = len(us)
    if us.shape != (count,) or vs.shape != (count,):
        raise ValueError("us and vs must be one-dimensional")
    if ellipse_matrices.shape != (count, 2, 2):
        raise ValueError("ellipse_matrices must have shape [N, 2, 2]")
    if count:
        height, width = confidence.shape
        if np.any(us < 0) or np.any(us >= width) or np.any(vs < 0) or np.any(vs >= height):
            raise ValueError("ellipse centers must lie inside the image")
        matrices64 = ellipse_matrices.astype(np.float64)
        if not np.isfinite(matrices64).all():
            raise ValueError("ellipse_matrices must be finite and positive definite")
        if not np.allclose(matrices64, np.swapaxes(matrices64, -1, -2), atol=1.0e-7):
            raise ValueError("ellipse_matrices must be symmetric")
        if np.any(np.linalg.eigvalsh(matrices64) <= 0.0):
            raise ValueError("ellipse_matrices must be positive definite")
    if min_valid_points < 2:
        raise ValueError("min_valid_points must be at least two")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must lie in [0, 1]")
    if not np.isfinite(confidence_threshold):
        raise ValueError("confidence_threshold must be finite")
    if continuity_neighbors not in {4, 8}:
        raise ValueError("continuity_neighbors must be 4 or 8")
    if not np.isfinite(continuity_ratio_max) or continuity_ratio_max < 1.0:
        raise ValueError("continuity_ratio_max must be finite and at least one")
    if pixel_budget <= 0:
        raise ValueError("pixel_budget must be positive")
