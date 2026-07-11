from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .types import EllipseKeypoints


def detect_multiscale_keypoints(
    *,
    view_id: int,
    image: np.ndarray,
    confidence: np.ndarray,
    world_points: np.ndarray,
    sigmas: Sequence[float],
    response_threshold: float,
    max_keypoints: int,
    min_distance: int,
    nms_radius_factor: float,
    structure_sigma_factor: float,
    ellipse_radius_factor: float,
    min_ellipse_area: float,
    max_ellipse_area: float,
    max_axis_ratio: float,
    confidence_threshold: float,
    image_valid_mask: np.ndarray | None = None,
) -> EllipseKeypoints:
    """Detect scale-space LoG extrema and attach a structure-tensor ellipse.

    Coordinates use ``(x, y) == (u, v)`` throughout.  Each ellipse is represented
    by a positive-definite matrix E and contains offsets d satisfying
    ``d.T @ inv(E) @ d <= 1``.
    """
    sigma_values = _validate_detector_parameters(
        sigmas=sigmas,
        response_threshold=response_threshold,
        max_keypoints=max_keypoints,
        min_distance=min_distance,
        nms_radius_factor=nms_radius_factor,
        structure_sigma_factor=structure_sigma_factor,
        ellipse_radius_factor=ellipse_radius_factor,
        min_ellipse_area=min_ellipse_area,
        max_ellipse_area=max_ellipse_area,
        max_axis_ratio=max_axis_ratio,
    )
    if max_keypoints == 0:
        return _empty_keypoints()

    gray = to_gray(image)
    if confidence.shape != gray.shape or world_points.shape != (*gray.shape, 3):
        raise ValueError("image, confidence, and world_points must share H/W dimensions")
    if not np.isfinite(confidence_threshold):
        raise ValueError("confidence_threshold must be finite")
    valid = valid_pixel_mask(confidence, world_points, confidence_threshold)
    if image_valid_mask is not None:
        content_valid = np.asarray(image_valid_mask, dtype=bool)
        if content_valid.shape != gray.shape:
            raise ValueError("image_valid_mask must have shape [H, W]")
        valid &= content_valid
    scale_sigmas = add_guard_scales(sigma_values)
    all_blurred_levels = [masked_gaussian_blur(gray, sigma, valid) for sigma in scale_sigmas]
    responses = np.stack(
        [
            scale_normalized_laplacian(level, sigma)
            for level, sigma in zip(all_blurred_levels, scale_sigmas, strict=True)
        ],
        axis=0,
    )
    extrema = scale_space_extrema(responses)
    extrema &= valid[None, :, :]
    extrema &= np.abs(responses) >= float(response_threshold)

    level_ids, vs, us = np.nonzero(extrema)
    if len(us) == 0:
        return _empty_keypoints()

    scores = np.abs(responses[level_ids, vs, us]).astype(np.float32)
    selected = suppress_keypoints(
        us=us,
        vs=vs,
        levels=level_ids,
        scores=scores,
        sigmas=scale_sigmas,
        image_shape=gray.shape,
        max_keypoints=max_keypoints,
        min_distance=min_distance,
        radius_factor=nms_radius_factor,
    )
    us = us[selected].astype(np.int64)
    vs = vs[selected].astype(np.int64)
    level_ids = level_ids[selected].astype(np.int64)
    scores = scores[selected]

    configured_level_ids = level_ids - 1
    required_levels = np.unique(configured_level_ids)
    tensor_levels = {
        int(level): structure_tensor(
            all_blurred_levels[int(level) + 1],
            integration_sigma=structure_sigma_factor * sigma_values[int(level)],
            valid_mask=valid,
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

    return EllipseKeypoints(
        view_ids=np.full((len(us),), view_id, dtype=np.int64),
        us=us,
        vs=vs,
        scores=scores,
        sigmas=np.asarray(
            [sigma_values[level] for level in configured_level_ids], dtype=np.float32
        ),
        levels=configured_level_ids,
        ellipse_matrices=ellipse_matrices,
        ellipse_areas=ellipse_areas,
    )


def valid_pixel_mask(
    confidence: np.ndarray,
    world_points: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    finite = np.isfinite(world_points).all(axis=-1)
    return finite & np.isfinite(confidence) & (confidence >= confidence_threshold)


def add_guard_scales(sigmas: Sequence[float]) -> tuple[float, ...]:
    """Add geometric guard levels around the configured usable LoD range."""
    values = tuple(float(value) for value in sigmas)
    lower_ratio = values[1] / values[0]
    upper_ratio = values[-1] / values[-2]
    return (values[0] / lower_ratio, *values, values[-1] * upper_ratio)


def scale_space_extrema(responses: np.ndarray) -> np.ndarray:
    """Return strict signed extrema in a 3x3x3 scale-space neighborhood."""
    values = np.asarray(responses, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("responses must have shape [S, H, W]")
    scales, height, width = values.shape
    if scales < 3:
        raise ValueError("At least three LoG scales are required")

    neighbor_max = np.full_like(values, -np.inf)
    neighbor_min = np.full_like(values, np.inf)
    padded = np.pad(values, ((1, 1), (1, 1), (1, 1)), mode="edge")
    for ds in range(3):
        for dy in range(3):
            for dx in range(3):
                if ds == 1 and dy == 1 and dx == 1:
                    continue
                neighbor = padded[ds : ds + scales, dy : dy + height, dx : dx + width]
                neighbor_max = np.maximum(neighbor_max, neighbor)
                neighbor_min = np.minimum(neighbor_min, neighbor)

    extrema = (values > neighbor_max) | (values < neighbor_min)
    extrema[0] = False
    extrema[-1] = False
    if height:
        extrema[:, 0] = False
        extrema[:, -1] = False
    if width:
        extrema[:, :, 0] = False
        extrema[:, :, -1] = False
    return extrema


def suppress_keypoints(
    *,
    us: np.ndarray,
    vs: np.ndarray,
    levels: np.ndarray,
    scores: np.ndarray,
    sigmas: Sequence[float],
    image_shape: tuple[int, int],
    max_keypoints: int,
    min_distance: int,
    radius_factor: float,
) -> np.ndarray:
    """Greedily suppress weaker extrema across all LoG levels."""
    order = np.argsort(-scores, kind="stable")
    blocked = np.zeros(image_shape, dtype=bool)
    height, width = image_shape
    selected: list[int] = []
    for index in order:
        u = int(us[index])
        v = int(vs[index])
        if blocked[v, u]:
            continue
        selected.append(int(index))
        if len(selected) >= max_keypoints:
            break
        radius = max(int(min_distance), int(math.ceil(radius_factor * sigmas[int(levels[index])])))
        y0 = max(0, v - radius)
        y1 = min(height, v + radius + 1)
        x0 = max(0, u - radius)
        x1 = min(width, u + radius + 1)
        blocked[y0:y1, x0:x1] = True
    return np.asarray(selected, dtype=np.int64)


def structure_tensor(
    image: np.ndarray,
    *,
    integration_sigma: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a Gaussian-integrated 2D structure tensor in x/y coordinates."""
    dx, dy = image_gradients(image)
    j_xx = masked_gaussian_blur(dx * dx, integration_sigma, valid_mask)
    j_xy = masked_gaussian_blur(dx * dy, integration_sigma, valid_mask)
    j_yy = masked_gaussian_blur(dy * dy, integration_sigma, valid_mask)
    output = np.empty((*image.shape, 2, 2), dtype=np.float32)
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


def to_gray(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 2:
        return values.astype(np.float32)
    if values.ndim != 3:
        raise ValueError("image must have shape [H, W] or [H, W, C]")
    if values.shape[-1] == 1:
        return values[..., 0].astype(np.float32)
    if values.shape[-1] < 3:
        raise ValueError("image must have at least three channels")
    return (
        0.299 * values[..., 0].astype(np.float32)
        + 0.587 * values[..., 1].astype(np.float32)
        + 0.114 * values[..., 2].astype(np.float32)
    )


def _validate_detector_parameters(
    *,
    sigmas: Sequence[float],
    response_threshold: float,
    max_keypoints: int,
    min_distance: int,
    nms_radius_factor: float,
    structure_sigma_factor: float,
    ellipse_radius_factor: float,
    min_ellipse_area: float,
    max_ellipse_area: float,
    max_axis_ratio: float,
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
        nms_radius_factor,
        structure_sigma_factor,
        ellipse_radius_factor,
        min_ellipse_area,
        max_ellipse_area,
        max_axis_ratio,
    )
    if not np.isfinite(floating_parameters).all():
        raise ValueError("sampling parameters must be finite")
    if response_threshold < 0.0:
        raise ValueError("response_threshold must be non-negative")
    if max_keypoints < 0 or min_distance < 0:
        raise ValueError("keypoint limits must be non-negative")
    if nms_radius_factor < 0.0 or structure_sigma_factor <= 0.0 or ellipse_radius_factor <= 0.0:
        raise ValueError("scale factors must be positive (NMS may be zero)")
    if min_ellipse_area <= 0.0 or max_ellipse_area < min_ellipse_area:
        raise ValueError("ellipse area bounds are invalid")
    if max_axis_ratio < 1.0:
        raise ValueError("max_axis_ratio must be at least one")
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
