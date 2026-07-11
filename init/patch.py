from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchPoints:
    center: np.ndarray
    points: np.ndarray
    center_confidence: float
    mean_confidence: float


def extract_local_patch(
    world_points: np.ndarray,
    confidence: np.ndarray,
    *,
    u: int,
    v: int,
    radius: int,
    min_valid_points: int,
    confidence_threshold: float,
    max_center_distance: float | None,
) -> PatchPoints | None:
    height, width, _ = world_points.shape
    if not (0 <= u < width and 0 <= v < height):
        return None

    center = world_points[v, u].astype(np.float32)
    center_conf = float(confidence[v, u])
    if not np.isfinite(center).all() or not np.isfinite(center_conf):
        return None
    if center_conf < confidence_threshold:
        return None

    window_radius = max(int(radius), 0)
    y0 = max(0, v - window_radius)
    y1 = min(height, v + window_radius + 1)
    x0 = max(0, u - window_radius)
    x1 = min(width, u + window_radius + 1)

    patch_points = world_points[y0:y1, x0:x1].reshape(-1, 3)
    patch_conf = confidence[y0:y1, x0:x1].reshape(-1)
    valid = np.isfinite(patch_points).all(axis=1)
    valid &= np.isfinite(patch_conf)
    valid &= patch_conf >= confidence_threshold

    points = patch_points[valid].astype(np.float32)
    conf = patch_conf[valid].astype(np.float32)
    if max_center_distance is not None and points.size:
        distances = np.linalg.norm(points - center[None, :], axis=1)
        close = distances <= float(max_center_distance)
        points = points[close]
        conf = conf[close]

    if points.shape[0] < min_valid_points:
        return None

    return PatchPoints(
        center=center,
        points=points,
        center_confidence=center_conf,
        mean_confidence=float(np.mean(conf)),
    )
