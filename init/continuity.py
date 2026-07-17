from __future__ import annotations

import numpy as np


def global_scene_step_scale(
    world_points: np.ndarray,
    valid_mask: np.ndarray,
    *,
    neighbors: int,
) -> float:
    """Return the median valid 3D step per image pixel for one whole view."""
    points = np.asarray(world_points, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if points.shape != (*valid.shape, 3):
        raise ValueError("world_points and valid_mask shapes are inconsistent")
    offsets = _unique_neighbor_offsets(neighbors)
    height, width = valid.shape
    steps: list[np.ndarray] = []
    for delta_y, delta_x in offsets:
        target_y, neighbor_y = _paired_slices(height, delta_y)
        target_x, neighbor_x = _paired_slices(width, delta_x)
        eligible = valid[target_y, target_x] & valid[neighbor_y, neighbor_x]
        delta = points[neighbor_y, neighbor_x] - points[target_y, target_x]
        values = np.linalg.norm(delta, axis=-1) / float(np.hypot(delta_x, delta_y))
        selected = values[eligible & np.isfinite(values)]
        if len(selected):
            steps.append(selected)
    if not steps:
        return float("nan")
    return float(np.median(np.concatenate(steps)))


def _unique_neighbor_offsets(neighbors: int) -> tuple[tuple[int, int], ...]:
    if neighbors == 4:
        return ((0, 1), (1, 0))
    if neighbors == 8:
        return ((0, 1), (1, 0), (1, 1), (1, -1))
    raise ValueError("neighbors must be 4 or 8")


def _paired_slices(length: int, delta: int) -> tuple[slice, slice]:
    if delta < 0:
        return slice(-delta, length), slice(0, length + delta)
    if delta > 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(0, length), slice(0, length)
