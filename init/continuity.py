from __future__ import annotations

import numpy as np


_NEIGHBOR_OFFSETS = {
    4: ((-1, 0), (0, -1), (0, 1), (1, 0)),
    8: (
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, 0),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ),
}

_UNIQUE_NEIGHBOR_OFFSETS = {
    4: ((0, 1), (1, 0)),
    8: ((0, 1), (1, 0), (1, 1), (1, -1)),
}


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
    offsets = neighbor_offsets(neighbors, unique=True)
    height, width = valid.shape
    steps: list[np.ndarray] = []
    for delta_y, delta_x in offsets:
        target_y, neighbor_y = paired_slices(height, delta_y)
        target_x, neighbor_x = paired_slices(width, delta_x)
        eligible = valid[target_y, target_x] & valid[neighbor_y, neighbor_x]
        delta = points[neighbor_y, neighbor_x] - points[target_y, target_x]
        values = np.linalg.norm(delta, axis=-1) / float(np.hypot(delta_x, delta_y))
        selected = values[eligible & np.isfinite(values)]
        if len(selected):
            steps.append(selected)
    if not steps:
        return float("nan")
    return float(np.median(np.concatenate(steps)))


def neighbor_offsets(
    neighbors: int,
    *,
    unique: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Return directed or one-per-pair offsets for a 4/8-neighbor grid."""
    offsets = _UNIQUE_NEIGHBOR_OFFSETS if unique else _NEIGHBOR_OFFSETS
    try:
        return offsets[neighbors]
    except KeyError as error:
        raise ValueError("neighbors must be 4 or 8") from error


def paired_slices(length: int, delta: int) -> tuple[slice, slice]:
    """Return aligned target/source slices with source = target + delta."""
    if delta < 0:
        return slice(-delta, length), slice(0, length + delta)
    if delta > 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(0, length), slice(0, length)
