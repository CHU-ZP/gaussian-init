from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .continuity import global_scene_step_scale
from .regions import PixelRegion, region_image_slices
from .sampling import rgb_to_normalized_lab


@dataclass(frozen=True)
class GridSupplementConfig:
    enabled: bool = True
    cell_size: int = 12
    min_valid_fraction: float = 0.6
    color_delta_e_max: float = 25.0
    boundary_continuity_fraction_min: float = 0.6
    continuity_ratio_max: float = 3.0
    max_component_pixels: int = 1500
    convex_fill_ratio_min: float = 0.9
    require_hole_free: bool = True

    def __post_init__(self) -> None:
        finite_values = (
            self.min_valid_fraction,
            self.color_delta_e_max,
            self.boundary_continuity_fraction_min,
            self.continuity_ratio_max,
            self.convex_fill_ratio_min,
        )
        if not np.isfinite(finite_values).all():
            raise ValueError("grid supplement parameters must be finite")
        if self.cell_size <= 0:
            raise ValueError("sampling.grid_supplement.cell_size must be positive")
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError(
                "sampling.grid_supplement.min_valid_fraction must lie in [0, 1]"
            )
        if self.color_delta_e_max < 0.0:
            raise ValueError("sampling.grid_supplement.color_delta_e_max must be non-negative")
        if not 0.0 <= self.boundary_continuity_fraction_min <= 1.0:
            raise ValueError(
                "sampling.grid_supplement.boundary_continuity_fraction_min must lie in [0, 1]"
            )
        if self.continuity_ratio_max <= 0.0:
            raise ValueError(
                "sampling.grid_supplement.continuity_ratio_max must be positive"
            )
        if self.max_component_pixels <= 0:
            raise ValueError(
                "sampling.grid_supplement.max_component_pixels must be positive"
            )
        if not 0.0 <= self.convex_fill_ratio_min <= 1.0:
            raise ValueError(
                "sampling.grid_supplement.convex_fill_ratio_min must lie in [0, 1]"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "GridSupplementConfig":
        config = values or {}
        return cls(
            enabled=bool(config.get("enabled", cls.enabled)),
            cell_size=int(config.get("cell_size", cls.cell_size)),
            min_valid_fraction=float(
                config.get("min_valid_fraction", cls.min_valid_fraction)
            ),
            color_delta_e_max=float(config.get("color_delta_e_max", cls.color_delta_e_max)),
            boundary_continuity_fraction_min=float(
                config.get(
                    "boundary_continuity_fraction_min",
                    cls.boundary_continuity_fraction_min,
                )
            ),
            continuity_ratio_max=float(
                config.get("continuity_ratio_max", cls.continuity_ratio_max)
            ),
            max_component_pixels=int(
                config.get("max_component_pixels", cls.max_component_pixels)
            ),
            convex_fill_ratio_min=float(
                config.get("convex_fill_ratio_min", cls.convex_fill_ratio_min)
            ),
            require_hole_free=bool(
                config.get("require_hole_free", cls.require_hole_free)
            ),
        )


@dataclass(frozen=True)
class GridSupplementResult:
    regions: list[PixelRegion]
    coverage_mask: np.ndarray
    valid_pixels: int
    covered_pixels: int
    candidate_cells: int
    merged_cells: int
    shape_rejected_merges: int
    continuity_reference_scale: float


@dataclass(frozen=True)
class _GridCell:
    row: int
    column: int
    x0: int
    y0: int
    mask: np.ndarray
    lab: np.ndarray
    score: float

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    @property
    def support_count(self) -> int:
        return int(np.count_nonzero(self.mask))


def build_grid_supplement(
    image: np.ndarray,
    world_points: np.ndarray,
    ellipse_supports: Sequence[PixelRegion],
    *,
    image_valid_mask: np.ndarray | None,
    ellipse_min_valid_points: int,
    ellipse_min_valid_fraction: float,
    continuity_neighbors: int,
    config: GridSupplementConfig,
) -> GridSupplementResult:
    """Use ellipse coverage only to exclude fully covered image cells."""
    image_values = np.asarray(image, dtype=np.float32)
    points = np.asarray(world_points, dtype=np.float32)
    if image_values.ndim != 3 or image_values.shape[-1] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if points.shape != image_values.shape:
        raise ValueError("image and world_points must share H/W dimensions")
    content_valid = (
        np.ones(points.shape[:2], dtype=bool)
        if image_valid_mask is None
        else np.asarray(image_valid_mask, dtype=bool)
    )
    if content_valid.shape != points.shape[:2]:
        raise ValueError("image_valid_mask must have shape [H, W]")
    valid = content_valid.copy()
    valid &= np.isfinite(points).all(axis=-1)
    continuity_reference_scale = global_scene_step_scale(
        points, valid, neighbors=continuity_neighbors
    )

    coverage = _ellipse_coverage(
        ellipse_supports,
        valid,
        min_valid_points=ellipse_min_valid_points,
        min_valid_fraction=ellipse_min_valid_fraction,
    )
    valid_pixels = int(np.count_nonzero(valid))
    covered_pixels = int(np.count_nonzero(coverage))
    if not config.enabled:
        return GridSupplementResult(
            regions=[],
            coverage_mask=coverage,
            valid_pixels=valid_pixels,
            covered_pixels=covered_pixels,
            candidate_cells=0,
            merged_cells=0,
            shape_rejected_merges=0,
            continuity_reference_scale=continuity_reference_scale,
        )

    lab = rgb_to_normalized_lab(image_values).astype(np.float64)
    lab *= np.asarray([100.0, 128.0, 128.0], dtype=np.float64)
    cells = _candidate_cells(
        lab,
        valid,
        coverage,
        min_valid_fraction=config.min_valid_fraction,
        cell_size=config.cell_size,
        max_component_pixels=config.max_component_pixels,
    )
    regions, shape_rejected_merges = _merge_cells(
        cells,
        valid=valid,
        world_points=points,
        reference_scale=continuity_reference_scale,
        config=config,
    )
    return GridSupplementResult(
        regions=regions,
        coverage_mask=coverage,
        valid_pixels=valid_pixels,
        covered_pixels=covered_pixels,
        candidate_cells=len(cells),
        merged_cells=len(cells) - len(regions),
        shape_rejected_merges=shape_rejected_merges,
        continuity_reference_scale=continuity_reference_scale,
    )


def _ellipse_coverage(
    regions: Sequence[PixelRegion],
    valid: np.ndarray,
    *,
    min_valid_points: int,
    min_valid_fraction: float,
) -> np.ndarray:
    coverage = np.zeros(valid.shape, dtype=bool)
    for region in regions:
        aligned = region_image_slices(region, valid.shape)
        if aligned is None:
            continue
        image_slices, local_slices = aligned
        local_support = region.mask[local_slices]
        eligible = local_support & valid[image_slices]
        valid_count = int(np.count_nonzero(eligible))
        valid_fraction = valid_count / max(region.support_count, 1)
        if valid_count < min_valid_points or valid_fraction < min_valid_fraction:
            continue
        coverage_view = coverage[image_slices]
        coverage_view |= eligible
    return coverage


def _candidate_cells(
    lab: np.ndarray,
    valid: np.ndarray,
    coverage: np.ndarray,
    *,
    min_valid_fraction: float,
    cell_size: int,
    max_component_pixels: int,
) -> list[_GridCell]:
    height, width = valid.shape
    cells: list[_GridCell] = []
    for y0 in range(0, height, cell_size):
        y1 = min(height, y0 + cell_size)
        for x0 in range(0, width, cell_size):
            x1 = min(width, x0 + cell_size)
            cell_valid = valid[y0:y1, x0:x1]
            valid_count = int(np.count_nonzero(cell_valid))
            if valid_count / cell_valid.size < min_valid_fraction:
                continue
            covered_count = int(np.count_nonzero(coverage[y0:y1, x0:x1] & cell_valid))
            covered_fraction = covered_count / max(valid_count, 1)
            if covered_count == valid_count:
                continue
            # Ellipse coverage is only a cell-level gate. Once a cell is a
            # candidate, retain its complete valid support so color, boundary
            # continuity and 3D fitting operate on one regular grid region.
            support = cell_valid
            support_count = int(np.count_nonzero(support))
            if support_count > max_component_pixels:
                continue
            cells.append(
                _GridCell(
                    row=y0 // cell_size,
                    column=x0 // cell_size,
                    x0=x0,
                    y0=y0,
                    mask=support,
                    lab=np.median(lab[y0:y1, x0:x1][support], axis=0),
                    score=1.0 - covered_fraction,
                )
            )
    return cells


def _merge_cells(
    cells: Sequence[_GridCell],
    *,
    valid: np.ndarray,
    world_points: np.ndarray,
    reference_scale: float,
    config: GridSupplementConfig,
) -> tuple[list[PixelRegion], int]:
    if not cells:
        return [], 0
    by_position = {(cell.row, cell.column): index for index, cell in enumerate(cells)}
    edges: list[tuple[float, float, int, int]] = []
    for left, cell in enumerate(cells):
        for position in ((cell.row, cell.column + 1), (cell.row + 1, cell.column)):
            right = by_position.get(position)
            if right is None:
                continue
            delta_e = float(np.linalg.norm(cell.lab - cells[right].lab))
            if delta_e > config.color_delta_e_max:
                continue
            continuity = _boundary_continuity_fraction(
                cell,
                cells[right],
                valid=valid,
                world_points=world_points,
                reference_scale=reference_scale,
                ratio_max=config.continuity_ratio_max,
            )
            if continuity < config.boundary_continuity_fraction_min:
                continue
            edges.append((-continuity, delta_e, left, right))

    union_find = _UnionFind(len(cells))
    members = {index: [index] for index in range(len(cells))}
    areas = {index: cells[index].support_count for index in range(len(cells))}
    shape_rejected_merges = 0
    for _negative_continuity, _delta_e, left, right in sorted(edges):
        left_root = union_find.find(left)
        right_root = union_find.find(right)
        if left_root == right_root:
            continue
        combined_area = areas[left_root] + areas[right_root]
        if combined_area > config.max_component_pixels:
            continue
        combined_members = members[left_root] + members[right_root]
        if not _component_shape_is_valid(
            cells,
            combined_members,
            convex_fill_ratio_min=config.convex_fill_ratio_min,
            require_hole_free=config.require_hole_free,
        ):
            shape_rejected_merges += 1
            continue
        new_root = union_find.union(left_root, right_root)
        removed_root = right_root if new_root == left_root else left_root
        members[new_root] = combined_members
        areas[new_root] = combined_area
        del members[removed_root]
        del areas[removed_root]

    regions = [
        _component_region(cells, component)
        for component in sorted(members.values(), key=min)
    ]
    return regions, shape_rejected_merges


def _component_shape_is_valid(
    cells: Sequence[_GridCell],
    component: Sequence[int],
    *,
    convex_fill_ratio_min: float,
    require_hole_free: bool,
) -> bool:
    selected = [cells[index] for index in component]
    if require_hole_free and _component_has_grid_hole(selected):
        return False
    return _component_convex_fill_ratio(selected) + 1.0e-12 >= convex_fill_ratio_min


def _component_convex_fill_ratio(cells: Sequence[_GridCell]) -> float:
    if not cells:
        return 0.0
    points: list[tuple[int, int]] = []
    component_area = 0.0
    for cell in cells:
        x1 = cell.x0 + cell.width
        y1 = cell.y0 + cell.height
        points.extend(
            (
                (cell.x0, cell.y0),
                (x1, cell.y0),
                (x1, y1),
                (cell.x0, y1),
            )
        )
        component_area += float(cell.width * cell.height)
    hull = _convex_hull(points)
    hull_area = _polygon_area(hull)
    if hull_area <= 0.0:
        return 0.0
    return min(component_area / hull_area, 1.0)


def _convex_hull(points: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(
        origin: tuple[int, int],
        left: tuple[int, int],
        right: tuple[int, int],
    ) -> int:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _polygon_area(points: Sequence[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    area_twice = 0
    for left, right in zip(points, (*points[1:], points[0]), strict=True):
        area_twice += left[0] * right[1] - left[1] * right[0]
    return 0.5 * abs(float(area_twice))


def _component_has_grid_hole(cells: Sequence[_GridCell]) -> bool:
    if len(cells) < 8:
        return False
    min_row = min(cell.row for cell in cells)
    max_row = max(cell.row for cell in cells)
    min_column = min(cell.column for cell in cells)
    max_column = max(cell.column for cell in cells)
    occupied = np.zeros(
        (max_row - min_row + 1, max_column - min_column + 1),
        dtype=bool,
    )
    for cell in cells:
        occupied[cell.row - min_row, cell.column - min_column] = True

    padded = np.pad(occupied, 1, constant_values=False)
    outside = np.zeros_like(padded)
    outside[0, 0] = True
    stack = [(0, 0)]
    while stack:
        row, column = stack.pop()
        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row = row + delta_row
            next_column = column + delta_column
            if not 0 <= next_row < padded.shape[0] or not 0 <= next_column < padded.shape[1]:
                continue
            if padded[next_row, next_column] or outside[next_row, next_column]:
                continue
            outside[next_row, next_column] = True
            stack.append((next_row, next_column))
    return bool(np.any(~padded[1:-1, 1:-1] & ~outside[1:-1, 1:-1]))


def _boundary_continuity_fraction(
    left: _GridCell,
    right: _GridCell,
    *,
    valid: np.ndarray,
    world_points: np.ndarray,
    reference_scale: float,
    ratio_max: float,
) -> float:
    if not np.isfinite(reference_scale):
        return 0.0
    if left.column + 1 == right.column and left.row == right.row:
        y0 = max(left.y0, right.y0)
        y1 = min(left.y0 + left.height, right.y0 + right.height)
        left_y = slice(y0, y1)
        left_x = left.x0 + left.width - 1
        right_x = right.x0
        eligible = valid[left_y, left_x] & valid[left_y, right_x]
        deltas = world_points[left_y, right_x] - world_points[left_y, left_x]
    elif left.row + 1 == right.row and left.column == right.column:
        x0 = max(left.x0, right.x0)
        x1 = min(left.x0 + left.width, right.x0 + right.width)
        left_x = slice(x0, x1)
        left_y = left.y0 + left.height - 1
        right_y = right.y0
        eligible = valid[left_y, left_x] & valid[right_y, left_x]
        deltas = world_points[right_y, left_x] - world_points[left_y, left_x]
    else:
        return 0.0
    eligible_count = int(np.count_nonzero(eligible))
    if eligible_count == 0:
        return 0.0
    steps = np.linalg.norm(deltas, axis=-1)
    tolerance = float(ratio_max) * max(float(reference_scale), 1.0e-12)
    continuous = eligible & np.isfinite(steps) & (steps <= tolerance)
    return float(np.count_nonzero(continuous)) / eligible_count


def _component_region(cells: Sequence[_GridCell], component: Sequence[int]) -> PixelRegion:
    selected = [cells[index] for index in component]
    x0 = min(cell.x0 for cell in selected)
    y0 = min(cell.y0 for cell in selected)
    x1 = max(cell.x0 + cell.width for cell in selected)
    y1 = max(cell.y0 + cell.height for cell in selected)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    support_counts = np.asarray([cell.support_count for cell in selected], dtype=np.float64)
    scores = np.asarray([cell.score for cell in selected], dtype=np.float64)
    for cell in selected:
        local_y = slice(cell.y0 - y0, cell.y0 - y0 + cell.height)
        local_x = slice(cell.x0 - x0, cell.x0 - x0 + cell.width)
        mask[local_y, local_x] |= cell.mask
    return PixelRegion(
        x0=x0,
        y0=y0,
        mask=mask,
        source="grid",
        score=float(np.sum(scores * support_counts) / np.sum(support_counts)),
    )


class _UnionFind:
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
