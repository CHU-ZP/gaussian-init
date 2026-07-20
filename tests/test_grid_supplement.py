from __future__ import annotations

import numpy as np

from init.grid_supplement import GridSupplementConfig, build_grid_supplement
from init.regions import PixelRegion, fit_regions


def test_grid_supplement_adds_only_uncovered_cell() -> None:
    image, points = _flat_view(height=16, width=32)
    ellipse = PixelRegion(
        x0=0,
        y0=0,
        mask=np.ones((16, 16), dtype=bool),
        source="ellipse",
        score=2.0,
    )
    result = build_grid_supplement(
        image,
        points,
        [ellipse],
        image_valid_mask=np.ones((16, 32), dtype=bool),
        ellipse_min_valid_points=16,
        ellipse_min_valid_fraction=0.6,
        continuity_neighbors=8,
        config=GridSupplementConfig(cell_size=16),
    )

    assert result.covered_pixels == 16 * 16
    assert result.candidate_cells == 1
    assert len(result.regions) == 1
    assert result.regions[0].x0 == 16
    assert result.regions[0].support_count == 16 * 16


def test_partially_covered_candidate_uses_complete_valid_cell() -> None:
    image, points = _flat_view(height=16, width=16)
    ellipse_mask = np.ones((16, 16), dtype=bool)
    ellipse_mask[-1, -1] = False
    ellipse = PixelRegion(
        x0=0,
        y0=0,
        mask=ellipse_mask,
        source="ellipse",
        score=2.0,
    )
    result = build_grid_supplement(
        image,
        points,
        [ellipse],
        image_valid_mask=np.ones((16, 16), dtype=bool),
        ellipse_min_valid_points=16,
        ellipse_min_valid_fraction=0.6,
        continuity_neighbors=8,
        config=GridSupplementConfig(cell_size=16),
    )

    assert result.covered_pixels == 16 * 16 - 1
    assert result.candidate_cells == 1
    assert len(result.regions) == 1
    assert result.regions[0].support_count == 16 * 16
    assert np.isclose(result.regions[0].score, 1.0 / (16 * 16))


def test_ellipse_overlap_does_not_mask_grid_merge_boundary() -> None:
    image, points = _flat_view(height=16, width=32)
    ellipse = PixelRegion(
        x0=15,
        y0=0,
        mask=np.ones((16, 2), dtype=bool),
        source="ellipse",
        score=2.0,
    )
    result = build_grid_supplement(
        image,
        points,
        [ellipse],
        image_valid_mask=np.ones((16, 32), dtype=bool),
        ellipse_min_valid_points=16,
        ellipse_min_valid_fraction=0.6,
        continuity_neighbors=8,
        config=GridSupplementConfig(cell_size=16),
    )

    assert result.candidate_cells == 2
    assert result.merged_cells == 1
    assert len(result.regions) == 1
    assert result.regions[0].support_count == 16 * 32


def test_grid_supplement_merges_similar_continuous_neighbors() -> None:
    image, points = _flat_view(height=16, width=32)
    result = _build_without_ellipses(image, points)

    assert result.candidate_cells == 2
    assert result.merged_cells == 1
    assert len(result.regions) == 1
    assert result.regions[0].support_count == 16 * 32


def test_grid_supplement_does_not_merge_across_depth_jump() -> None:
    image, points = _flat_view(height=16, width=32)
    points[:, 16:, 2] += 5.0
    result = _build_without_ellipses(image, points)

    assert result.candidate_cells == 2
    assert result.merged_cells == 0
    assert len(result.regions) == 2


def test_grid_supplement_does_not_merge_different_colors() -> None:
    image, points = _flat_view(height=16, width=32)
    image[:, :16] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    image[:, 16:] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    result = _build_without_ellipses(image, points)

    assert result.candidate_cells == 2
    assert result.merged_cells == 0
    assert len(result.regions) == 2


def test_grid_supplement_rejects_l_shaped_component() -> None:
    image, points = _flat_view(height=32, width=32)
    covered_cell = PixelRegion(
        x0=16,
        y0=16,
        mask=np.ones((16, 16), dtype=bool),
        source="ellipse",
        score=2.0,
    )
    result = build_grid_supplement(
        image,
        points,
        [covered_cell],
        image_valid_mask=np.ones((32, 32), dtype=bool),
        ellipse_min_valid_points=16,
        ellipse_min_valid_fraction=0.6,
        continuity_neighbors=8,
        config=GridSupplementConfig(cell_size=16, convex_fill_ratio_min=0.9),
    )

    assert result.candidate_cells == 3
    assert result.merged_cells == 1
    assert result.shape_rejected_merges == 1
    assert sorted(region.support_count for region in result.regions) == [256, 512]


def test_grid_supplement_can_allow_l_shape_with_lower_fill_ratio() -> None:
    image, points = _flat_view(height=32, width=32)
    covered_cell = PixelRegion(
        x0=16,
        y0=16,
        mask=np.ones((16, 16), dtype=bool),
        source="ellipse",
        score=2.0,
    )
    result = build_grid_supplement(
        image,
        points,
        [covered_cell],
        image_valid_mask=np.ones((32, 32), dtype=bool),
        ellipse_min_valid_points=16,
        ellipse_min_valid_fraction=0.6,
        continuity_neighbors=8,
        config=GridSupplementConfig(cell_size=16, convex_fill_ratio_min=0.8),
    )

    assert result.candidate_cells == 3
    assert result.merged_cells == 2
    assert result.shape_rejected_merges == 0
    assert len(result.regions) == 1
    assert result.regions[0].support_count == 3 * 16 * 16


def test_grid_supplement_rejects_component_with_grid_hole() -> None:
    image, points = _flat_view(height=48, width=48)
    covered_center = PixelRegion(
        x0=16,
        y0=16,
        mask=np.ones((16, 16), dtype=bool),
        source="ellipse",
        score=2.0,
    )
    result = build_grid_supplement(
        image,
        points,
        [covered_center],
        image_valid_mask=np.ones((48, 48), dtype=bool),
        ellipse_min_valid_points=16,
        ellipse_min_valid_fraction=0.6,
        continuity_neighbors=8,
        config=GridSupplementConfig(
            cell_size=16,
            max_component_pixels=3000,
            convex_fill_ratio_min=0.0,
            require_hole_free=True,
        ),
    )

    assert result.candidate_cells == 8
    assert result.shape_rejected_merges > 0
    assert len(result.regions) > 1


def test_region_fit_uses_largest_component_centroid_and_mean_color() -> None:
    image, points = _flat_view(height=8, width=8)
    points[:, 5:, 2] += 5.0
    image[:, :5] = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
    region = PixelRegion(
        x0=0,
        y0=0,
        mask=np.ones((8, 8), dtype=bool),
        source="grid",
        score=1.0,
    )
    result = fit_regions(
        points,
        [region],
        colors=image,
        min_valid_points=2,
        min_valid_fraction=0.0,
        continuity_neighbors=8,
        continuity_ratio_max=3.0,
        device="cpu",
        pixel_budget=1000,
    )

    expected_points = points[:, :5].reshape(-1, 3)
    assert result.valid.tolist() == [True]
    assert result.valid_counts.tolist() == [8 * 5]
    assert np.allclose(result.means[0], np.mean(expected_points, axis=0), atol=1.0e-6)
    assert np.allclose(result.mean_colors[0], [0.2, 0.4, 0.6], atol=1.0e-6)


def _build_without_ellipses(
    image: np.ndarray,
    points: np.ndarray,
):
    return build_grid_supplement(
        image,
        points,
        [],
        image_valid_mask=np.ones(points.shape[:2], dtype=bool),
        ellipse_min_valid_points=16,
        ellipse_min_valid_fraction=0.6,
        continuity_neighbors=8,
        config=GridSupplementConfig(cell_size=16),
    )


def _flat_view(*, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:height, :width]
    points = np.stack(
        [0.01 * xx, 0.01 * yy, np.ones_like(xx, dtype=np.float64)],
        axis=-1,
    ).astype(np.float32)
    image = np.full((height, width, 3), 0.5, dtype=np.float32)
    return image, points
