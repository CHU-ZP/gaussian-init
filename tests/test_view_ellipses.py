from __future__ import annotations

import numpy as np
import pytest

from scripts.view_ellipses import (
    ellipse_axes,
    ellipse_color,
    ellipse_outline,
    parse_view_selection,
)


def test_ellipse_outline_matches_matrix_and_axes() -> None:
    angle = np.deg2rad(30.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    matrix = rotation @ np.diag(np.asarray([16.0, 4.0], dtype=np.float32)) @ rotation.T

    axes, directions = ellipse_axes(matrix)
    outline = np.asarray(
        ellipse_outline(u=12.0, v=9.0, ellipse_matrix=matrix, samples=64),
        dtype=np.float32,
    )
    offsets = outline - np.asarray([12.0, 9.0], dtype=np.float32)
    quadratic = np.einsum("ni,ij,nj->n", offsets, np.linalg.inv(matrix), offsets)

    assert np.allclose(axes, [4.0, 2.0], atol=1.0e-5)
    assert abs(float(np.linalg.det(directions))) == pytest.approx(1.0, abs=1.0e-5)
    assert np.allclose(quadratic, 1.0, atol=1.0e-5)


def test_ellipse_view_selection_and_ratio_colors() -> None:
    assert parse_view_selection(None, views=4) == [0, 1, 2, 3]
    assert parse_view_selection("all", views=4) == [0, 1, 2, 3]
    assert parse_view_selection("0,2-4,2", views=5) == [0, 2, 3, 4]
    with pytest.raises(ValueError, match="must lie"):
        parse_view_selection("5", views=5)

    assert ellipse_color(1.5, highlight_ratio=4.0) == (48, 224, 96)
    assert ellipse_color(2.5, highlight_ratio=4.0) == (255, 196, 32)
    assert ellipse_color(4.0, highlight_ratio=4.0) == (255, 48, 48)
