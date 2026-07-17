from __future__ import annotations

import numpy as np
from PIL import Image

from init.sampling import SamplingConfig
from scripts.view_ellipse_merging import (
    build_raw_ellipse_candidates,
    render_candidate_ellipses,
)


def test_build_raw_ellipse_candidates_constructs_same_scale_ellipses() -> None:
    height = width = 15
    yy, xx = np.mgrid[:height, :width]
    blurred = (0.03 * xx + 0.02 * yy).astype(np.float32)
    response = np.zeros((height, width), dtype=np.float32)
    response[7, 8] = -0.25
    extrema = np.zeros((height, width), dtype=bool)
    extrema[7, 8] = True
    keypoints = build_raw_ellipse_candidates(
        view_id=3,
        level=2,
        sigma=2.5,
        blurred_channels=blurred[None, ...],
        response=response,
        extrema=extrema,
        valid_mask=np.ones((height, width), dtype=bool),
        structure_weights=np.ones((1,), dtype=np.float32),
        sampling=SamplingConfig(max_axis_ratio=4.0),
    )
    assert len(keypoints) == 1
    assert keypoints.view_ids.tolist() == [3]
    assert keypoints.levels.tolist() == [2]
    assert keypoints.sigmas.tolist() == [2.5]
    assert keypoints.scores.tolist() == [0.25]
    assert np.linalg.eigvalsh(keypoints.ellipse_matrices[0]).min() > 0.0


def test_raw_candidate_ellipse_rendering() -> None:
    height = width = 15
    blurred = np.zeros((height, width), dtype=np.float32)
    response = np.zeros((height, width), dtype=np.float32)
    response[7, 8] = 0.2
    extrema = np.zeros((height, width), dtype=bool)
    extrema[7, 8] = True
    keypoints = build_raw_ellipse_candidates(
        view_id=0,
        level=0,
        sigma=1.0,
        blurred_channels=blurred[None, ...],
        response=response,
        extrema=extrema,
        valid_mask=np.ones((height, width), dtype=bool),
        structure_weights=np.ones((1,), dtype=np.float32),
        sampling=SamplingConfig(),
    )
    image = Image.new("RGB", (width, height), (80, 80, 80))
    ellipses = render_candidate_ellipses(
        image,
        keypoints=keypoints,
        dominant_channels=np.zeros((height, width), dtype=np.uint8),
    )
    rendered = np.asarray(ellipses)
    assert ellipses.size == (width, height)
    assert tuple(rendered[7, 8]) == (255, 255, 255)
    assert np.count_nonzero(np.any(rendered != 80, axis=-1)) > 1
