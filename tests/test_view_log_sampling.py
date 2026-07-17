from __future__ import annotations

import numpy as np
from PIL import Image

from init.ellipses import ellipse_mask
from init.types import EllipseKeypoints
from scripts.view_log_sampling import (
    CHANNEL_COLORS,
    add_header,
    build_support_counts,
    concatenate_horizontally,
    concatenate_vertically,
    dominant_channel_map,
    keypoint_channel_counts,
    lab_blur_to_image,
    render_channel_contributions,
    render_log_response,
    render_sampling_overlay,
)


def make_keypoints() -> EllipseKeypoints:
    return EllipseKeypoints(
        view_ids=np.asarray([0], dtype=np.int64),
        us=np.asarray([5], dtype=np.int64),
        vs=np.asarray([4], dtype=np.int64),
        scores=np.asarray([0.2], dtype=np.float32),
        sigmas=np.asarray([2.0], dtype=np.float32),
        levels=np.asarray([0], dtype=np.int64),
        ellipse_matrices=np.asarray([[[9.0, 0.0], [0.0, 4.0]]], dtype=np.float32),
        ellipse_areas=np.asarray([6.0 * np.pi], dtype=np.float32),
    )


def test_support_counts_match_discrete_ellipse_and_valid_mask() -> None:
    keypoints = make_keypoints()
    valid = np.ones((9, 11), dtype=bool)
    valid[4, 7] = False
    counts = build_support_counts(
        valid.shape,
        keypoints=keypoints,
        indices=np.asarray([0]),
        valid_mask=valid,
    )
    expected = (
        ellipse_mask(
            valid.shape,
            u=5,
            v=4,
            ellipse_matrix=keypoints.ellipse_matrices[0],
        )
        & valid
    )
    assert counts.dtype == np.uint16
    assert np.array_equal(counts > 0, expected)


def test_log_and_sampling_panels_have_expected_dimensions() -> None:
    keypoints = make_keypoints()
    shape = (9, 11)
    valid = np.ones(shape, dtype=bool)
    response = np.zeros(shape, dtype=np.float32)
    response[4, 5] = 1.0
    response[2, 2] = 0.5
    raw = np.zeros(shape, dtype=bool)
    raw[4, 5] = True
    counts = build_support_counts(
        shape,
        keypoints=keypoints,
        indices=np.asarray([0]),
        valid_mask=valid,
    )
    dominant = np.full(shape, 1, dtype=np.uint8)

    heatmap = render_log_response(
        response,
        valid_mask=valid,
        raw_extrema=raw,
        selected_us=keypoints.us,
        selected_vs=keypoints.vs,
    )
    overlay = render_sampling_overlay(
        Image.new("RGB", (shape[1], shape[0]), (100, 100, 100)),
        keypoints=keypoints,
        indices=np.asarray([0]),
        support_counts=counts,
        dominant_channels=dominant,
    )
    assert heatmap.size == (shape[1], shape[0])
    assert overlay.size == heatmap.size
    assert np.asarray(heatmap)[2, 2, 0] > np.asarray(heatmap)[2, 2, 2]
    assert np.asarray(heatmap)[4, 5].max() == 255
    assert np.any(np.all(np.asarray(overlay) == CHANNEL_COLORS[1], axis=-1))

    header = add_header(overlay, "test")
    row = concatenate_horizontally([header, header])
    sheet = concatenate_vertically([row, row])
    assert header.size == (shape[1], shape[0] + 28)
    assert row.size == (shape[1] * 2, shape[0] + 28)
    assert sheet.size == (shape[1] * 2, (shape[0] + 28) * 2)


def test_multichannel_diagnostic_panels_and_counts() -> None:
    height, width = 8, 10
    responses = np.zeros((3, height, width), dtype=np.float32)
    responses[0, :, :3] = 2.0
    responses[1, :, 3:7] = 3.0
    responses[2, :, 7:] = 4.0
    dominant = dominant_channel_map(
        responses,
        response_scales=np.ones((3,), dtype=np.float32),
        channel_weights=np.ones((3,), dtype=np.float32),
    )
    assert np.array_equal(np.unique(dominant), [0, 1, 2])
    counts = keypoint_channel_counts(
        dominant,
        us=np.asarray([1, 4, 8]),
        vs=np.asarray([2, 2, 2]),
    )
    assert counts.tolist() == [1, 1, 1]

    valid = np.ones((height, width), dtype=bool)
    lab_blur = np.zeros((3, height, width), dtype=np.float32)
    lab_blur[0] = 0.5
    blur_image = lab_blur_to_image(lab_blur, valid_mask=valid)
    contribution_image = render_channel_contributions(
        np.sqrt(np.sum(responses**2, axis=0)),
        dominant_channels=dominant,
        valid_mask=valid,
        selected_us=np.asarray([1, 4, 8]),
        selected_vs=np.asarray([2, 2, 2]),
    )
    assert blur_image.size == (width, height)
    assert contribution_image.size == (width, height)
