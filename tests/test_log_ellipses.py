from __future__ import annotations

import numpy as np

from init.sampling import (
    EllipseMergeConfig,
    build_log_scale_space,
    detect_multiscale_keypoints,
    merge_same_scale_ellipses,
    multichannel_structure_tensor,
    normalized_lab_to_rgb,
    rgb_to_normalized_lab,
    robust_channel_response_scales,
    scale_space_maxima,
    tensor_to_ellipse,
)
from init.types import EllipseKeypoints


def test_log_detects_bright_and_dark_blobs_at_their_scales() -> None:
    height = width = 128
    yy, xx = np.mgrid[:height, :width]
    image_gray = 0.5 + 0.45 * gaussian_blob(xx, yy, 35, 42, 3.0)
    image_gray -= 0.35 * gaussian_blob(xx, yy, 91, 78, 7.0)
    image = np.repeat(image_gray[..., None], 3, axis=-1).astype(np.float32)
    world_points = np.stack([xx / width, yy / width, np.ones_like(xx)], axis=-1).astype(np.float32)

    keypoints = detect_multiscale_keypoints(
        view_id=2,
        image=image,
        world_points=world_points,
        sigmas=[1.0, 2.0, 3.0, 4.5, 7.0, 10.0, 14.0],
        response_threshold=1.0,
        max_keypoints=64,
        structure_sigma_factor=1.5,
        ellipse_radius_factor=2.5,
        min_ellipse_area=12.0,
        max_ellipse_area=800.0,
        max_axis_ratio=4.0,
    )

    small = nearest_keypoint(keypoints.us, keypoints.vs, 35, 42)
    large = nearest_keypoint(keypoints.us, keypoints.vs, 91, 78)
    assert small[0] <= 1.0
    assert large[0] <= 1.0
    assert 2.0 <= keypoints.sigmas[small[1]] <= 4.5
    assert 4.5 <= keypoints.sigmas[large[1]] <= 10.0
    assert np.all(keypoints.view_ids == 2)
    assert np.all(keypoints.ellipse_areas >= 12.0 - 1.0e-4)
    # Raw supports stop at 800 px, while accepted same-scale merges may grow by 1.5x.
    assert np.all(keypoints.ellipse_areas <= 1200.0 + 1.0e-3)
    assert len(keypoints) <= 64


def test_flat_image_has_no_log_extrema() -> None:
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)
    points = np.zeros((32, 32, 3), dtype=np.float32)
    keypoints = detect_multiscale_keypoints(
        view_id=0,
        image=image,
        world_points=points,
        sigmas=[1.0, 2.0, 4.0],
        response_threshold=1.0e-4,
        max_keypoints=16,
        structure_sigma_factor=1.5,
        ellipse_radius_factor=2.5,
        min_ellipse_area=4.0,
        max_ellipse_area=100.0,
        max_axis_ratio=4.0,
    )
    assert len(keypoints) == 0


def test_same_scale_similar_ellipses_are_moment_merged() -> None:
    candidates, lab, points, valid = make_merge_inputs(
        centers=[(8, 8), (10, 8)],
        levels=[0, 0],
        colors=[(0.5, 0.0, 0.0), (0.5, 0.0, 0.0)],
    )
    merged = merge_same_scale_ellipses(
        candidates,
        normalized_lab=lab,
        world_points=points,
        valid_mask=valid,
        config=EllipseMergeConfig(),
        max_keypoints=16,
        max_axis_ratio=4.0,
    )

    assert len(merged) == 1
    assert merged.us.tolist() == [8]
    assert merged.levels.tolist() == [0]
    assert merged.ellipse_areas[0] > candidates.ellipse_areas[0]


def test_ellipse_merge_rejects_color_and_cross_scale_pairs() -> None:
    candidates, lab, points, valid = make_merge_inputs(
        centers=[(8, 8), (10, 8), (8, 8)],
        levels=[0, 0, 1],
        colors=[(0.5, 0.0, 0.0), (0.9, 0.0, 0.0), (0.5, 0.0, 0.0)],
    )
    merged = merge_same_scale_ellipses(
        candidates,
        normalized_lab=lab,
        world_points=points,
        valid_mask=valid,
        config=EllipseMergeConfig(color_delta_e_max=20.0),
        max_keypoints=16,
        max_axis_ratio=4.0,
    )

    assert len(merged) == 3
    assert sorted(merged.levels.tolist()) == [0, 0, 1]


def test_ellipse_merge_rejects_vggt_depth_discontinuity() -> None:
    candidates, lab, points, valid = make_merge_inputs(
        centers=[(8, 8), (10, 8)],
        levels=[0, 0],
        colors=[(0.5, 0.0, 0.0), (0.5, 0.0, 0.0)],
    )
    # Both centers remain on the same surface; only their connecting path crosses
    # a 3D spike. A center-to-center test would miss this discontinuity.
    points[:, 9, 2] = 10.0
    merged = merge_same_scale_ellipses(
        candidates,
        normalized_lab=lab,
        world_points=points,
        valid_mask=valid,
        config=EllipseMergeConfig(),
        max_keypoints=16,
        max_axis_ratio=4.0,
    )

    assert len(merged) == 2


def test_constrained_union_preserves_valid_submerge_when_chain_is_too_large() -> None:
    candidates, lab, points, valid = make_merge_inputs(
        centers=[(7, 8), (10, 8), (13, 8)],
        levels=[0, 0, 0],
        colors=[(0.5, 0.0, 0.0)] * 3,
    )
    merged = merge_same_scale_ellipses(
        candidates,
        normalized_lab=lab,
        world_points=points,
        valid_mask=valid,
        config=EllipseMergeConfig(
            iou_min=0.3,
            merged_area_factor_max=1.5,
            merged_area_absolute_max=105.0,
        ),
        max_keypoints=16,
        max_axis_ratio=4.0,
    )

    assert len(merged) == 2
    assert sorted(merged.us.tolist()) == [7, 13]
    assert np.max(merged.ellipse_areas) > np.max(candidates.ellipse_areas)


def test_relaxed_color_and_removed_axis_scale_gate_allow_overlap_merge() -> None:
    candidates, lab, points, valid = make_merge_inputs(
        centers=[(8, 8), (9, 8)],
        levels=[0, 0],
        colors=[(0.50, 0.0, 0.0), (0.75, 0.0, 0.0)],
    )
    candidates.ellipse_matrices[1] = np.diag([10.24, 10.24])
    candidates.ellipse_areas[1] = 10.24 * np.pi

    merged = merge_same_scale_ellipses(
        candidates,
        normalized_lab=lab,
        world_points=points,
        valid_mask=valid,
        config=EllipseMergeConfig(),
        max_keypoints=16,
        max_axis_ratio=4.0,
    )

    assert len(merged) == 1


def test_merged_area_can_exceed_raw_candidate_cap_but_stays_relative() -> None:
    raw_area = 800.0
    matrix_value = raw_area / np.pi
    candidates, lab, points, valid = make_merge_inputs(
        centers=[(24, 32), (32, 32)],
        levels=[0, 0],
        colors=[(0.5, 0.0, 0.0)] * 2,
        image_size=64,
        matrix_value=matrix_value,
    )

    merged = merge_same_scale_ellipses(
        candidates,
        normalized_lab=lab,
        world_points=points,
        valid_mask=valid,
        config=EllipseMergeConfig(),
        max_keypoints=16,
        max_axis_ratio=4.0,
    )

    assert len(merged) == 1
    assert raw_area < merged.ellipse_areas[0] <= 1.5 * raw_area


def test_padding_mask_does_not_create_log_seam_keypoints() -> None:
    image = np.ones((64, 64, 3), dtype=np.float32)
    image[16:48] = 0.2
    content_mask = np.zeros((64, 64), dtype=bool)
    content_mask[16:48] = True
    points = np.zeros((64, 64, 3), dtype=np.float32)
    keypoints = detect_multiscale_keypoints(
        view_id=0,
        image=image,
        world_points=points,
        sigmas=[1.0, 2.0, 4.0, 8.0],
        response_threshold=1.0e-4,
        max_keypoints=64,
        structure_sigma_factor=1.5,
        ellipse_radius_factor=2.5,
        min_ellipse_area=4.0,
        max_ellipse_area=200.0,
        max_axis_ratio=4.0,
        image_valid_mask=content_mask,
    )
    assert len(keypoints) == 0


def test_structure_tensor_ellipse_follows_edge_and_clamps_area() -> None:
    ramp = np.broadcast_to(np.arange(64, dtype=np.float32)[None, :], (64, 64))
    tensor = multichannel_structure_tensor(ramp[None, ...], integration_sigma=2.0)[32, 32]
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    gradient_direction = eigenvectors[:, 1]
    assert eigenvalues[1] > 100.0 * max(float(eigenvalues[0]), 1.0e-12)
    assert abs(float(np.dot(gradient_direction, np.asarray([1.0, 0.0])))) > 0.99

    ellipse, area = tensor_to_ellipse(
        tensor,
        sigma=3.0,
        radius_factor=3.0,
        min_area=40.0,
        max_area=120.0,
        max_axis_ratio=6.0,
    )
    ellipse_values, ellipse_vectors = np.linalg.eigh(ellipse)
    long_axis = ellipse_vectors[:, 1]
    assert 40.0 <= area <= 120.0 + 1.0e-4
    assert abs(float(np.dot(long_axis, np.asarray([0.0, 1.0])))) > 0.99
    assert ellipse_values[1] / ellipse_values[0] <= 6.0**2 + 1.0e-3


def test_normalized_lab_reference_colors() -> None:
    image = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.8, 0.2, 0.6]]],
        dtype=np.float32,
    )
    lab = rgb_to_normalized_lab(image)
    assert np.allclose(lab[0, 0], 0.0, atol=1.0e-6)
    assert np.isclose(lab[0, 1, 0], 1.0, atol=1.0e-5)
    assert np.allclose(lab[0, 1, 1:], 0.0, atol=1.0e-4)
    assert np.allclose(normalized_lab_to_rgb(lab), image, atol=2.0e-5)


def test_lab_log_detects_isoluminant_color_blob() -> None:
    height = width = 96
    yy, xx = np.mgrid[:height, :width]
    blob = gaussian_blob(xx, yy, 48, 45, 5.0)[..., None]
    red = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    same_gray_green = np.asarray([0.0, 0.299 / 0.587, 0.0], dtype=np.float32)
    image = (same_gray_green * (1.0 - blob) + red * blob).astype(np.float32)
    grayscale = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    assert np.ptp(grayscale) < 1.0e-6

    world_points = np.stack([xx / width, yy / height, np.ones_like(xx)], axis=-1).astype(np.float32)
    lab = detect_multiscale_keypoints(
        view_id=0,
        image=image,
        world_points=world_points,
        sigmas=[1.0, 2.0, 3.0, 4.5, 7.0, 10.0],
        response_threshold=1.0,
        max_keypoints=64,
        structure_sigma_factor=1.5,
        ellipse_radius_factor=2.5,
        min_ellipse_area=4.0,
        max_ellipse_area=500.0,
        max_axis_ratio=4.0,
        response_mad_epsilon=0.01,
    )
    distance, index = nearest_keypoint(lab.us, lab.vs, 48, 45)
    assert distance <= 1.0
    assert 3.0 <= lab.sigmas[index] <= 7.0


def test_lab_structure_tensor_uses_isoluminant_color_edge() -> None:
    height = width = 48
    red = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    same_gray_green = np.asarray([0.0, 0.299 / 0.587, 0.0], dtype=np.float32)
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:, : width // 2] = red
    image[:, width // 2 :] = same_gray_green
    grayscale = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    gray_tensor = multichannel_structure_tensor(grayscale[None, ...], integration_sigma=2.0)[24, 24]
    lab_channels = np.moveaxis(rgb_to_normalized_lab(image), -1, 0)
    color_tensor = multichannel_structure_tensor(
        lab_channels,
        integration_sigma=2.0,
        channel_weights=np.ones((3,), dtype=np.float32),
    )[24, 24]
    eigenvalues, eigenvectors = np.linalg.eigh(color_tensor)
    assert np.trace(gray_tensor) < 1.0e-10
    assert eigenvalues[1] > 100.0 * max(float(eigenvalues[0]), 1.0e-12)
    assert abs(float(np.dot(eigenvectors[:, 1], np.asarray([1.0, 0.0])))) > 0.99


def test_mad_response_scaling_and_magnitude_maxima() -> None:
    responses = np.asarray([[[[0.0, 1.0, 2.0]]]], dtype=np.float32)
    scales = robust_channel_response_scales(
        responses,
        valid_mask=np.ones((1, 3), dtype=bool),
        epsilon=0.25,
    )
    assert np.allclose(scales, [1.25])

    scale_space = np.zeros((3, 5, 5), dtype=np.float32)
    scale_space[1, 2, 2] = 3.0
    scale_space[1, 1, 1] = -4.0
    maxima = scale_space_maxima(scale_space)
    assert maxima[1, 2, 2]
    assert not maxima[1, 1, 1]


def test_lab_scale_space_exposes_fused_magnitude_metadata() -> None:
    image = np.full((16, 16, 3), 0.5, dtype=np.float32)
    scale_space = build_log_scale_space(
        image,
        sigmas=[1.0, 2.0, 4.0],
        valid_mask=np.ones((16, 16), dtype=bool),
        chroma_weight=0.75,
        response_mad_epsilon=0.01,
    )
    assert scale_space.channel_names == ("L", "a", "b")
    assert scale_space.blurred_channels.shape == (3, 5, 16, 16)
    assert scale_space.channel_responses.shape == (3, 5, 16, 16)
    assert np.allclose(scale_space.structure_weights, [1.0, 0.75, 0.75])
    assert np.all(scale_space.response_scales >= 0.01)
    assert np.all(scale_space.responses >= 0.0)


def make_merge_inputs(
    *,
    centers: list[tuple[int, int]],
    levels: list[int],
    colors: list[tuple[float, float, float]],
    image_size: int = 24,
    matrix_value: float = 25.0,
) -> tuple[EllipseKeypoints, np.ndarray, np.ndarray, np.ndarray]:
    height = width = image_size
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([0.01 * xx, 0.01 * yy, np.ones_like(xx)], axis=-1).astype(np.float32)
    valid = np.ones((height, width), dtype=bool)
    lab = np.zeros((height, width, 3), dtype=np.float32)
    for (u, v), color in zip(centers, colors, strict=True):
        lab[v, u] = np.asarray(color, dtype=np.float32)
    matrix = np.diag(np.asarray([matrix_value, matrix_value], dtype=np.float32))
    candidates = EllipseKeypoints(
        view_ids=np.zeros((len(centers),), dtype=np.int64),
        us=np.asarray([center[0] for center in centers], dtype=np.int64),
        vs=np.asarray([center[1] for center in centers], dtype=np.int64),
        scores=np.arange(len(centers), 0, -1, dtype=np.float32),
        sigmas=np.asarray([1.0 if level == 0 else 2.0 for level in levels], dtype=np.float32),
        levels=np.asarray(levels, dtype=np.int64),
        ellipse_matrices=np.repeat(matrix[None], len(centers), axis=0),
        ellipse_areas=np.full((len(centers),), matrix_value * np.pi, dtype=np.float32),
    )
    return candidates, lab, points, valid


def gaussian_blob(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float,
) -> np.ndarray:
    return np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2))


def nearest_keypoint(
    us: np.ndarray,
    vs: np.ndarray,
    target_u: int,
    target_v: int,
) -> tuple[float, int]:
    distances = np.hypot(us - target_u, vs - target_v)
    index = int(np.argmin(distances))
    return float(distances[index]), index
