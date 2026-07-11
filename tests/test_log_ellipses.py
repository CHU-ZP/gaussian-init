from __future__ import annotations

import numpy as np

from init.sampling import (
    detect_multiscale_keypoints,
    structure_tensor,
    tensor_to_ellipse,
)


def test_log_detects_bright_and_dark_blobs_at_their_scales() -> None:
    height = width = 128
    yy, xx = np.mgrid[:height, :width]
    image_gray = 0.5 + 0.45 * gaussian_blob(xx, yy, 35, 42, 3.0)
    image_gray -= 0.35 * gaussian_blob(xx, yy, 91, 78, 7.0)
    image = np.repeat(image_gray[..., None], 3, axis=-1).astype(np.float32)
    world_points = np.stack([xx / width, yy / width, np.ones_like(xx)], axis=-1).astype(np.float32)
    confidence = np.ones((height, width), dtype=np.float32)

    keypoints = detect_multiscale_keypoints(
        view_id=2,
        image=image,
        confidence=confidence,
        world_points=world_points,
        sigmas=[1.0, 2.0, 3.0, 4.5, 7.0, 10.0, 14.0],
        response_threshold=0.005,
        max_keypoints=64,
        min_distance=4,
        nms_radius_factor=3.0,
        structure_sigma_factor=1.5,
        ellipse_radius_factor=2.5,
        min_ellipse_area=12.0,
        max_ellipse_area=800.0,
        max_axis_ratio=8.0,
        confidence_threshold=0.2,
    )

    small = nearest_keypoint(keypoints.us, keypoints.vs, 35, 42)
    large = nearest_keypoint(keypoints.us, keypoints.vs, 91, 78)
    assert small[0] <= 1.0
    assert large[0] <= 1.0
    assert 2.0 <= keypoints.sigmas[small[1]] <= 4.5
    assert 4.5 <= keypoints.sigmas[large[1]] <= 10.0
    assert np.all(keypoints.view_ids == 2)
    assert np.all(keypoints.ellipse_areas >= 12.0 - 1.0e-4)
    assert np.all(keypoints.ellipse_areas <= 800.0 + 1.0e-3)
    assert len(keypoints) <= 12


def test_flat_image_has_no_log_extrema() -> None:
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)
    points = np.zeros((32, 32, 3), dtype=np.float32)
    confidence = np.ones((32, 32), dtype=np.float32)
    keypoints = detect_multiscale_keypoints(
        view_id=0,
        image=image,
        confidence=confidence,
        world_points=points,
        sigmas=[1.0, 2.0, 4.0],
        response_threshold=1.0e-4,
        max_keypoints=16,
        min_distance=1,
        nms_radius_factor=1.0,
        structure_sigma_factor=1.5,
        ellipse_radius_factor=2.5,
        min_ellipse_area=4.0,
        max_ellipse_area=100.0,
        max_axis_ratio=4.0,
        confidence_threshold=0.2,
    )
    assert len(keypoints) == 0


def test_padding_mask_does_not_create_log_seam_keypoints() -> None:
    image = np.ones((64, 64, 3), dtype=np.float32)
    image[16:48] = 0.2
    content_mask = np.zeros((64, 64), dtype=bool)
    content_mask[16:48] = True
    points = np.zeros((64, 64, 3), dtype=np.float32)
    confidence = np.ones((64, 64), dtype=np.float32)
    keypoints = detect_multiscale_keypoints(
        view_id=0,
        image=image,
        confidence=confidence,
        world_points=points,
        sigmas=[1.0, 2.0, 4.0, 8.0],
        response_threshold=1.0e-4,
        max_keypoints=64,
        min_distance=2,
        nms_radius_factor=3.0,
        structure_sigma_factor=1.5,
        ellipse_radius_factor=2.5,
        min_ellipse_area=4.0,
        max_ellipse_area=200.0,
        max_axis_ratio=4.0,
        confidence_threshold=0.2,
        image_valid_mask=content_mask,
    )
    assert len(keypoints) == 0


def test_structure_tensor_ellipse_follows_edge_and_clamps_area() -> None:
    ramp = np.broadcast_to(np.arange(64, dtype=np.float32)[None, :], (64, 64))
    tensor = structure_tensor(ramp, integration_sigma=2.0)[32, 32]
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
