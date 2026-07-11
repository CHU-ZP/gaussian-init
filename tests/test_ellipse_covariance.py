from __future__ import annotations

import numpy as np

from init.ellipses import compute_covariances, ellipse_mask
from init.gaussian_params import (
    rgb_to_sh_dc,
    rotation_matrix_to_quaternion,
    scale_quat_to_covariance,
    sh_dc_to_rgb,
)
from init.fusion import voxel_fuse
from init.filters import PCAFilterConfig
from init.pca import decompose_covariance
from init.types import GaussianProposals


def test_ellipse_covariance_matches_numpy_and_ignores_bounding_box_corners() -> None:
    height = width = 31
    yy, xx = np.mgrid[:height, :width]
    world_points = np.stack(
        [0.1 * xx + 0.03 * yy, -0.02 * xx + 0.08 * yy, np.ones_like(xx)],
        axis=-1,
    ).astype(np.float32)
    confidence = np.ones((height, width), dtype=np.float32)
    angle = np.deg2rad(30.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    ellipse = rotation @ np.diag(np.asarray([7.0**2, 3.0**2], dtype=np.float32)) @ rotation.T
    mask = ellipse_mask((height, width), u=15, v=15, ellipse_matrix=ellipse)

    extent_x = int(np.ceil(np.sqrt(ellipse[0, 0])))
    extent_y = int(np.ceil(np.sqrt(ellipse[1, 1])))
    for y in (15 - extent_y, 15 + extent_y):
        for x in (15 - extent_x, 15 + extent_x):
            if not mask[y, x]:
                world_points[y, x, 2] = 100.0

    results = compute_covariances(
        world_points,
        confidence,
        np.asarray([15]),
        np.asarray([15]),
        ellipse[None],
        confidence_threshold=0.2,
        min_valid_points=8,
        min_valid_fraction=1.0,
        max_center_distance=None,
        confidence_weighted=False,
        device="cpu",
        pixel_budget=10000,
    )
    deltas = world_points[mask] - world_points[15, 15]
    expected = deltas.T @ deltas / deltas.shape[0]
    assert results.valid.tolist() == [True]
    assert results.valid_counts[0] == np.count_nonzero(mask)
    assert np.allclose(results.covariances[0], expected, atol=1.0e-6)
    assert results.covariances[0, 2, 2] == 0.0


def test_covariance_rejects_clipped_support_and_invalid_ellipse() -> None:
    points = np.zeros((21, 21, 3), dtype=np.float32)
    confidence = np.ones((21, 21), dtype=np.float32)
    ellipse = np.asarray([[[49.0, 0.0], [0.0, 49.0]]], dtype=np.float32)
    results = compute_covariances(
        points,
        confidence,
        np.asarray([0]),
        np.asarray([0]),
        ellipse,
        confidence_threshold=0.2,
        min_valid_points=2,
        min_valid_fraction=0.75,
        max_center_distance=None,
        confidence_weighted=False,
        device="cpu",
        pixel_budget=10000,
    )
    assert not results.valid[0]
    assert results.valid_counts[0] < 0.5 * results.support_counts[0]

    invalid = np.asarray([[[-4.0, 0.0], [0.0, -4.0]]], dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "positive definite"):
        compute_covariances(
            points,
            confidence,
            np.asarray([10]),
            np.asarray([10]),
            invalid,
            confidence_threshold=0.2,
            min_valid_points=2,
            min_valid_fraction=0.5,
            max_center_distance=None,
            confidence_weighted=False,
            device="cpu",
            pixel_budget=10000,
        )


def test_covariance_scale_quaternion_and_sh_round_trip() -> None:
    angle = np.deg2rad(27.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    expected_scales = np.asarray([0.3, 0.12, 0.04], dtype=np.float32)
    covariance = rotation @ np.diag(expected_scales**2) @ rotation.T
    result = decompose_covariance(covariance, eigenvalue_epsilon=0.0)
    quat = rotation_matrix_to_quaternion(result.basis)
    reconstructed = scale_quat_to_covariance(result.scales, quat)
    assert np.isclose(np.linalg.norm(quat), 1.0)
    assert np.allclose(reconstructed, covariance, atol=1.0e-6)

    rgb = np.asarray([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]], dtype=np.float32)
    sh_dc = rgb_to_sh_dc(rgb)
    assert sh_dc[0, 0] < 0.0
    assert sh_dc[0, 1] == 0.0
    assert np.allclose(sh_dc_to_rgb(sh_dc), rgb, atol=1.0e-7)


def test_single_proposal_fusion_does_not_add_regularization_twice() -> None:
    covariance = np.diag(np.asarray([0.09, 0.01, 1.0e-8], dtype=np.float32))
    pca = decompose_covariance(covariance, eigenvalue_epsilon=0.0)
    proposal = GaussianProposals.from_lists(
        means=[np.asarray([0.0, 0.0, 1.0], dtype=np.float32)],
        covariances=[pca.covariance],
        scales=[pca.scales],
        quats=[rotation_matrix_to_quaternion(pca.basis)],
        sh_dc=[np.asarray([-0.4, 0.2, 0.7], dtype=np.float32)],
        opacities=[0.1],
        confidences=[0.9],
        view_ids=[0],
        scores=[1.0],
    )
    fused = voxel_fuse(proposal, voxel_size=0.1, eigenvalue_epsilon=1.0e-8)
    assert np.allclose(fused.covariances, proposal.covariances, atol=1.0e-12)
    assert np.allclose(fused.sh_dc, proposal.sh_dc)


def test_fusion_reapplies_scale_filter_after_center_spread() -> None:
    covariance = np.diag(np.asarray([0.01, 0.01, 0.01], dtype=np.float32))
    pca = decompose_covariance(covariance, eigenvalue_epsilon=0.0)
    proposals = GaussianProposals.from_lists(
        means=[
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([3.0, 0.0, 0.0], dtype=np.float32),
        ],
        covariances=[pca.covariance, pca.covariance],
        scales=[pca.scales, pca.scales],
        quats=[
            rotation_matrix_to_quaternion(pca.basis),
            rotation_matrix_to_quaternion(pca.basis),
        ],
        sh_dc=[np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)],
        opacities=[0.1, 0.1],
        confidences=[1.0, 1.0],
        view_ids=[0, 1],
        scores=[1.0, 1.0],
    )
    fused = voxel_fuse(
        proposals,
        voxel_size=4.0,
        pca_filter=PCAFilterConfig(scale_min=1.0e-5, scale_max=1.0, condition_max=1.0e8),
    )
    assert len(fused) == 0
