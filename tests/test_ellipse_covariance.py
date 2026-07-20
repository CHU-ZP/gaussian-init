from __future__ import annotations

import numpy as np

from init.continuity import global_scene_step_scale
from init.ellipses import compute_covariances, ellipse_mask
from init.gaussian_params import (
    rgb_to_sh_dc,
    rotation_matrix_to_quaternion,
    scale_quat_to_covariance,
    sh_dc_to_rgb,
)
from init.fusion import FusionConfig, similarity_graph_fuse
from init.filters import PCAFilterConfig, valid_pca
from init.pca import decompose_covariance, floor_normal_scale
from init.types import GaussianProposals


def test_ellipse_covariance_matches_numpy_and_ignores_bounding_box_corners() -> None:
    height = width = 31
    yy, xx = np.mgrid[:height, :width]
    world_points = np.stack(
        [0.1 * xx + 0.03 * yy, -0.02 * xx + 0.08 * yy, np.ones_like(xx)],
        axis=-1,
    ).astype(np.float32)
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
        np.asarray([15]),
        np.asarray([15]),
        ellipse[None],
        min_valid_points=8,
        min_valid_fraction=1.0,
        continuity_neighbors=8,
        continuity_ratio_max=4.0,
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
    ellipse = np.asarray([[[49.0, 0.0], [0.0, 49.0]]], dtype=np.float32)
    results = compute_covariances(
        points,
        np.asarray([0]),
        np.asarray([0]),
        ellipse,
        min_valid_points=2,
        min_valid_fraction=0.75,
        continuity_neighbors=8,
        continuity_ratio_max=4.0,
        device="cpu",
        pixel_budget=10000,
    )
    assert not results.valid[0]
    assert results.valid_counts[0] < 0.5 * results.support_counts[0]

    invalid = np.asarray([[[-4.0, 0.0], [0.0, -4.0]]], dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "positive definite"):
        compute_covariances(
            points,
            np.asarray([10]),
            np.asarray([10]),
            invalid,
            min_valid_points=2,
            min_valid_fraction=0.5,
            continuity_neighbors=8,
            continuity_ratio_max=4.0,
            device="cpu",
            pixel_budget=10000,
        )


def test_covariance_keeps_only_center_connected_globally_bounded_surface() -> None:
    height = width = 13
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([0.2 * xx, 0.2 * yy, np.zeros_like(xx)], axis=-1).astype(np.float32)
    points[:, 7:, 2] = 20.0
    ellipse = np.asarray([[[36.0, 0.0], [0.0, 36.0]]], dtype=np.float32)
    support = ellipse_mask((height, width), u=6, v=6, ellipse_matrix=ellipse[0])
    expected_mask = support & (xx < 7)

    kwargs = {
        "min_valid_points": 2,
        "min_valid_fraction": 0.0,
        "continuity_neighbors": 8,
        "continuity_ratio_max": 4.0,
        "device": "cpu",
        "pixel_budget": 10000,
    }
    results = compute_covariances(
        points,
        np.asarray([6]),
        np.asarray([6]),
        ellipse,
        **kwargs,
    )
    deltas = points[expected_mask] - points[6, 6]
    expected = deltas.T @ deltas / deltas.shape[0]
    assert results.valid.tolist() == [True]
    assert results.valid_counts.tolist() == [int(expected_mask.sum())]
    assert np.allclose(results.covariances[0], expected, atol=1.0e-6)
    assert np.isclose(results.continuity_reference_scale, 0.2, atol=1.0e-6)

    scaled = compute_covariances(
        points * 37.0,
        np.asarray([6]),
        np.asarray([6]),
        ellipse,
        **kwargs,
    )
    assert scaled.valid_counts.tolist() == results.valid_counts.tolist()
    assert np.allclose(scaled.covariances[0], results.covariances[0] * 37.0**2, atol=2.0e-4)
    assert np.isclose(
        scaled.continuity_reference_scale,
        37.0 * results.continuity_reference_scale,
        atol=1.0e-5,
    )


def test_global_step_limit_rejects_locally_inflated_depth_bridge() -> None:
    height = width = 9
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([0.2 * xx, 0.2 * yy, np.zeros_like(xx)], axis=-1).astype(np.float32)
    points[:, 4, 2] = 20.0
    ellipse = np.asarray([[[16.0, 0.0], [0.0, 16.0]]], dtype=np.float32)
    support = ellipse_mask((height, width), u=3, v=4, ellipse_matrix=ellipse[0])
    expected_mask = support & (xx < 4)

    reference = global_scene_step_scale(
        points,
        np.ones((height, width), dtype=bool),
        neighbors=8,
    )
    assert np.isclose(reference, 0.2, atol=1.0e-6)

    results = compute_covariances(
        points,
        np.asarray([3]),
        np.asarray([4]),
        ellipse,
        min_valid_points=2,
        min_valid_fraction=0.0,
        continuity_neighbors=8,
        continuity_ratio_max=3.0,
        device="cpu",
        pixel_budget=10000,
    )
    deltas = points[expected_mask] - points[4, 3]
    expected = deltas.T @ deltas / deltas.shape[0]
    assert results.valid.tolist() == [True]
    assert results.valid_counts.tolist() == [int(expected_mask.sum())]
    assert np.allclose(results.covariances[0], expected, atol=1.0e-6)


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


def test_pca_filter_accepts_planes_and_rejects_line_like_fits() -> None:
    config = PCAFilterConfig(
        scale_min=1.0e-5,
        scale_max=2.0,
        min_secondary_eigenvalue_ratio=0.01,
    )
    plane = decompose_covariance(
        np.diag(np.asarray([1.0, 0.04, 1.0e-12], dtype=np.float32)),
        eigenvalue_epsilon=0.0,
    )
    line = decompose_covariance(
        np.diag(np.asarray([1.0, 0.001, 1.0e-8], dtype=np.float32)),
        eigenvalue_epsilon=0.0,
    )
    assert valid_pca(plane, config)
    assert not valid_pca(line, config)


def test_normal_scale_floor_preserves_tangent_eigenpairs() -> None:
    result = decompose_covariance(
        np.diag(np.asarray([0.09, 0.01, 1.0e-10], dtype=np.float32)),
        eigenvalue_epsilon=0.0,
    )
    floored = floor_normal_scale(result, minimum_scale=0.02)
    assert np.allclose(floored.scales, [0.3, 0.1, 0.02], atol=1.0e-7)
    assert np.array_equal(floored.eigenvalues[:2], result.eigenvalues[:2])
    assert np.array_equal(floored.basis, result.basis)
    assert np.allclose(
        floored.covariance,
        floored.basis @ np.diag(floored.eigenvalues) @ floored.basis.T,
        atol=1.0e-8,
    )


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
        view_ids=[0],
        scores=[1.0],
    )
    result = similarity_graph_fuse(
        proposal,
        config=FusionConfig(voxel_size=0.1),
        eigenvalue_epsilon=1.0e-8,
    )
    fused = result.gaussians
    assert np.allclose(fused.covariances, proposal.covariances, atol=1.0e-12)
    assert np.allclose(fused.sh_dc, proposal.sh_dc)
    assert result.stats.candidate_pairs == 0
    assert result.stats.singleton_components == 1


def test_fusion_falls_back_when_merged_component_fails_pca() -> None:
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
        view_ids=[0, 1],
        scores=[1.0, 1.0],
    )
    result = similarity_graph_fuse(
        proposals,
        config=FusionConfig(voxel_size=4.0),
        pca_filter=PCAFilterConfig(
            scale_min=1.0e-5,
            scale_max=1.0,
            min_secondary_eigenvalue_ratio=1.0e-8,
        ),
    )
    assert len(result.gaussians) == 2
    assert np.allclose(result.gaussians.means, proposals.means)
    assert result.stats.compatible_pairs == 1
    assert result.stats.merged_components == 0
    assert result.stats.fallback_components == 1


def test_similarity_graph_fuses_compatible_pair() -> None:
    proposals = _fusion_proposals(
        means=[[0.010, 0.0, 0.0], [0.012, 0.0, 0.0]],
        covariances=[
            np.diag([0.04, 0.01, 0.0025]),
            np.diag([0.04, 0.01, 0.0025]),
        ],
    )
    result = similarity_graph_fuse(proposals, config=FusionConfig(voxel_size=0.1))

    assert len(result.gaussians) == 1
    assert result.gaussians.view_ids.tolist() == [-1]
    assert result.stats.candidate_pairs == 1
    assert result.stats.compatible_pairs == 1
    assert result.stats.merged_components == 1


def test_similarity_graph_rejects_incompatible_normal() -> None:
    proposals = _fusion_proposals(
        means=[[0.010, 0.0, 0.0], [0.011, 0.0, 0.0]],
        covariances=[
            np.diag([0.0001, 0.01, 0.04]),
            np.diag([0.04, 0.01, 0.0001]),
        ],
    )
    result = similarity_graph_fuse(proposals, config=FusionConfig(voxel_size=0.1))

    assert len(result.gaussians) == 2
    assert result.stats.compatible_pairs == 0
    assert result.stats.pairs_failing_normal == 1


def test_similarity_graph_rejects_incompatible_color() -> None:
    proposals = _fusion_proposals(
        means=[[0.010, 0.0, 0.0], [0.011, 0.0, 0.0]],
        covariances=[np.eye(3) * 0.001, np.eye(3) * 0.001],
        colors=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    result = similarity_graph_fuse(proposals, config=FusionConfig(voxel_size=0.1))

    assert len(result.gaussians) == 2
    assert result.stats.compatible_pairs == 0
    assert result.stats.pairs_failing_color == 1


def test_similarity_graph_rejects_insufficient_overlap() -> None:
    proposals = _fusion_proposals(
        means=[[0.001, 0.0, 0.0], [0.091, 0.0, 0.0]],
        covariances=[np.eye(3) * 1.0e-6, np.eye(3) * 1.0e-6],
    )
    result = similarity_graph_fuse(proposals, config=FusionConfig(voxel_size=0.1))

    assert len(result.gaussians) == 2
    assert result.stats.compatible_pairs == 0
    assert result.stats.pairs_failing_overlap == 1


def test_similarity_graph_uses_connected_components() -> None:
    proposals = _fusion_proposals(
        means=[[0.001, 0.0, 0.0], [0.046, 0.0, 0.0], [0.091, 0.0, 0.0]],
        covariances=[np.eye(3) * 1.0e-6] * 3,
    )
    result = similarity_graph_fuse(proposals, config=FusionConfig(voxel_size=0.1))

    assert len(result.gaussians) == 1
    assert result.stats.candidate_pairs == 3
    assert result.stats.compatible_pairs == 2
    assert result.stats.components == 1


def _fusion_proposals(
    *,
    means: list[list[float]],
    covariances: list[np.ndarray],
    colors: list[list[float]] | None = None,
) -> GaussianProposals:
    pca_results = [
        decompose_covariance(np.asarray(covariance, dtype=np.float32), eigenvalue_epsilon=0.0)
        for covariance in covariances
    ]
    if colors is None:
        colors = [[0.5, 0.5, 0.5]] * len(means)
    return GaussianProposals.from_lists(
        means=[np.asarray(mean, dtype=np.float32) for mean in means],
        covariances=[result.covariance for result in pca_results],
        scales=[result.scales for result in pca_results],
        quats=[rotation_matrix_to_quaternion(result.basis) for result in pca_results],
        sh_dc=[rgb_to_sh_dc(np.asarray(color, dtype=np.float32)) for color in colors],
        opacities=[0.1] * len(means),
        view_ids=list(range(len(means))),
        scores=[1.0] * len(means),
    )
