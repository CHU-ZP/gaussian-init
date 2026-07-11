from __future__ import annotations

from collections import defaultdict

import numpy as np

from .filters import PCAFilterConfig, valid_pca
from .gaussian_params import rotation_matrix_to_quaternion
from .pca import decompose_covariance
from .types import GaussianProposals


def voxel_fuse(
    proposals: GaussianProposals,
    *,
    voxel_size: float,
    eigenvalue_epsilon: float = 0.0,
    pca_filter: PCAFilterConfig | None = None,
) -> GaussianProposals:
    if len(proposals) == 0:
        return proposals
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")

    clusters: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    voxel_indices = np.floor(proposals.means / float(voxel_size)).astype(np.int64)
    for index, voxel in enumerate(voxel_indices):
        clusters[tuple(int(value) for value in voxel)].append(index)

    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    quats: list[np.ndarray] = []
    sh_dc: list[np.ndarray] = []
    opacities: list[float] = []
    confidences: list[float] = []
    view_ids: list[int] = []
    scores: list[float] = []

    for indices in clusters.values():
        idx = np.asarray(indices, dtype=np.int64)
        weights = np.maximum(proposals.confidences[idx], 1.0e-6).astype(np.float64)
        weights /= np.sum(weights)

        cluster_means = proposals.means[idx].astype(np.float64)
        mean = np.sum(cluster_means * weights[:, None], axis=0)
        deltas = cluster_means - mean[None, :]
        second_moments = proposals.covariances[idx].astype(np.float64) + (
            deltas[:, :, None] @ deltas[:, None, :]
        )
        covariance = np.sum(second_moments * weights[:, None, None], axis=0)
        covariance = 0.5 * (covariance + covariance.T)

        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance)[0])
        regularization = max(float(eigenvalue_epsilon) - minimum_eigenvalue, 0.0)
        pca_result = decompose_covariance(
            covariance.astype(np.float32),
            eigenvalue_epsilon=regularization,
        )
        if pca_filter is not None and not valid_pca(pca_result, pca_filter):
            continue

        means.append(mean.astype(np.float32))
        covariances.append(pca_result.covariance)
        scales.append(pca_result.scales)
        quats.append(rotation_matrix_to_quaternion(pca_result.basis))
        sh_dc.append(np.sum(proposals.sh_dc[idx] * weights[:, None], axis=0).astype(np.float32))
        opacities.append(float(np.sum(proposals.opacities[idx] * weights)))
        confidences.append(float(np.sum(proposals.confidences[idx] * weights)))
        view_ids.append(-1)
        scores.append(float(np.sum(proposals.scores[idx] * weights)))

    return GaussianProposals.from_lists(
        means=means,
        covariances=covariances,
        scales=scales,
        quats=quats,
        sh_dc=sh_dc,
        opacities=opacities,
        confidences=confidences,
        view_ids=view_ids,
        scores=scores,
    )
