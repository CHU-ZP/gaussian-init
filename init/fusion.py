from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from .filters import PCAFilterConfig, valid_pca
from .gaussian_params import rotation_matrix_to_quaternion, sh_dc_to_rgb
from .pca import decompose_covariance
from .sampling import rgb_to_normalized_lab
from .types import GaussianProposals


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = True
    voxel_size: float = 0.015
    overlap_mahalanobis_max: float = 9.0
    covariance_regularization_factor: float = 0.25
    normal_angle_max_degrees: float = 30.0
    scale_ratio_max: float = 3.0
    color_delta_e_max: float = 20.0

    def __post_init__(self) -> None:
        values = (
            self.voxel_size,
            self.overlap_mahalanobis_max,
            self.covariance_regularization_factor,
            self.normal_angle_max_degrees,
            self.scale_ratio_max,
            self.color_delta_e_max,
        )
        if not np.isfinite(values).all():
            raise ValueError("fusion parameters must be finite")
        if self.voxel_size <= 0.0:
            raise ValueError("fusion.voxel_size must be positive")
        if self.overlap_mahalanobis_max <= 0.0:
            raise ValueError("fusion.overlap_mahalanobis_max must be positive")
        if self.covariance_regularization_factor < 0.0:
            raise ValueError("fusion.covariance_regularization_factor must be non-negative")
        if not 0.0 <= self.normal_angle_max_degrees <= 90.0:
            raise ValueError("fusion.normal_angle_max_degrees must lie in [0, 90]")
        if self.scale_ratio_max < 1.0:
            raise ValueError("fusion.scale_ratio_max must be at least one")
        if self.color_delta_e_max < 0.0:
            raise ValueError("fusion.color_delta_e_max must be non-negative")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "FusionConfig":
        config = values or {}
        return cls(
            enabled=bool(config.get("enabled", cls.enabled)),
            voxel_size=float(config.get("voxel_size", cls.voxel_size)),
            overlap_mahalanobis_max=float(
                config.get("overlap_mahalanobis_max", cls.overlap_mahalanobis_max)
            ),
            covariance_regularization_factor=float(
                config.get(
                    "covariance_regularization_factor",
                    cls.covariance_regularization_factor,
                )
            ),
            normal_angle_max_degrees=float(
                config.get("normal_angle_max_degrees", cls.normal_angle_max_degrees)
            ),
            scale_ratio_max=float(config.get("scale_ratio_max", cls.scale_ratio_max)),
            color_delta_e_max=float(config.get("color_delta_e_max", cls.color_delta_e_max)),
        )


@dataclass(frozen=True)
class FusionStats:
    candidate_pairs: int
    compatible_pairs: int
    pairs_failing_overlap: int
    pairs_failing_normal: int
    pairs_failing_scale: int
    pairs_failing_color: int
    components: int
    singleton_components: int
    merged_components: int
    fallback_components: int
    output_gaussians: int


@dataclass(frozen=True)
class FusionResult:
    gaussians: GaussianProposals
    stats: FusionStats


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

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def similarity_graph_fuse(
    proposals: GaussianProposals,
    *,
    config: FusionConfig,
    eigenvalue_epsilon: float = 0.0,
    pca_filter: PCAFilterConfig | None = None,
) -> FusionResult:
    """Fuse compatible same-voxel proposals using a pair graph and union-find."""
    if len(proposals) == 0:
        return FusionResult(
            gaussians=proposals,
            stats=FusionStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

    count = len(proposals)
    means = np.asarray(proposals.means, dtype=np.float64)
    covariances = np.asarray(proposals.covariances, dtype=np.float64)
    scales = np.sort(np.asarray(proposals.scales, dtype=np.float64), axis=1)
    _, eigenvectors = np.linalg.eigh(covariances)
    normals = eigenvectors[:, :, 0]
    rgb = np.clip(sh_dc_to_rgb(proposals.sh_dc), 0.0, 1.0)
    normalized_lab = rgb_to_normalized_lab(rgb[:, None, :])[:, 0]
    lab = normalized_lab.astype(np.float64) * np.asarray([100.0, 128.0, 128.0])

    voxel_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    voxel_indices = np.floor(means / config.voxel_size).astype(np.int64)
    for index, voxel in enumerate(voxel_indices):
        voxel_groups[tuple(int(value) for value in voxel)].append(index)

    union_find = _UnionFind(count)
    candidate_pairs = 0
    compatible_pairs = 0
    failing_overlap = 0
    failing_normal = 0
    failing_scale = 0
    failing_color = 0
    normal_cosine_min = float(np.cos(np.deg2rad(config.normal_angle_max_degrees)))
    covariance_floor = (config.covariance_regularization_factor * config.voxel_size) ** 2
    identity = np.eye(3, dtype=np.float64)

    for indices in voxel_groups.values():
        for left, right in combinations(indices, 2):
            candidate_pairs += 1
            normal_compatible = (
                abs(float(np.dot(normals[left], normals[right]))) >= normal_cosine_min
            )
            scale_ratios = np.maximum(
                scales[left] / np.maximum(scales[right], 1.0e-20),
                scales[right] / np.maximum(scales[left], 1.0e-20),
            )
            scale_compatible = float(np.max(scale_ratios)) <= config.scale_ratio_max
            color_compatible = (
                float(np.linalg.norm(lab[left] - lab[right])) <= config.color_delta_e_max
            )

            delta = means[left] - means[right]
            combined_covariance = (
                0.5 * (covariances[left] + covariances[right]) + covariance_floor * identity
            )
            try:
                solved_delta = np.linalg.solve(combined_covariance, delta)
            except np.linalg.LinAlgError:
                solved_delta = np.linalg.pinv(combined_covariance) @ delta
            mahalanobis_squared = float(delta @ solved_delta)
            overlap_compatible = mahalanobis_squared <= config.overlap_mahalanobis_max

            failing_normal += int(not normal_compatible)
            failing_scale += int(not scale_compatible)
            failing_color += int(not color_compatible)
            failing_overlap += int(not overlap_compatible)
            if not (
                normal_compatible and scale_compatible and color_compatible and overlap_compatible
            ):
                continue
            compatible_pairs += 1
            union_find.union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(union_find.find(index), []).append(index)

    output = _ProposalBuilder()
    singleton_components = 0
    merged_components = 0
    fallback_components = 0
    for indices in components.values():
        if len(indices) == 1:
            singleton_components += 1
            output.append_original(proposals, indices[0])
            continue
        fused = _fuse_component(
            proposals,
            indices,
            eigenvalue_epsilon=eigenvalue_epsilon,
            pca_filter=pca_filter,
        )
        if fused is None:
            fallback_components += 1
            for index in indices:
                output.append_original(proposals, index)
            continue
        merged_components += 1
        output.append_fused(*fused)

    gaussians = output.build()
    return FusionResult(
        gaussians=gaussians,
        stats=FusionStats(
            candidate_pairs=candidate_pairs,
            compatible_pairs=compatible_pairs,
            pairs_failing_overlap=failing_overlap,
            pairs_failing_normal=failing_normal,
            pairs_failing_scale=failing_scale,
            pairs_failing_color=failing_color,
            components=len(components),
            singleton_components=singleton_components,
            merged_components=merged_components,
            fallback_components=fallback_components,
            output_gaussians=len(gaussians),
        ),
    )


def _fuse_component(
    proposals: GaussianProposals,
    indices: list[int],
    *,
    eigenvalue_epsilon: float,
    pca_filter: PCAFilterConfig | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float] | None:
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
        return None
    return (
        mean.astype(np.float32),
        pca_result.covariance,
        pca_result.scales,
        rotation_matrix_to_quaternion(pca_result.basis),
        np.sum(proposals.sh_dc[idx] * weights[:, None], axis=0).astype(np.float32),
        float(np.sum(proposals.opacities[idx] * weights)),
        float(np.sum(proposals.confidences[idx] * weights)),
        float(np.sum(proposals.scores[idx] * weights)),
    )


class _ProposalBuilder:
    def __init__(self) -> None:
        self.means: list[np.ndarray] = []
        self.covariances: list[np.ndarray] = []
        self.scales: list[np.ndarray] = []
        self.quats: list[np.ndarray] = []
        self.sh_dc: list[np.ndarray] = []
        self.opacities: list[float] = []
        self.confidences: list[float] = []
        self.view_ids: list[int] = []
        self.scores: list[float] = []

    def append_original(self, proposals: GaussianProposals, index: int) -> None:
        self.means.append(proposals.means[index])
        self.covariances.append(proposals.covariances[index])
        self.scales.append(proposals.scales[index])
        self.quats.append(proposals.quats[index])
        self.sh_dc.append(proposals.sh_dc[index])
        self.opacities.append(float(proposals.opacities[index]))
        self.confidences.append(float(proposals.confidences[index]))
        self.view_ids.append(int(proposals.view_ids[index]))
        self.scores.append(float(proposals.scores[index]))

    def append_fused(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        scales: np.ndarray,
        quat: np.ndarray,
        sh_dc: np.ndarray,
        opacity: float,
        confidence: float,
        score: float,
    ) -> None:
        self.means.append(mean)
        self.covariances.append(covariance)
        self.scales.append(scales)
        self.quats.append(quat)
        self.sh_dc.append(sh_dc)
        self.opacities.append(opacity)
        self.confidences.append(confidence)
        self.view_ids.append(-1)
        self.scores.append(score)

    def build(self) -> GaussianProposals:
        return GaussianProposals.from_lists(
            means=self.means,
            covariances=self.covariances,
            scales=self.scales,
            quats=self.quats,
            sh_dc=self.sh_dc,
            opacities=self.opacities,
            confidences=self.confidences,
            view_ids=self.view_ids,
            scores=self.scores,
        )
