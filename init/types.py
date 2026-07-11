from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EllipseKeypoints:
    view_ids: np.ndarray
    us: np.ndarray
    vs: np.ndarray
    scores: np.ndarray
    sigmas: np.ndarray
    levels: np.ndarray
    ellipse_matrices: np.ndarray
    ellipse_areas: np.ndarray

    def __len__(self) -> int:
        return int(self.us.shape[0])


@dataclass(frozen=True)
class GaussianProposals:
    means: np.ndarray
    covariances: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    sh_dc: np.ndarray
    opacities: np.ndarray
    confidences: np.ndarray
    view_ids: np.ndarray
    scores: np.ndarray

    @classmethod
    def empty(cls) -> "GaussianProposals":
        return cls(
            means=np.empty((0, 3), dtype=np.float32),
            covariances=np.empty((0, 3, 3), dtype=np.float32),
            scales=np.empty((0, 3), dtype=np.float32),
            quats=np.empty((0, 4), dtype=np.float32),
            sh_dc=np.empty((0, 3), dtype=np.float32),
            opacities=np.empty((0,), dtype=np.float32),
            confidences=np.empty((0,), dtype=np.float32),
            view_ids=np.empty((0,), dtype=np.int64),
            scores=np.empty((0,), dtype=np.float32),
        )

    @classmethod
    def from_lists(
        cls,
        means: list[np.ndarray],
        covariances: list[np.ndarray],
        scales: list[np.ndarray],
        quats: list[np.ndarray],
        sh_dc: list[np.ndarray],
        opacities: list[float],
        confidences: list[float],
        view_ids: list[int],
        scores: list[float],
    ) -> "GaussianProposals":
        if not means:
            return cls.empty()
        return cls(
            means=np.asarray(means, dtype=np.float32),
            covariances=np.asarray(covariances, dtype=np.float32),
            scales=np.asarray(scales, dtype=np.float32),
            quats=np.asarray(quats, dtype=np.float32),
            sh_dc=np.asarray(sh_dc, dtype=np.float32),
            opacities=np.asarray(opacities, dtype=np.float32),
            confidences=np.asarray(confidences, dtype=np.float32),
            view_ids=np.asarray(view_ids, dtype=np.int64),
            scores=np.asarray(scores, dtype=np.float32),
        )

    @classmethod
    def concat(cls, proposals: Iterable["GaussianProposals"]) -> "GaussianProposals":
        items = [item for item in proposals if len(item) > 0]
        if not items:
            return cls.empty()
        return cls(
            means=np.concatenate([item.means for item in items], axis=0),
            covariances=np.concatenate([item.covariances for item in items], axis=0),
            scales=np.concatenate([item.scales for item in items], axis=0),
            quats=np.concatenate([item.quats for item in items], axis=0),
            sh_dc=np.concatenate([item.sh_dc for item in items], axis=0),
            opacities=np.concatenate([item.opacities for item in items], axis=0),
            confidences=np.concatenate([item.confidences for item in items], axis=0),
            view_ids=np.concatenate([item.view_ids for item in items], axis=0),
            scores=np.concatenate([item.scores for item in items], axis=0),
        )

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def to_torch_dict(self) -> dict:
        import torch

        return {
            "means": torch.from_numpy(np.asarray(self.means, dtype=np.float32)),
            "scales": torch.from_numpy(np.asarray(self.scales, dtype=np.float32)),
            "quats": torch.from_numpy(np.asarray(self.quats, dtype=np.float32)),
            "opacities": torch.from_numpy(np.asarray(self.opacities, dtype=np.float32)),
            "sh_dc": torch.from_numpy(np.asarray(self.sh_dc, dtype=np.float32)),
            "covariances": torch.from_numpy(np.asarray(self.covariances, dtype=np.float32)),
            "confidences": torch.from_numpy(np.asarray(self.confidences, dtype=np.float32)),
            "view_ids": torch.from_numpy(np.asarray(self.view_ids, dtype=np.int64)),
            "scores": torch.from_numpy(np.asarray(self.scores, dtype=np.float32)),
        }
