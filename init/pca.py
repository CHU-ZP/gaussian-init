from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAResult:
    covariance: np.ndarray
    eigenvalues: np.ndarray
    basis: np.ndarray

    @property
    def scales(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.eigenvalues, 0.0)).astype(np.float32)

    @property
    def condition_number(self) -> float:
        smallest = max(float(self.eigenvalues[-1]), 1.0e-20)
        return float(self.eigenvalues[0]) / smallest


def estimate_local_pca(points: np.ndarray, *, eigenvalue_epsilon: float) -> PCAResult:
    points64 = np.asarray(points, dtype=np.float64)
    if points64.ndim != 2 or points64.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if points64.shape[0] < 2:
        raise ValueError("PCA needs at least two points")

    centered = points64 - np.mean(points64, axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(points64.shape[0] - 1, 1)
    covariance += np.eye(3, dtype=np.float64) * float(eigenvalue_epsilon)

    eigenvalues, basis = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = basis[:, order]

    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0

    return PCAResult(
        covariance=covariance.astype(np.float32),
        eigenvalues=eigenvalues.astype(np.float32),
        basis=basis.astype(np.float32),
    )
