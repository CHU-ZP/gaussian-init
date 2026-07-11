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


def decompose_covariance(
    covariance: np.ndarray,
    *,
    eigenvalue_epsilon: float,
) -> PCAResult:
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (3, 3):
        raise ValueError("covariance must have shape [3, 3]")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must be finite")
    if not np.isfinite(eigenvalue_epsilon) or eigenvalue_epsilon < 0.0:
        raise ValueError("eigenvalue_epsilon must be finite and non-negative")
    covariance = 0.5 * (covariance + covariance.T)
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
