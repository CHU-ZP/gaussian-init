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


def floor_normal_scale(result: PCAResult, *, minimum_scale: float) -> PCAResult:
    """Give a planar PCA fit finite normal thickness without changing its tangents."""
    if not np.isfinite(minimum_scale) or minimum_scale < 0.0:
        raise ValueError("minimum_scale must be finite and non-negative")

    scales = result.scales.astype(np.float64)
    if not np.isfinite(scales).all():
        raise ValueError("PCA scales must be finite")
    # The normal must not become wider than the weaker tangent direction. This
    # cap matters only for pathological inputs or a badly oversized reference
    # scale; ordinary grid surfaces receive exactly the requested floor.
    normal_scale = min(max(float(scales[-1]), minimum_scale), float(scales[-2]))
    if normal_scale <= float(scales[-1]):
        return result

    eigenvalues = result.eigenvalues.astype(np.float64).copy()
    eigenvalues[-1] = normal_scale**2
    basis = result.basis.astype(np.float64)
    covariance = basis @ np.diag(eigenvalues) @ basis.T
    covariance = 0.5 * (covariance + covariance.T)
    return PCAResult(
        covariance=covariance.astype(np.float32),
        eigenvalues=eigenvalues.astype(np.float32),
        basis=result.basis.copy(),
    )
