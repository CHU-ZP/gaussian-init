from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pca import PCAResult


@dataclass(frozen=True)
class PCAFilterConfig:
    scale_min: float
    scale_max: float
    min_secondary_eigenvalue_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.scale_min,
            self.scale_max,
            self.min_secondary_eigenvalue_ratio,
        )
        if not np.isfinite(values).all():
            raise ValueError("PCA filter bounds must be finite")
        if self.scale_min <= 0.0 or self.scale_max < self.scale_min:
            raise ValueError("PCA scale bounds must satisfy 0 < scale_min <= scale_max")
        if not 0.0 < self.min_secondary_eigenvalue_ratio <= 1.0:
            raise ValueError("PCA min_secondary_eigenvalue_ratio must lie in (0, 1]")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PCAFilterConfig":
        return cls(
            scale_min=float(config.get("scale_min", 1.0e-5)),
            scale_max=float(config.get("scale_max", 1.0)),
            min_secondary_eigenvalue_ratio=float(
                config.get("min_secondary_eigenvalue_ratio", 0.01)
            ),
        )


def valid_pca(result: PCAResult, config: PCAFilterConfig) -> bool:
    if not np.isfinite(result.covariance).all():
        return False
    if not np.isfinite(result.eigenvalues).all():
        return False
    if np.any(result.eigenvalues <= 0.0):
        return False

    scales = result.scales
    if not np.isfinite(scales).all():
        return False
    # A surface patch is expected to have two supported tangent directions and
    # may legitimately be arbitrarily thin along its normal. Therefore the
    # lower scale bound applies to the secondary tangent scale, not the normal
    # scale. The eigenvalue ratio rejects only rank-one/line-like fits.
    if scales[1] < config.scale_min:
        return False
    if scales[0] > config.scale_max:
        return False
    secondary_ratio = float(result.eigenvalues[1]) / float(result.eigenvalues[0])
    return secondary_ratio >= config.min_secondary_eigenvalue_ratio
