from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pca import PCAResult


@dataclass(frozen=True)
class PCAFilterConfig:
    scale_min: float
    scale_max: float
    condition_max: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PCAFilterConfig":
        return cls(
            scale_min=float(config.get("scale_min", 1.0e-5)),
            scale_max=float(config.get("scale_max", 1.0)),
            condition_max=float(config.get("condition_max", 5000.0)),
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
    if np.any(scales < config.scale_min):
        return False
    if np.any(scales > config.scale_max):
        return False
    return result.condition_number <= config.condition_max
