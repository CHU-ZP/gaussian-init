from __future__ import annotations

import numpy as np
import torch


def compute_knn_isotropic_scales(
    points: np.ndarray,
    *,
    neighbors: int = 3,
    device: str = "auto",
    chunk_size: int = 512,
    minimum_scale: float = 0.0,
) -> np.ndarray:
    """Compute the RMS distance to each point's exact k nearest neighbors."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if neighbors <= 0 or points.shape[0] <= neighbors:
        raise ValueError("points must contain more entries than the requested neighbors")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not np.isfinite(minimum_scale) or minimum_scale < 0.0:
        raise ValueError("minimum_scale must be finite and non-negative")
    if not np.isfinite(points).all():
        raise ValueError("points contain non-finite values")

    resolved_device = resolve_torch_device(device)
    values = torch.from_numpy(points).to(device=resolved_device, dtype=torch.float32)
    # Centering is distance-invariant and improves precision when world
    # coordinates share a large common translation.
    values = values - values.mean(dim=0, keepdim=True)
    scales = torch.empty(points.shape[0], device=resolved_device, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, points.shape[0], chunk_size):
            stop = min(start + chunk_size, points.shape[0])
            distances = torch.cdist(
                values[start:stop],
                values,
                p=2.0,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            rows = torch.arange(stop - start, device=resolved_device)
            columns = torch.arange(start, stop, device=resolved_device)
            distances[rows, columns] = torch.inf
            nearest = torch.topk(
                distances,
                k=neighbors,
                dim=1,
                largest=False,
                sorted=False,
            ).values
            scales[start:stop] = torch.sqrt(torch.mean(nearest.square(), dim=1))

    if minimum_scale > 0.0:
        scales.clamp_min_(float(minimum_scale))
    result = scales.cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise RuntimeError("Nearest-neighbor scale estimation produced non-positive values")
    return result


def resolve_torch_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for kNN scale estimation but is unavailable")
    return device
