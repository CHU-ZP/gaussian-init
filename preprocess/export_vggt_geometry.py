from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from init.io import load_vggt_predictions, write_ply


def export_point_cloud(
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    stride: int,
    max_points: int | None,
) -> None:
    predictions = load_vggt_predictions(predictions_path)
    points = predictions["world_points"][:, ::stride, ::stride, :].reshape(-1, 3)
    confidence = predictions["confidence"][:, ::stride, ::stride].reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    points = points[valid]
    confidence = confidence[valid]

    if max_points is not None and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
        confidence = confidence[indices]

    colors = np.repeat(normalize_confidence(confidence)[:, None], 3, axis=1)
    write_ply(output_path, points, colors)
    print(f"Exported {points.shape[0]} points to {output_path}")


def normalize_confidence(confidence: np.ndarray) -> np.ndarray:
    """Map unbounded VGGT confidence scores to a robust grayscale range."""
    values = np.asarray(confidence, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("confidence must be a non-empty finite vector")
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        return np.ones_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VGGT dense geometry to PLY.")
    parser.add_argument("--predictions", required=True, help="Input predictions.npz path.")
    parser.add_argument("--output", required=True, help="Output PLY path.")
    parser.add_argument("--stride", type=int, default=8, help="Subsample stride.")
    parser.add_argument("--max-points", type=int, default=500000, help="Maximum exported points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_point_cloud(
        args.predictions,
        args.output,
        stride=max(args.stride, 1),
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
