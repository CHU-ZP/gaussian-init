from __future__ import annotations

import numpy as np


def covariance_to_scale_quat(
    covariance: np.ndarray,
    *,
    eigenvalue_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = estimate_local_pca_from_covariance(covariance, eigenvalue_epsilon=eigenvalue_epsilon)
    scales = np.sqrt(np.maximum(result[0], eigenvalue_epsilon)).astype(np.float32)
    quat = rotation_matrix_to_quaternion(result[1])
    return scales, quat


def estimate_local_pca_from_covariance(
    covariance: np.ndarray,
    *,
    eigenvalue_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    cov64 = np.asarray(covariance, dtype=np.float64)
    cov64 = 0.5 * (cov64 + cov64.T)
    cov64 += np.eye(3, dtype=np.float64) * float(eigenvalue_epsilon)
    eigenvalues, basis = np.linalg.eigh(cov64)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = basis[:, order]
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    return eigenvalues.astype(np.float32), basis.astype(np.float32)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3]")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s

    quat = np.asarray([qw, qx, qy, qz], dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def scale_quat_to_covariance(scales: np.ndarray, quat: np.ndarray) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(quat)
    diagonal = np.diag(np.asarray(scales, dtype=np.float64) ** 2)
    return (rotation @ diagonal @ rotation.T).astype(np.float32)


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1.0e-20)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
