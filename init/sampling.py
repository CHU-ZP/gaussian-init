from __future__ import annotations

import numpy as np

from .types import PixelSamples


def valid_pixel_mask(
    confidence: np.ndarray,
    world_points: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    finite = np.isfinite(world_points).all(axis=-1)
    return finite & np.isfinite(confidence) & (confidence >= confidence_threshold)


def sample_pixels(
    *,
    view_id: int,
    image: np.ndarray,
    confidence: np.ndarray,
    world_points: np.ndarray,
    mode: str,
    stride: int,
    max_samples: int,
    confidence_threshold: float,
    salient_fraction: float,
    min_distance: int,
) -> PixelSamples:
    mask = valid_pixel_mask(confidence, world_points, confidence_threshold)
    mode_normalized = mode.lower()

    if mode_normalized == "uniform":
        us, vs, scores, kinds = uniform_samples(mask, confidence, stride, max_samples)
    elif mode_normalized in {"salient", "edge"}:
        us, vs, scores, kinds = salient_samples(
            image,
            mask,
            max_samples=max_samples,
            min_distance=min_distance,
            kind=mode_normalized,
        )
    elif mode_normalized == "hybrid":
        salient_budget = int(round(max_samples * np.clip(salient_fraction, 0.0, 1.0)))
        uniform_budget = max_samples - salient_budget
        s_us, s_vs, s_scores, s_kinds = salient_samples(
            image,
            mask,
            max_samples=salient_budget,
            min_distance=min_distance,
            kind="salient",
        )
        u_us, u_vs, u_scores, u_kinds = uniform_samples(mask, confidence, stride, uniform_budget)
        us, vs, scores, kinds = deduplicate_samples(
            [s_us, u_us],
            [s_vs, u_vs],
            [s_scores, u_scores],
            [s_kinds, u_kinds],
            max_samples=max_samples,
        )
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    return PixelSamples(
        view_ids=np.full((len(us),), view_id, dtype=np.int64),
        us=np.asarray(us, dtype=np.int64),
        vs=np.asarray(vs, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
        kinds=tuple(kinds),
    )


def uniform_samples(
    mask: np.ndarray,
    confidence: np.ndarray,
    stride: int,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if max_samples <= 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty.astype(np.float32), ()

    height, width = mask.shape
    step = max(int(stride), 1)
    y_start = min(step // 2, max(height - 1, 0))
    x_start = min(step // 2, max(width - 1, 0))
    ys, xs = np.meshgrid(
        np.arange(y_start, height, step, dtype=np.int64),
        np.arange(x_start, width, step, dtype=np.int64),
        indexing="ij",
    )
    us = xs.reshape(-1)
    vs = ys.reshape(-1)
    keep = mask[vs, us]
    us = us[keep]
    vs = vs[keep]
    scores = confidence[vs, us].astype(np.float32)

    if len(us) > max_samples:
        indices = np.linspace(0, len(us) - 1, max_samples, dtype=np.int64)
        us = us[indices]
        vs = vs[indices]
        scores = scores[indices]

    return us, vs, scores, tuple("uniform" for _ in range(len(us)))


def salient_samples(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    max_samples: int,
    min_distance: int,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if max_samples <= 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty.astype(np.float32), ()

    saliency = gradient_saliency(image)
    saliency = np.where(mask, saliency, 0.0)
    order = np.argsort(saliency.reshape(-1))[::-1]
    blocked = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    radius = max(int(min_distance), 0)

    selected_u: list[int] = []
    selected_v: list[int] = []
    selected_scores: list[float] = []

    for flat_index in order:
        if len(selected_u) >= max_samples:
            break
        score = float(saliency.reshape(-1)[flat_index])
        if score <= 0.0:
            break
        y, x = divmod(int(flat_index), width)
        if blocked[y, x] or not mask[y, x]:
            continue

        selected_u.append(x)
        selected_v.append(y)
        selected_scores.append(score)

        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        blocked[y0:y1, x0:x1] = True

    return (
        np.asarray(selected_u, dtype=np.int64),
        np.asarray(selected_v, dtype=np.int64),
        np.asarray(selected_scores, dtype=np.float32),
        tuple(kind for _ in selected_u),
    )


def gradient_saliency(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    dx = np.zeros_like(gray, dtype=np.float32)
    dy = np.zeros_like(gray, dtype=np.float32)
    dx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    dx[:, 0] = gray[:, 1] - gray[:, 0] if gray.shape[1] > 1 else 0.0
    dx[:, -1] = gray[:, -1] - gray[:, -2] if gray.shape[1] > 1 else 0.0
    dy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    dy[0, :] = gray[1, :] - gray[0, :] if gray.shape[0] > 1 else 0.0
    dy[-1, :] = gray[-1, :] - gray[-2, :] if gray.shape[0] > 1 else 0.0
    return np.sqrt(dx * dx + dy * dy)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[-1] == 1:
        return image[..., 0].astype(np.float32)
    return (
        0.299 * image[..., 0].astype(np.float32)
        + 0.587 * image[..., 1].astype(np.float32)
        + 0.114 * image[..., 2].astype(np.float32)
    )


def deduplicate_samples(
    us_list: list[np.ndarray],
    vs_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    kinds_list: list[tuple[str, ...]],
    *,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    best_by_pixel: dict[tuple[int, int], tuple[float, str]] = {}
    insertion_order: list[tuple[int, int]] = []

    for us, vs, scores, kinds in zip(us_list, vs_list, scores_list, kinds_list, strict=True):
        for u, v, score, kind in zip(us, vs, scores, kinds, strict=True):
            key = (int(u), int(v))
            if key not in best_by_pixel:
                insertion_order.append(key)
                best_by_pixel[key] = (float(score), kind)
            elif float(score) > best_by_pixel[key][0]:
                best_by_pixel[key] = (float(score), kind)

    selected = insertion_order[:max_samples]
    us = np.asarray([key[0] for key in selected], dtype=np.int64)
    vs = np.asarray([key[1] for key in selected], dtype=np.int64)
    scores = np.asarray([best_by_pixel[key][0] for key in selected], dtype=np.float32)
    kinds = tuple(best_by_pixel[key][1] for key in selected)
    return us, vs, scores, kinds
