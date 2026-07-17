from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    difference = torch.abs(prediction - target)
    if mask is None:
        return difference.mean()
    mask = _image_mask(mask, prediction)
    denominator = torch.clamp(mask.sum() * prediction.shape[-1], min=1.0)
    return (difference * mask).sum() / denominator


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    prediction = _as_nchw(prediction)
    target = _as_nchw(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if mask is not None:
        image_mask = _image_mask(mask, prediction.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        prediction = prediction * image_mask
        target = target * image_mask
    size = min(window_size, prediction.shape[-2], prediction.shape[-1])
    if size % 2 == 0:
        size -= 1
    if size < 1:
        raise ValueError("SSIM input images cannot be empty")
    coords = torch.arange(size, dtype=prediction.dtype, device=prediction.device)
    coords -= (size - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).expand(
        prediction.shape[1], 1, size, size
    )
    padding = size // 2
    means_prediction = F.conv2d(
        prediction, kernel_2d, padding=padding, groups=prediction.shape[1]
    )
    means_target = F.conv2d(target, kernel_2d, padding=padding, groups=target.shape[1])
    prediction_sq = means_prediction.square()
    target_sq = means_target.square()
    product = means_prediction * means_target
    variance_prediction = F.conv2d(
        prediction.square(), kernel_2d, padding=padding, groups=prediction.shape[1]
    ) - prediction_sq
    variance_target = F.conv2d(
        target.square(), kernel_2d, padding=padding, groups=target.shape[1]
    ) - target_sq
    covariance = F.conv2d(
        prediction * target, kernel_2d, padding=padding, groups=prediction.shape[1]
    ) - product
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2.0 * product + c1) * (2.0 * covariance + c2)) / (
        (prediction_sq + target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return score.mean()


def photometric_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    ssim_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0.0 <= ssim_weight <= 1.0:
        raise ValueError("ssim_weight must be in [0, 1]")
    l1 = masked_l1_loss(prediction, target, mask)
    ssim_value = ssim(prediction, target, mask=mask)
    ssim_loss = 1.0 - ssim_value
    total = (1.0 - ssim_weight) * l1 + ssim_weight * ssim_loss
    return total, {"l1": l1, "ssim": ssim_value}


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    squared_error = (prediction - target).square()
    if mask is None:
        mse = squared_error.mean()
    else:
        image_mask = _image_mask(mask, prediction)
        denominator = torch.clamp(image_mask.sum() * prediction.shape[-1], min=1.0)
        mse = (squared_error * image_mask).sum() / denominator
    return -10.0 * torch.log10(torch.clamp(mse, min=eps))


def _as_nchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("Images must have shape [H,W,3] or [B,H,W,3]")
    return image.permute(0, 3, 1, 2)


def _image_mask(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)
    expected = (image.shape[0], image.shape[1], image.shape[2], 1)
    if tuple(mask.shape) != expected:
        raise ValueError(f"mask must have shape {expected}, got {tuple(mask.shape)}")
    return mask.to(dtype=image.dtype, device=image.device)
