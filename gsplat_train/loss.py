from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target)


def psnr(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return -10.0 * torch.log10(torch.clamp(mse, min=eps))
