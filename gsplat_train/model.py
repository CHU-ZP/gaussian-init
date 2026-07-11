from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_gaussian_state(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GaussianModel(nn.Module):
    def __init__(self, state: dict) -> None:
        super().__init__()
        required = {"means", "scales", "quats", "opacities", "colors"}
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Gaussian state is missing keys: {missing}")

        self.means = nn.Parameter(state["means"].float())
        self.log_scales = nn.Parameter(torch.log(torch.clamp(state["scales"].float(), min=1.0e-8)))
        self.quats = nn.Parameter(normalize_quaternions(state["quats"].float()))
        self.opacity_logits = nn.Parameter(logit(torch.clamp(state["opacities"].float(), 1.0e-6, 1.0 - 1.0e-6)))
        self.colors = nn.Parameter(torch.clamp(state["colors"].float(), 0.0, 1.0))

    @classmethod
    def from_file(cls, path: str | Path) -> "GaussianModel":
        return cls(load_gaussian_state(path))

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def normalized_quats(self) -> torch.Tensor:
        return normalize_quaternions(self.quats)


def normalize_quaternions(quats: torch.Tensor) -> torch.Tensor:
    return quats / torch.clamp(torch.linalg.norm(quats, dim=-1, keepdim=True), min=1.0e-8)


def logit(values: torch.Tensor) -> torch.Tensor:
    return torch.log(values / (1.0 - values))
