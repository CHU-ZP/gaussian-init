from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


INITIALIZATION_KEYS = ("means", "scales", "quats", "opacities", "sh_dc")
RAW_PARAMETER_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def load_torch_state(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GaussianModel(nn.Module):
    """Trainable gsplat parameters with explicit raw/activated conventions.

    ``params["scales"]`` stores log-scales and ``params["opacities"]`` stores
    logits.  This naming is required by gsplat's densification strategies.
    """

    def __init__(self, state: Mapping[str, Any], *, sh_degree: int = 3, raw: bool = False) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        self.max_sh_degree = int(sh_degree)
        tensors = (
            self._parameters_from_raw_state(state)
            if raw
            else self._parameters_from_initialization(state, sh_degree=self.max_sh_degree)
        )
        self.params = nn.ParameterDict(
            {name: nn.Parameter(value.contiguous()) for name, value in tensors.items()}
        )
        self.validate()

    @classmethod
    def from_file(cls, path: str | Path, *, sh_degree: int = 3) -> "GaussianModel":
        state = load_torch_state(path)
        if "splats" in state:
            degree = int(state.get("max_sh_degree", sh_degree))
            return cls(state["splats"], sh_degree=degree, raw=True)
        return cls(state, sh_degree=sh_degree)

    @classmethod
    def from_raw_state(
        cls, state: Mapping[str, Any], *, sh_degree: int | None = None
    ) -> "GaussianModel":
        inferred_degree = infer_sh_degree(int(state["sh0"].shape[1] + state["shN"].shape[1]))
        degree = inferred_degree if sh_degree is None else int(sh_degree)
        return cls(state, sh_degree=degree, raw=True)

    @property
    def means(self) -> torch.Tensor:
        return self.params["means"]

    @property
    def log_scales(self) -> torch.Tensor:
        return self.params["scales"]

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.params["scales"])

    @property
    def quats(self) -> torch.Tensor:
        return self.params["quats"]

    @property
    def normalized_quats(self) -> torch.Tensor:
        return normalize_quaternions(self.params["quats"])

    @property
    def opacity_logits(self) -> torch.Tensor:
        return self.params["opacities"]

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.params["opacities"])

    @property
    def sh_dc(self) -> torch.Tensor:
        return self.params["sh0"][:, 0, :]

    @property
    def sh_coefficients(self) -> torch.Tensor:
        return torch.cat([self.params["sh0"], self.params["shN"]], dim=1)

    def activated_state(self, *, include_covariances: bool = True) -> dict[str, torch.Tensor]:
        state = {
            "means": self.means.detach(),
            "scales": self.scales.detach(),
            "quats": self.normalized_quats.detach(),
            "opacities": self.opacities.detach(),
            "sh_dc": self.sh_dc.detach(),
            "sh_rest": self.params["shN"].detach(),
        }
        if include_covariances:
            state["covariances"] = scale_quat_to_covariance_torch(
                state["scales"], state["quats"]
            )
        return state

    def raw_state_cpu(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.params.items()}

    def validate(self) -> None:
        missing = sorted(set(RAW_PARAMETER_KEYS).difference(self.params.keys()))
        if missing:
            raise ValueError(f"Raw Gaussian parameters are missing keys: {missing}")
        count = int(self.params["means"].shape[0])
        expected = {
            "means": (count, 3),
            "scales": (count, 3),
            "quats": (count, 4),
            "opacities": (count,),
            "sh0": (count, 1, 3),
            "shN": (count, (self.max_sh_degree + 1) ** 2 - 1, 3),
        }
        for name, shape in expected.items():
            value = self.params[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")

    @staticmethod
    def _parameters_from_initialization(
        state: Mapping[str, Any], *, sh_degree: int
    ) -> dict[str, torch.Tensor]:
        missing = sorted(set(INITIALIZATION_KEYS).difference(state))
        if missing:
            raise ValueError(f"Gaussian initialization is missing keys: {missing}")
        means = as_float_tensor(state["means"])
        count = int(means.shape[0])
        sh_dc = as_float_tensor(state["sh_dc"])
        sh_rest_count = (sh_degree + 1) ** 2 - 1
        if "sh_rest" in state:
            sh_rest = as_float_tensor(state["sh_rest"])
            if sh_rest.ndim == 2 and sh_rest_count == 0:
                sh_rest = sh_rest.reshape(count, 0, 3)
        else:
            sh_rest = torch.zeros((count, sh_rest_count, 3), dtype=torch.float32)
        return {
            "means": means,
            "scales": torch.log(torch.clamp(as_float_tensor(state["scales"]), min=1.0e-8)),
            "quats": normalize_quaternions(as_float_tensor(state["quats"])),
            "opacities": logit(
                torch.clamp(as_float_tensor(state["opacities"]), 1.0e-6, 1.0 - 1.0e-6)
            ),
            "sh0": sh_dc[:, None, :],
            "shN": sh_rest,
        }

    @staticmethod
    def _parameters_from_raw_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        missing = sorted(set(RAW_PARAMETER_KEYS).difference(state))
        if missing:
            raise ValueError(f"Raw Gaussian state is missing keys: {missing}")
        return {name: as_float_tensor(state[name]) for name in RAW_PARAMETER_KEYS}


def as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    return torch.as_tensor(value, dtype=torch.float32)


def infer_sh_degree(basis_count: int) -> int:
    degree = int(round(basis_count**0.5)) - 1
    if degree < 0 or (degree + 1) ** 2 != basis_count:
        raise ValueError(f"Invalid spherical-harmonic basis count: {basis_count}")
    return degree


def normalize_quaternions(quats: torch.Tensor) -> torch.Tensor:
    return quats / torch.clamp(torch.linalg.norm(quats, dim=-1, keepdim=True), min=1.0e-8)


def quaternion_to_rotation_matrix_torch(quats: torch.Tensor) -> torch.Tensor:
    quats = normalize_quaternions(quats)
    w, x, y, z = quats.unbind(dim=-1)
    return torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quats.shape[:-1], 3, 3)


def scale_quat_to_covariance_torch(
    scales: torch.Tensor, quats: torch.Tensor
) -> torch.Tensor:
    rotation = quaternion_to_rotation_matrix_torch(quats)
    scaled_rotation = rotation * scales.unsqueeze(-2)
    return scaled_rotation @ scaled_rotation.transpose(-1, -2)


def logit(values: torch.Tensor) -> torch.Tensor:
    return torch.log(values / (1.0 - values))
