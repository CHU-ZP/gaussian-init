from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class RenderConfig:
    near_plane: float = 0.01
    far_plane: float = 1.0e10
    radius_clip: float = 0.0
    eps2d: float = 0.3
    packed: bool = True
    antialiased: bool = False
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RenderConfig":
        values = values or {}
        background = tuple(float(value) for value in values.get("background", cls.background))
        if len(background) != 3 or any(value < 0.0 or value > 1.0 for value in background):
            raise ValueError("render.background must contain three values in [0, 1]")
        config = cls(
            near_plane=float(values.get("near_plane", cls.near_plane)),
            far_plane=float(values.get("far_plane", cls.far_plane)),
            radius_clip=float(values.get("radius_clip", cls.radius_clip)),
            eps2d=float(values.get("eps2d", cls.eps2d)),
            packed=bool(values.get("packed", cls.packed)),
            antialiased=bool(values.get("antialiased", cls.antialiased)),
            background=background,
        )
        if config.near_plane <= 0.0 or config.far_plane <= config.near_plane:
            raise ValueError("render near/far planes are invalid")
        if config.radius_clip < 0.0 or config.eps2d < 0.0:
            raise ValueError("render radius_clip and eps2d must be non-negative")
        return config


def rasterize_gaussians(
    params: Mapping[str, torch.Tensor],
    *,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int,
    config: RenderConfig,
    background: torch.Tensor | None = None,
    absgrad: bool = False,
    render_mode: str = "RGB",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise RuntimeError(
            "gsplat is required for rendering; install the project's 'train' extra"
        ) from exc

    if viewmats.ndim == 2:
        viewmats = viewmats.unsqueeze(0)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if viewmats.shape[-2:] != (4, 4) or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("viewmats and intrinsics must end in [4,4] and [3,3]")
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    if colors.shape[1] < (sh_degree + 1) ** 2:
        raise ValueError(
            f"SH degree {sh_degree} requires {(sh_degree + 1) ** 2} bases, "
            f"but only {colors.shape[1]} are available"
        )
    if render_mode != "RGB":
        raise ValueError("The training renderer currently supports render_mode='RGB' only")
    if background is None:
        background = torch.tensor(
            config.background,
            dtype=params["means"].dtype,
            device=params["means"].device,
        ).expand(viewmats.shape[0], 3)
    elif background.ndim == 1:
        background = background.unsqueeze(0).expand(viewmats.shape[0], 3)
    if tuple(background.shape) != (viewmats.shape[0], 3):
        raise ValueError(
            f"background must have shape {(viewmats.shape[0], 3)}, "
            f"got {tuple(background.shape)}"
        )

    rendered, alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=int(width),
        height=int(height),
        near_plane=config.near_plane,
        far_plane=config.far_plane,
        radius_clip=config.radius_clip,
        eps2d=config.eps2d,
        sh_degree=int(sh_degree),
        packed=config.packed,
        # gsplat 1.5.3's packed CUDA wrapper drops the camera dimension when
        # validating backgrounds. Composite explicitly below to keep packed
        # rendering and the documented per-camera background contract.
        backgrounds=None,
        render_mode=render_mode,
        absgrad=absgrad,
        rasterize_mode="antialiased" if config.antialiased else "classic",
    )
    rendered = rendered + background[:, None, None, :] * (1.0 - alphas)
    return rendered, alphas, info
