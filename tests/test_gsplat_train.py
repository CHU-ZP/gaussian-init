from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from gsplat_train.dataset import split_view_indices, validate_world_point_projection
from gsplat_train.loss import photometric_loss, psnr
from gsplat_train.model import GaussianModel
from gsplat_train.render import RenderConfig, rasterize_gaussians
from gsplat_train.train import apply_gsplat_153_opacity_reset, create_optimizers


def make_initialization(count: int = 2) -> dict[str, torch.Tensor]:
    return {
        "means": torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]])[:count],
        "scales": torch.full((count, 3), 0.1),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        "opacities": torch.full((count,), 0.1),
        "sh_dc": torch.zeros((count, 3)),
    }


def test_gaussian_model_maps_activated_initialization_to_raw_gsplat_parameters() -> None:
    initialization = make_initialization()
    model = GaussianModel(initialization, sh_degree=3)

    assert set(model.params) == {"means", "scales", "quats", "opacities", "sh0", "shN"}
    assert model.params["sh0"].shape == (2, 1, 3)
    assert model.params["shN"].shape == (2, 15, 3)
    assert torch.allclose(model.scales, initialization["scales"])
    assert torch.allclose(model.opacities, initialization["opacities"])
    assert torch.allclose(model.sh_dc, initialization["sh_dc"])
    exported = model.activated_state()
    assert exported["covariances"].shape == (2, 3, 3)
    assert torch.allclose(exported["covariances"], torch.eye(3)[None] * 0.01)


def test_loss_is_zero_and_psnr_is_bounded_for_identical_images() -> None:
    image = torch.rand(1, 16, 16, 3)
    mask = torch.ones(1, 16, 16, dtype=torch.bool)
    loss, components = photometric_loss(image, image, mask=mask)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1.0e-6)
    assert torch.allclose(components["ssim"], torch.tensor(1.0), atol=1.0e-6)
    assert float(psnr(image, image, mask=mask)) == pytest.approx(80.0)


def test_camera_projection_validation_and_deterministic_split() -> None:
    height, width = 8, 10
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([xx / 100.0, yy / 100.0, np.ones_like(xx)], axis=-1)[None].astype(
        np.float32
    )
    intrinsics = np.asarray([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]])
    error = validate_world_point_projection(
        points,
        np.ones((1, height, width), dtype=bool),
        intrinsics.astype(np.float32),
        np.eye(4, dtype=np.float32)[None],
        max_error_px=0.01,
        sample_stride=1,
    )
    train, test = split_view_indices(10, test_every=4)

    assert error < 1.0e-6
    assert test.tolist() == [0, 4, 8]
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(10))


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("gsplat") is None,
    reason="gsplat CUDA test requires the train environment",
)
def test_renderer_and_default_strategy_can_resize_initialized_parameters(tmp_path: Path) -> None:
    from gsplat import DefaultStrategy

    model = GaussianModel(make_initialization(count=1), sh_degree=0).cuda()
    optimizers = create_optimizers(model, {}, scene_scale=1.0)
    strategy = DefaultStrategy(
        grow_grad2d=0.0,
        grow_scale3d=10.0,
        prune_scale3d=100.0,
        refine_start_iter=-1,
        refine_stop_iter=10,
        refine_every=1,
        reset_every=1000,
    )
    strategy.check_sanity(model.params, optimizers)
    state = strategy.initialize_state(scene_scale=1.0)
    rendered, _, info = rasterize_gaussians(
        model.params,
        viewmats=torch.eye(4, device="cuda"),
        intrinsics=torch.tensor(
            [[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
            device="cuda",
        ),
        width=32,
        height=32,
        sh_degree=0,
        config=RenderConfig(packed=True),
    )
    strategy.step_pre_backward(model.params, optimizers, state, 0, info)
    horizontal_weight = torch.linspace(0.0, 1.0, 32, device="cuda")[None, None, :, None]
    loss = (rendered * horizontal_weight).sum()
    loss.backward()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    strategy.step_post_backward(model.params, optimizers, state, 0, info, packed=True)

    assert model.means.shape[0] > 1
    assert all(parameter.shape[0] == model.means.shape[0] for parameter in model.params.values())


@pytest.mark.skipif(
    importlib.util.find_spec("gsplat") is None,
    reason="gsplat compatibility test requires the train environment",
)
def test_pinned_gsplat_opacity_reset_compatibility() -> None:
    from gsplat import DefaultStrategy

    initialization = make_initialization(count=1)
    initialization["opacities"] = torch.tensor([0.5])
    model = GaussianModel(initialization, sh_degree=0)
    optimizers = create_optimizers(model, {}, scene_scale=1.0)
    strategy = DefaultStrategy(reset_every=1, prune_opa=0.005)
    apply_gsplat_153_opacity_reset(
        strategy,
        model.params,
        optimizers,
        strategy.initialize_state(),
        step=1,
    )

    assert float(model.opacities[0]) == pytest.approx(0.01, abs=1.0e-6)
