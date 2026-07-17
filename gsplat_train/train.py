from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from init.io import load_config, resolve_scene_path

from .checkpoint import (
    load_checkpoint,
    move_strategy_state_to_device,
    restore_optimizer_states,
    restore_rng_state,
    save_checkpoint,
)
from .dataset import SceneData, load_scene_data, split_view_indices
from .loss import photometric_loss, psnr
from .model import GaussianModel
from .render import RenderConfig, rasterize_gaussians


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize an initialization with gsplat.")
    parser.add_argument("--config", required=True, help="Path to a gsplat training YAML config.")
    parser.add_argument("--scene-root", default=None, help="Override scene root from config.")
    parser.add_argument("--init", default=None, help="Override Gaussian initialization file.")
    parser.add_argument("--resume", default=None, help="Resume a training checkpoint.")
    parser.add_argument("--device", default=None, help="Override training device.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total steps.")
    parser.add_argument("--output-dir", default=None, help="Override scene-relative output dir.")
    parser.add_argument(
        "--disable-densification", action="store_true", help="Keep Gaussian topology fixed."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate data and parameters only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config, args=args)


def train(config: dict[str, Any], *, args: argparse.Namespace) -> None:
    training = config.get("training", {})
    scene_cfg = config.get("scene", {})
    scene_root = Path(args.scene_root or scene_cfg.get("root", "data/scene_x"))
    device = torch.device(args.device or training.get("device", "cuda"))
    seed = int(training.get("seed", 42))
    set_random_seed(seed)

    scene = load_scene_data(config, scene_root_override=scene_root, validate_projection=True)
    test_every = int(training.get("test_every", 8))
    train_indices, test_indices = split_view_indices(len(scene), test_every=test_every)
    max_sh_degree = int(training.get("sh_degree", 3))
    resume_value = args.resume or training.get("resume")
    resume_path = None if resume_value is None else resolve_scene_path(scene_root, resume_value)

    checkpoint = None
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path)
        model = GaussianModel.from_raw_state(
            checkpoint["splats"], sh_degree=int(checkpoint["max_sh_degree"])
        )
        if model.max_sh_degree != max_sh_degree:
            raise ValueError(
                f"Checkpoint SH degree {model.max_sh_degree} does not match config "
                f"degree {max_sh_degree}"
            )
        train_indices = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        test_indices = np.asarray(checkpoint["test_indices"], dtype=np.int64)
        start_step = int(checkpoint["step"]) + 1
    else:
        init_value = args.init or training.get(
            "init_path", scene_cfg.get("output_path", "init/fused_gaussians.pt")
        )
        init_path = resolve_scene_path(scene_root, init_value)
        model = GaussianModel.from_file(init_path, sh_degree=max_sh_degree)
        start_step = 0

    print(
        f"Loaded {len(scene)} frames ({len(train_indices)} train, {len(test_indices)} test), "
        f"{model.means.shape[0]} Gaussians, scene_scale={scene.scene_scale:.6g}."
    )
    if scene.reprojection_error_px is not None:
        print(f"VGGT camera projection p95 error: {scene.reprojection_error_px:.6g}px")
    if args.dry_run:
        return
    if device.type != "cuda":
        raise ValueError("gsplat optimization requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    model = model.to(device)
    optimizers = create_optimizers(model, training, scene_scale=scene.scene_scale)
    max_steps = int(args.max_steps or training.get("max_steps", 30_000))
    if max_steps <= start_step:
        raise ValueError(f"max_steps={max_steps} must be greater than resume step {start_step}")
    schedule_steps = int(training.get("lr_schedule_steps", training.get("max_steps", max_steps)))
    scheduler = create_position_scheduler(
        optimizers["means"], training, max_steps=schedule_steps
    )

    densification_cfg = training.get("densification", {})
    densification_enabled = bool(densification_cfg.get("enabled", True))
    densification_enabled &= not args.disable_densification
    strategy = create_strategy(densification_cfg) if densification_enabled else None
    strategy_state = None
    if strategy is not None:
        strategy.check_sanity(model.params, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene.scene_scale)

    if checkpoint is not None:
        checkpoint_scale = float(checkpoint["scene_scale"])
        if not np.isclose(checkpoint_scale, scene.scene_scale, rtol=1.0e-5, atol=1.0e-8):
            raise ValueError(
                f"Checkpoint scene_scale {checkpoint_scale:.8g} does not match current "
                f"scene_scale {scene.scene_scale:.8g}"
            )
        restore_optimizer_states(optimizers, checkpoint["optimizers"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if strategy is not None:
            restored = move_strategy_state_to_device(checkpoint.get("strategy_state"), device)
            if restored is not None:
                strategy_state = restored
            else:
                print("Starting a fresh densification state from a fixed-topology checkpoint.")
        restore_rng_state(checkpoint.get("rng"))

    render_config = RenderConfig.from_mapping(config.get("render"))
    output_dir = resolve_scene_path(
        scene_root, args.output_dir or training.get("output_dir", "gsplat")
    )
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = int(training.get("checkpoint_every", 1_000))
    log_every = max(int(training.get("log_every", 10)), 1)
    ssim_weight = float(training.get("ssim_weight", 0.2))
    sh_interval = int(training.get("sh_degree_interval", 1_000))
    random_background = bool(training.get("random_background", False))
    latest_metrics: dict[str, float] = {}

    try:
        for step in range(start_step, max_steps):
            view_index = int(random.choice(train_indices.tolist()))
            target, mask, intrinsics, viewmat = scene.frame(view_index, device=device)
            active_sh_degree = active_degree(
                step, maximum=model.max_sh_degree, interval=sh_interval
            )
            background = torch.rand(3, device=device) if random_background else None
            rendered, _, info = rasterize_gaussians(
                model.params,
                viewmats=viewmat,
                intrinsics=intrinsics,
                width=scene.width,
                height=scene.height,
                sh_degree=active_sh_degree,
                config=render_config,
                background=background,
                absgrad=bool(strategy is not None and strategy.absgrad),
            )
            prediction = rendered[0]
            if strategy is not None:
                assert strategy_state is not None
                strategy.step_pre_backward(
                    model.params, optimizers, strategy_state, step, info
                )
            loss, components = photometric_loss(
                prediction, target, mask=mask, ssim_weight=ssim_weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            loss.backward()

            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if strategy is not None:
                strategy.step_post_backward(
                    model.params,
                    optimizers,
                    strategy_state,
                    step,
                    info,
                    packed=render_config.packed,
                )
                apply_gsplat_153_opacity_reset(
                    strategy, model.params, optimizers, strategy_state, step=step
                )

            latest_metrics = {
                "step": step,
                "loss": float(loss.detach()),
                "l1": float(components["l1"].detach()),
                "ssim": float(components["ssim"].detach()),
                "psnr": float(psnr(prediction.detach(), target, mask=mask).detach()),
                "gaussians": int(model.means.shape[0]),
                "view": view_index,
                "sh_degree": active_sh_degree,
            }
            if step % log_every == 0 or step == max_steps - 1:
                ensure_finite_parameters(model)
                print(
                    f"step={step:06d} loss={latest_metrics['loss']:.6f} "
                    f"psnr={latest_metrics['psnr']:.3f} "
                    f"ssim={latest_metrics['ssim']:.4f} "
                    f"N={latest_metrics['gaussians']} sh={active_sh_degree}"
                )
            if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                save_training_checkpoint(
                    checkpoint_dir / f"step_{step:06d}.pt",
                    step=step,
                    model=model,
                    optimizers=optimizers,
                    scheduler=scheduler,
                    strategy_state=strategy_state,
                    config=config,
                    scene=scene,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
    except KeyboardInterrupt:
        interrupted_step = max(start_step, int(latest_metrics.get("step", start_step)))
        path = checkpoint_dir / f"interrupted_step_{interrupted_step:06d}.pt"
        save_training_checkpoint(
            path,
            step=interrupted_step,
            model=model,
            optimizers=optimizers,
            scheduler=scheduler,
            strategy_state=strategy_state,
            config=config,
            scene=scene,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        print(f"Interrupted checkpoint saved to {path}")
        return

    final_step = max_steps - 1
    final_checkpoint = checkpoint_dir / f"step_{final_step:06d}.pt"
    save_training_checkpoint(
        final_checkpoint,
        step=final_step,
        model=model,
        optimizers=optimizers,
        scheduler=scheduler,
        strategy_state=strategy_state,
        config=config,
        scene=scene,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    final_path = output_dir / "final_gaussians.pt"
    save_viewer_export(final_path, model=model, step=final_step, config=config)
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(latest_metrics, handle, indent=2)
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"Viewer-compatible Gaussians: {final_path}")


def create_optimizers(
    model: GaussianModel, training: Mapping[str, Any], *, scene_scale: float
) -> dict[str, torch.optim.Optimizer]:
    learning_rates = training.get("learning_rates", {})
    rates = {
        "means": float(learning_rates.get("means", 1.6e-4)) * scene_scale,
        "scales": float(learning_rates.get("scales", 5.0e-3)),
        "quats": float(learning_rates.get("quats", 1.0e-3)),
        "opacities": float(learning_rates.get("opacities", 5.0e-2)),
        "sh0": float(learning_rates.get("sh0", 2.5e-3)),
        "shN": float(learning_rates.get("shN", 2.5e-3 / 20.0)),
    }
    if any(rate <= 0.0 for rate in rates.values()):
        raise ValueError("All Gaussian learning rates must be positive")
    return {
        name: torch.optim.Adam([model.params[name]], lr=rate, eps=1.0e-15)
        for name, rate in rates.items()
    }


def create_position_scheduler(
    optimizer: torch.optim.Optimizer, training: Mapping[str, Any], *, max_steps: int
) -> torch.optim.lr_scheduler.ExponentialLR | None:
    final_factor = float(training.get("means_lr_final_factor", 0.01))
    if final_factor == 1.0:
        return None
    if not 0.0 < final_factor <= 1.0:
        raise ValueError("means_lr_final_factor must be in (0, 1]")
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=final_factor ** (1.0 / max(max_steps, 1))
    )


def create_strategy(values: Mapping[str, Any]):
    from gsplat import DefaultStrategy

    supported = {
        "prune_opa",
        "grow_grad2d",
        "grow_scale3d",
        "grow_scale2d",
        "prune_scale3d",
        "prune_scale2d",
        "refine_scale2d_stop_iter",
        "refine_start_iter",
        "refine_stop_iter",
        "reset_every",
        "refine_every",
        "pause_refine_after_reset",
        "absgrad",
        "revised_opacity",
        "verbose",
    }
    kwargs = {name: value for name, value in values.items() if name in supported}
    return DefaultStrategy(**kwargs)


def apply_gsplat_153_opacity_reset(
    strategy: Any,
    params: torch.nn.ParameterDict,
    optimizers: Mapping[str, torch.optim.Optimizer],
    state: Mapping[str, Any],
    *,
    step: int,
) -> None:
    """Compensate the reset-condition precedence bug in pinned gsplat 1.5.3."""
    import gsplat

    public_version = str(getattr(gsplat, "__version__", "")).split("+", maxsplit=1)[0]
    if (
        public_version != "1.5.3"
        or step <= 0
        or step >= strategy.refine_stop_iter
        or step % strategy.reset_every != 0
    ):
        return
    from gsplat.strategy.ops import reset_opa

    reset_opa(
        params=params,
        optimizers=dict(optimizers),
        state=dict(state),
        value=strategy.prune_opa * 2.0,
    )


def active_degree(step: int, *, maximum: int, interval: int) -> int:
    if interval <= 0:
        return maximum
    return min(step // interval, maximum)


def ensure_finite_parameters(model: GaussianModel) -> None:
    for name, value in model.params.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite Gaussian parameter: {name}")


def save_training_checkpoint(
    path: Path,
    *,
    step: int,
    model: GaussianModel,
    optimizers: Mapping[str, torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    strategy_state: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    scene: SceneData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    save_checkpoint(
        path,
        step=step,
        model=model,
        optimizers=optimizers,
        scheduler=scheduler,
        strategy_state=strategy_state,
        config=config,
        scene_scale=scene.scene_scale,
        train_indices=train_indices,
        test_indices=test_indices,
    )


def save_viewer_export(
    path: str | Path, *, model: GaussianModel, step: int, config: Mapping[str, Any]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.cpu() for name, value in model.activated_state().items()}
    state["metadata"] = {
        "stage": "optimized_gaussians",
        "step": int(step),
        "max_sh_degree": int(model.max_sh_degree),
        "training_config": dict(config.get("training", {})),
    }
    torch.save(state, output)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
