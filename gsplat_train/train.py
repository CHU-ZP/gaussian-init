from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from init.io import load_config, resolve_scene_path, resolve_scene_root

from .benchmark import (
    EVAL_HISTORY_FIELDS,
    TRAIN_HISTORY_FIELDS,
    BenchmarkConfig,
    CsvHistory,
    last_csv_float,
    truncate_csv_after_step,
)
from .checkpoint import (
    load_checkpoint,
    move_strategy_state_to_device,
    restore_optimizer_states,
    restore_rng_state,
    save_checkpoint,
)
from .dataset import SceneData, load_scene_data, split_view_indices
from .eval import evaluate_model_views
from .loss import photometric_loss, psnr
from .model import GaussianModel
from .render import RenderConfig, rasterize_gaussians


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize an initialization with gsplat.")
    parser.add_argument("--config", required=True, help="Path to a gsplat training YAML config.")
    parser.add_argument("--scene-root", default=None, help="Override scene root from config.")
    parser.add_argument(
        "--scene-data",
        default=None,
        help="Override the dense-prediction or camera-only scene archive.",
    )
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
    scene_root = resolve_scene_root(config, args.scene_root)
    device = torch.device(args.device or training.get("device", "cuda"))
    seed = int(training.get("seed", 42))
    set_random_seed(seed)

    scene = load_scene_data(
        config,
        scene_root_override=scene_root,
        predictions_override=args.scene_data,
        validate_projection=True,
    )
    test_every = int(training.get("test_every", 8))
    train_indices, test_indices = split_view_indices(len(scene), test_every=test_every)
    benchmark_config = BenchmarkConfig.from_mapping(config.get("benchmark"))
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
        print(f"Camera reprojection diagnostic: {scene.reprojection_error_px:.6g}px")
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
    scheduler = create_position_scheduler(optimizers["means"], training, max_steps=schedule_steps)

    densification_cfg = training.get("densification", {})
    densification_enabled = bool(densification_cfg.get("enabled", True))
    densification_enabled &= not args.disable_densification
    strategy = create_strategy(densification_cfg) if densification_enabled else None
    strategy_state = None
    if strategy is not None:
        maximum = getattr(strategy, "max_gaussians", None)
        if maximum is not None and model.means.shape[0] > maximum:
            raise ValueError(
                f"Checkpoint/initialization has {model.means.shape[0]} Gaussians, above "
                f"densification.max_gaussians={maximum}. Increase the cap or resume with "
                "--disable-densification; existing Gaussians are never deleted just to meet a cap."
            )
        if maximum is not None:
            print(f"Densification Gaussian cap: {maximum}")
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
    train_history: CsvHistory | None = None
    eval_history: CsvHistory | None = None
    optimization_seconds = 0.0
    benchmark_indices = np.empty(0, dtype=np.int64)
    last_benchmark_step: int | None = None

    if benchmark_config.enabled:
        benchmark_indices = benchmark_config.evaluation_indices(
            view_count=len(scene),
            train_indices=train_indices,
            test_indices=test_indices,
        )
        history_path = output_dir / "train_history.csv"
        eval_history_path = output_dir / "eval_history.csv"
        if checkpoint is not None:
            truncate_csv_after_step(history_path, max_step=start_step)
            truncate_csv_after_step(eval_history_path, max_step=start_step)
            optimization_seconds = last_csv_float(history_path, "optimization_seconds", default=0.0)
        train_history = CsvHistory(
            history_path,
            TRAIN_HISTORY_FIELDS,
            fresh=checkpoint is None,
        )
        eval_history = CsvHistory(
            eval_history_path,
            EVAL_HISTORY_FIELDS,
            fresh=checkpoint is None,
        )
        print(
            "Benchmark recording enabled: "
            f"history every {benchmark_config.history_every} step(s), "
            f"{len(benchmark_indices)}-view evaluation every "
            f"{benchmark_config.eval_every} step(s)."
        )
        if benchmark_config.preview_every > 0:
            print(
                "Dense preview recording enabled: "
                f"views {list(benchmark_config.preview_views)} every step through "
                f"step {benchmark_config.preview_warmup_steps}, then every "
                f"{benchmark_config.preview_every} step(s)."
            )
        if checkpoint is None and benchmark_config.evaluate_initialization:
            run_benchmark_evaluation(
                step=0,
                optimization_seconds=optimization_seconds,
                scene=scene,
                model=model,
                indices=benchmark_indices,
                render_config=render_config,
                sh_degree=active_degree(0, maximum=model.max_sh_degree, interval=sh_interval),
                output_dir=output_dir,
                preview_views=benchmark_config.preview_views,
                history=eval_history,
            )
            last_benchmark_step = 0

    try:
        for step in range(start_step, max_steps):
            step_started = time.perf_counter()
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
                strategy.step_pre_backward(model.params, optimizers, strategy_state, step, info)
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
            optimization_seconds += time.perf_counter() - step_started
            completed_steps = step + 1
            if train_history is not None and (
                completed_steps % benchmark_config.history_every == 0
                or completed_steps == max_steps
            ):
                train_history.write(
                    {
                        "step": completed_steps,
                        "optimization_seconds": optimization_seconds,
                        "loss": latest_metrics["loss"],
                        "l1": latest_metrics["l1"],
                        "ssim": latest_metrics["ssim"],
                        "psnr": latest_metrics["psnr"],
                        "gaussians": latest_metrics["gaussians"],
                        "view": latest_metrics["view"],
                        "sh_degree": latest_metrics["sh_degree"],
                    }
                )
            if step % log_every == 0 or step == max_steps - 1:
                ensure_finite_parameters(model)
                print(
                    f"step={step:06d} loss={latest_metrics['loss']:.6f} "
                    f"psnr={latest_metrics['psnr']:.3f} "
                    f"ssim={latest_metrics['ssim']:.4f} "
                    f"N={latest_metrics['gaussians']} sh={active_sh_degree}"
                )
                if train_history is not None:
                    train_history.flush()
            full_evaluation_due = (
                eval_history is not None
                and completed_steps % benchmark_config.eval_every == 0
            )
            if full_evaluation_due:
                run_benchmark_evaluation(
                    step=completed_steps,
                    optimization_seconds=optimization_seconds,
                    scene=scene,
                    model=model,
                    indices=benchmark_indices,
                    render_config=render_config,
                    sh_degree=active_sh_degree,
                    output_dir=output_dir,
                    preview_views=benchmark_config.preview_views,
                    history=eval_history,
                )
                last_benchmark_step = completed_steps
            elif (
                eval_history is not None
                and benchmark_config.should_save_preview(completed_steps)
            ):
                run_benchmark_preview(
                    step=completed_steps,
                    scene=scene,
                    model=model,
                    render_config=render_config,
                    sh_degree=active_sh_degree,
                    output_dir=output_dir,
                    preview_views=benchmark_config.preview_views,
                )
            if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                if train_history is not None:
                    train_history.flush()
                if eval_history is not None:
                    eval_history.flush()
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
        if train_history is not None:
            train_history.close()
        if eval_history is not None:
            eval_history.close()
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

    if eval_history is not None and last_benchmark_step != max_steps:
        run_benchmark_evaluation(
            step=max_steps,
            optimization_seconds=optimization_seconds,
            scene=scene,
            model=model,
            indices=benchmark_indices,
            render_config=render_config,
            sh_degree=active_degree(
                max_steps - 1, maximum=model.max_sh_degree, interval=sh_interval
            ),
            output_dir=output_dir,
            preview_views=benchmark_config.preview_views,
            history=eval_history,
        )
    if train_history is not None:
        train_history.close()
    if eval_history is not None:
        eval_history.close()

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


def run_benchmark_evaluation(
    *,
    step: int,
    optimization_seconds: float,
    scene: SceneData,
    model: GaussianModel,
    indices: np.ndarray,
    render_config: RenderConfig,
    sh_degree: int,
    output_dir: Path,
    preview_views: tuple[int, ...],
    history: CsvHistory,
) -> dict[str, float | int]:
    preview_dir = output_dir / "benchmark" / "renders" / f"step_{step:06d}"
    summary, _ = evaluate_model_views(
        scene=scene,
        model=model,
        indices=indices,
        render_config=render_config,
        output_dir=preview_dir,
        save_view_ids=preview_views,
        sh_degree=sh_degree,
        print_per_view=False,
    )
    history.write(
        {
            "step": step,
            "optimization_seconds": optimization_seconds,
            "views": summary["views"],
            "gaussians": summary["gaussians"],
            "l1": summary["l1"],
            "psnr": summary["psnr"],
            "ssim": summary["ssim"],
            "alpha_mean": summary["alpha_mean"],
            "sh_degree": sh_degree,
        }
    )
    history.flush()
    print(
        f"benchmark step={step:06d} views={summary['views']} "
        f"psnr={summary['psnr']:.3f} ssim={summary['ssim']:.4f} "
        f"l1={summary['l1']:.6f} N={summary['gaussians']}"
    )
    return summary


def run_benchmark_preview(
    *,
    step: int,
    scene: SceneData,
    model: GaussianModel,
    render_config: RenderConfig,
    sh_degree: int,
    output_dir: Path,
    preview_views: tuple[int, ...],
) -> dict[str, float | int]:
    """Save lightweight fixed-view renders without evaluating every benchmark view."""
    preview_dir = output_dir / "benchmark" / "renders" / f"step_{step:06d}"
    summary, _ = evaluate_model_views(
        scene=scene,
        model=model,
        indices=np.asarray(preview_views, dtype=np.int64),
        render_config=render_config,
        output_dir=preview_dir,
        save_view_ids=preview_views,
        sh_degree=sh_degree,
        print_per_view=False,
        save_diagnostics=False,
    )
    print(
        f"preview step={step:06d} views={summary['views']} "
        f"psnr={summary['psnr']:.3f} ssim={summary['ssim']:.4f} "
        f"N={summary['gaussians']}"
    )
    return summary


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
    from gsplat.strategy.ops import duplicate, split

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
    control_keys = {"enabled", "max_gaussians"}
    unknown = sorted(set(values).difference(supported | control_keys))
    if unknown:
        raise ValueError(f"Unknown densification config keys: {unknown}")
    kwargs = {name: value for name, value in values.items() if name in supported}
    maximum = values.get("max_gaussians")
    if maximum is None:
        return DefaultStrategy(**kwargs)
    maximum = int(maximum)
    if maximum <= 0:
        raise ValueError("densification.max_gaussians must be positive")

    class CappedDefaultStrategy(DefaultStrategy):
        def __init__(self, *, max_gaussians: int, **strategy_kwargs: Any) -> None:
            super().__init__(**strategy_kwargs)
            self.max_gaussians = max_gaussians

        @torch.no_grad()
        def _grow_gs(
            self,
            params: Any,
            optimizers: dict[str, torch.optim.Optimizer],
            state: dict[str, Any],
            step: int,
        ) -> tuple[int, int]:
            remaining = self.max_gaussians - int(params["means"].shape[0])
            if remaining <= 0:
                return 0, 0

            counts = state["count"]
            gradients = state["grad2d"] / counts.clamp_min(1)
            is_gradient_high = gradients > self.grow_grad2d
            is_small = (
                torch.exp(params["scales"]).max(dim=-1).values
                <= self.grow_scale3d * state["scene_scale"]
            )
            duplicate_candidates = is_gradient_high & is_small
            split_candidates = is_gradient_high & ~is_small
            if step < self.refine_scale2d_stop_iter:
                split_candidates |= state["radii"] > self.grow_scale2d
                duplicate_candidates &= ~split_candidates
            candidates = duplicate_candidates | split_candidates
            candidate_indices = torch.nonzero(candidates, as_tuple=False).flatten()
            if candidate_indices.numel() > remaining:
                priorities = gradients[candidate_indices]
                keep = torch.topk(priorities, k=remaining, sorted=False).indices
                selected = torch.zeros_like(candidates)
                selected[candidate_indices[keep]] = True
            else:
                selected = candidates

            is_duplicate = selected & duplicate_candidates
            is_split = selected & split_candidates
            duplicate_count = int(is_duplicate.sum().item())
            split_count = int(is_split.sum().item())

            if duplicate_count > 0:
                duplicate(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=is_duplicate,
                )
            if split_count > 0:
                is_split = torch.cat(
                    [
                        is_split,
                        torch.zeros(
                            duplicate_count,
                            dtype=torch.bool,
                            device=is_split.device,
                        ),
                    ]
                )
                split(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=is_split,
                    revised_opacity=self.revised_opacity,
                )
            return duplicate_count, split_count

    return CappedDefaultStrategy(max_gaussians=maximum, **kwargs)


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
