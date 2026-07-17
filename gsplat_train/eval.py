from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from init.io import load_config, resolve_scene_path

from .dataset import load_scene_data, split_view_indices
from .loss import masked_l1_loss, psnr, ssim
from .model import GaussianModel, load_torch_state
from .render import RenderConfig, rasterize_gaussians


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and evaluate gsplat Gaussians.")
    parser.add_argument("--config", required=True, help="Path to a gsplat training config.")
    parser.add_argument("--scene-root", default=None, help="Override scene root from config.")
    parser.add_argument("--model", default=None, help="Initialization, export, or checkpoint.")
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--output", default=None, help="Override evaluation output directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-views", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config.get("training", {})
    scene_cfg = config.get("scene", {})
    scene_root = Path(args.scene_root or scene_cfg.get("root", "data/scene_x"))
    model_value = args.model or training.get("model_path", "gsplat/final_gaussians.pt")
    model_path = resolve_scene_path(scene_root, model_value)
    output_dir = resolve_scene_path(
        scene_root, args.output or training.get("eval_output_dir", "gsplat/eval")
    )
    evaluate(
        config,
        scene_root=scene_root,
        model_path=model_path,
        output_dir=output_dir,
        split=args.split,
        device=torch.device(args.device),
        max_views=args.max_views,
    )


def evaluate(
    config: dict[str, Any],
    *,
    scene_root: Path,
    model_path: Path,
    output_dir: Path,
    split: str,
    device: torch.device,
    max_views: int | None,
) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("gsplat evaluation requires an available CUDA device")
    scene = load_scene_data(config, scene_root_override=scene_root, validate_projection=True)
    file_state = load_torch_state(model_path)
    model = GaussianModel.from_file(
        model_path, sh_degree=int(config.get("training", {}).get("sh_degree", 3))
    ).to(device)
    if "train_indices" in file_state and "test_indices" in file_state:
        train_indices = np.asarray(file_state["train_indices"], dtype=np.int64)
        test_indices = np.asarray(file_state["test_indices"], dtype=np.int64)
    else:
        train_indices, test_indices = split_view_indices(
            len(scene), test_every=int(config.get("training", {}).get("test_every", 8))
        )
    if split == "train":
        indices = train_indices
    elif split == "test":
        indices = test_indices
        if len(indices) == 0:
            raise ValueError("This scene/config has no held-out test views")
    else:
        indices = np.arange(len(scene), dtype=np.int64)
    if max_views is not None:
        if max_views <= 0:
            raise ValueError("max_views must be positive")
        indices = indices[:max_views]

    output_dir.mkdir(parents=True, exist_ok=True)
    render_config = RenderConfig.from_mapping(config.get("render"))
    per_view: list[dict[str, float | int]] = []
    with torch.no_grad():
        for view_index in indices.tolist():
            target, mask, intrinsics, viewmat = scene.frame(view_index, device=device)
            rendered, alpha, _ = rasterize_gaussians(
                model.params,
                viewmats=viewmat,
                intrinsics=intrinsics,
                width=scene.width,
                height=scene.height,
                sh_degree=model.max_sh_degree,
                config=render_config,
            )
            prediction = rendered[0].clamp(0.0, 1.0)
            metrics = {
                "view": int(view_index),
                "l1": float(masked_l1_loss(prediction, target, mask).cpu()),
                "psnr": float(psnr(prediction, target, mask=mask).cpu()),
                "ssim": float(ssim(prediction, target, mask=mask).cpu()),
                "alpha_mean": float(alpha[0].mean().cpu()),
            }
            per_view.append(metrics)
            save_rgb(output_dir / f"view_{view_index:03d}_target.png", target)
            save_rgb(output_dir / f"view_{view_index:03d}_render.png", prediction)
            save_rgb(
                output_dir / f"view_{view_index:03d}_error.png",
                torch.abs(prediction - target),
            )
            save_gray(output_dir / f"view_{view_index:03d}_alpha.png", alpha[0, ..., 0])
            print(
                f"view={view_index:03d} psnr={metrics['psnr']:.3f} "
                f"ssim={metrics['ssim']:.4f} l1={metrics['l1']:.6f}"
            )

    summary = {
        "views": len(per_view),
        "gaussians": int(model.means.shape[0]),
        "l1": float(np.mean([item["l1"] for item in per_view])),
        "psnr": float(np.mean([item["psnr"] for item in per_view])),
        "ssim": float(np.mean([item["ssim"] for item in per_view])),
        "alpha_mean": float(np.mean([item["alpha_mean"] for item in per_view])),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_view": per_view}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def save_rgb(path: Path, image: torch.Tensor) -> None:
    array = image.detach().cpu().clamp(0.0, 1.0).numpy()
    Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB").save(path)


def save_gray(path: Path, image: torch.Tensor) -> None:
    array = image.detach().cpu().clamp(0.0, 1.0).numpy()
    Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="L").save(path)


if __name__ == "__main__":
    main()
