from __future__ import annotations

import argparse
from pathlib import Path

from init.io import load_config, resolve_scene_path

from .model import GaussianModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained or initialized Gaussian model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--scene-root", default=None, help="Override scene root from config.")
    parser.add_argument("--model", default=None, help="Model or init file to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    scene_cfg = config.get("scene", {})
    scene_root = Path(args.scene_root or scene_cfg.get("root", "data/scene_x"))
    model_path = resolve_scene_path(scene_root, args.model or scene_cfg.get("output_path", "init/fused_gaussians.pt"))
    model = GaussianModel.from_file(model_path)
    print(f"Loaded {model.means.shape[0]} Gaussians from {model_path}.")
    print("Rendering metrics should be added with the concrete gsplat evaluation loop.")


if __name__ == "__main__":
    main()
