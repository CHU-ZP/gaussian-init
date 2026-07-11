from __future__ import annotations

import argparse
from pathlib import Path

from init.io import load_config, resolve_scene_path

from .dataset import load_scene_data
from .model import GaussianModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gsplat from PCA Gaussian initialization.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--scene-root", default=None, help="Override scene root from config.")
    parser.add_argument("--init", default=None, help="Override Gaussian init file.")
    parser.add_argument("--dry-run", action="store_true", help="Only load data and initialization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    scene_cfg = config.get("scene", {})
    scene_root = Path(args.scene_root or scene_cfg.get("root", "data/scene_x"))
    init_path = resolve_scene_path(scene_root, args.init or scene_cfg.get("output_path", "init/fused_gaussians.pt"))

    scene = load_scene_data(config, scene_root_override=scene_root)
    model = GaussianModel.from_file(init_path)
    print(
        f"Loaded {len(scene)} frames and {model.means.shape[0]} Gaussians from {init_path}."
    )

    if args.dry_run:
        return

    try:
        import gsplat  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "gsplat is not installed in this environment. Install gsplat and then fill in "
            "the rasterization loop in gsplat_train/train.py."
        ) from exc

    raise SystemExit(
        "The concrete gsplat training loop is intentionally left as the Stage 0 integration task."
    )


if __name__ == "__main__":
    main()
