from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from init.io import load_config, resolve_scene_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the VGGT strategy-sampling and COLMAP pipelines while saving "
            "dense fixed-view previews, then build comparison GIFs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gsplat_compare_48.yaml"),
    )
    parser.add_argument("--scene-root", type=Path, default=Path("data/tnt_truck_48"))
    parser.add_argument("--vggt-scene-data", default="vggt/predictions.npz")
    parser.add_argument("--vggt-init", default="init/fused_gaussians.pt")
    parser.add_argument("--colmap-scene-data", default="colmap/scene.npz")
    parser.add_argument("--colmap-init", default="init/colmap_sparse_gaussians.pt")
    parser.add_argument("--output-root", default="comparisons/dense_gif")
    parser.add_argument("--views", default="12,36")
    parser.add_argument("--preview-every", type=int, default=100)
    parser.add_argument("--dense-first-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete this script's output root before retraining both methods.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    config_path = args.config.expanduser().resolve()
    scene_root = args.scene_root.expanduser().resolve()
    config = load_config(config_path)
    views = parse_views(args.views)
    configured_steps = int(config.get("training", {}).get("max_steps", 30_000))
    max_steps = configured_steps if args.max_steps is None else args.max_steps
    validate_settings(
        config,
        max_steps=max_steps,
        preview_every=args.preview_every,
        dense_first_steps=args.dense_first_steps,
        eval_every=args.eval_every,
    )

    inputs = {
        "VGGT scene archive": resolve_scene_path(scene_root, args.vggt_scene_data).resolve(),
        "VGGT initialization": resolve_scene_path(scene_root, args.vggt_init).resolve(),
        "COLMAP scene archive": resolve_scene_path(
            scene_root, args.colmap_scene_data
        ).resolve(),
        "COLMAP initialization": resolve_scene_path(scene_root, args.colmap_init).resolve(),
    }
    missing = [f"{name}: {path}" for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing pipeline inputs:\n" + "\n".join(missing))

    output_root = resolve_scene_path(scene_root, args.output_root).resolve()
    ensure_owned_output(output_root, scene_root=scene_root)
    if args.restart and output_root.exists():
        if args.dry_run:
            print(f"[dry-run] remove {output_root}")
        else:
            shutil.rmtree(output_root)

    run_dirs = {
        "VGGT + sampling": output_root / "vggt_log_grid",
        "COLMAP sparse": output_root / "colmap_sparse",
    }
    existing_config = output_root / "dense_training_config.yaml"
    has_complete_run = any(
        training_complete(path, max_steps) for path in run_dirs.values()
    )
    if (
        has_complete_run
        and not args.restart
        and not saved_schedule_matches(
            existing_config,
            max_steps=max_steps,
            preview_every=args.preview_every,
            dense_first_steps=args.dense_first_steps,
            eval_every=args.eval_every,
            views=views,
        )
    ):
        raise RuntimeError(
            "Existing dense-GIF runs use a different frame/checkpoint schedule. "
            "Use --restart to retrain them with the requested schedule."
        )
    incomplete = [
        path
        for path in run_dirs.values()
        if path.exists() and any(path.iterdir()) and not training_complete(path, max_steps)
    ]
    if incomplete and not args.restart:
        formatted = "\n".join(str(path) for path in incomplete)
        raise RuntimeError(
            "Incomplete dense-GIF outputs already exist. Use --restart to replace them:\n"
            + formatted
        )

    dense_config = make_dense_config(
        config,
        views=views,
        preview_every=args.preview_every,
        dense_first_steps=args.dense_first_steps,
        eval_every=args.eval_every,
        max_steps=max_steps,
    )
    generated_config = output_root / "dense_training_config.yaml"
    if args.dry_run:
        print(f"[dry-run] write derived config {generated_config}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        generated_config.write_text(
            yaml.safe_dump(dense_config, sort_keys=False), encoding="utf-8"
        )

    commands: list[list[str]] = []
    started = time.time()
    specs = (
        (
            "VGGT + sampling",
            inputs["VGGT scene archive"],
            inputs["VGGT initialization"],
            run_dirs["VGGT + sampling"],
        ),
        (
            "COLMAP sparse",
            inputs["COLMAP scene archive"],
            inputs["COLMAP initialization"],
            run_dirs["COLMAP sparse"],
        ),
    )
    for name, scene_data, initialization, run_dir in specs:
        if not args.restart and training_complete(run_dir, max_steps):
            print(f"[skip] complete {name} run: {run_dir}")
            continue
        command = [
            sys.executable,
            "-m",
            "gsplat_train.train",
            "--config",
            str(generated_config),
            "--scene-root",
            str(scene_root),
            "--scene-data",
            str(scene_data),
            "--init",
            str(initialization),
            "--output-dir",
            str(run_dir),
            "--max-steps",
            str(max_steps),
            "--device",
            args.device,
        ]
        commands.append(command)
        run_command(
            command,
            cwd=repository_root,
            dry_run=args.dry_run,
            stage=f"retrain {name}",
        )
        if not args.dry_run and not training_complete(run_dir, max_steps):
            raise RuntimeError(
                f"{name} stopped before {max_steps:,} steps. Rerun with --restart."
            )

    report_dir = output_root / "report"
    report_command = [
        sys.executable,
        str(repository_root / "scripts" / "compare_gsplat_runs.py"),
        "--run",
        f"vggt_log_grid={run_dirs['VGGT + sampling']}",
        "--run",
        f"colmap_sparse={run_dirs['COLMAP sparse']}",
        "--output",
        str(report_dir),
        "--render-steps",
        ",".join(str(step) for step in comparison_steps(max_steps)),
        "--render-views",
        ",".join(str(view) for view in views),
    ]
    commands.append(report_command)
    run_command(
        report_command,
        cwd=repository_root,
        dry_run=args.dry_run,
        stage="write scalar and still-image report",
    )

    gif_command = [
        sys.executable,
        str(repository_root / "scripts" / "render_gsplat_progress_gif.py"),
        "--run",
        f"VGGT + sampling={run_dirs['VGGT + sampling']}",
        "--run",
        f"COLMAP sparse={run_dirs['COLMAP sparse']}",
        "--output-dir",
        str(report_dir),
        "--views",
        ",".join(str(view) for view in views),
        "--step-every",
        str(args.preview_every),
        "--dense-first-steps",
        str(args.dense_first_steps),
        "--duration-ms",
        str(max(30, min(180, args.preview_every))),
    ]
    commands.append(gif_command)
    run_command(
        gif_command,
        cwd=repository_root,
        dry_run=args.dry_run,
        stage="encode dense progress GIFs",
    )

    if not args.dry_run:
        manifest = {
            "status": "complete",
            "max_steps": max_steps,
            "preview_every": args.preview_every,
            "dense_first_steps": args.dense_first_steps,
            "eval_every": args.eval_every,
            "checkpoint_every": 0,
            "preview_views": views,
            "runs": {name: str(path) for name, path in run_dirs.items()},
            "report": str(report_dir),
            "elapsed_seconds": time.time() - started,
            "commands": commands,
        }
        (output_root / "dense_gif_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Dense GIF comparison complete: {report_dir}")


def parse_views(value: str) -> list[int]:
    views = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not views:
        raise ValueError("--views must contain at least one view id")
    if len(views) != len(set(views)):
        raise ValueError("--views must not contain duplicates")
    if any(view < 0 for view in views):
        raise ValueError("View ids must be non-negative")
    return views


def validate_settings(
    config: dict[str, Any],
    *,
    max_steps: int,
    preview_every: int,
    dense_first_steps: int,
    eval_every: int,
) -> None:
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if preview_every <= 0:
        raise ValueError("--preview-every must be positive")
    if dense_first_steps < 0:
        raise ValueError("--dense-first-steps must be non-negative")
    if eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if int(config.get("training", {}).get("test_every", 8)) > 0:
        raise ValueError("Dense pipeline comparison requires training.test_every <= 0")


def make_dense_config(
    config: dict[str, Any],
    *,
    views: list[int],
    preview_every: int,
    dense_first_steps: int,
    eval_every: int,
    max_steps: int,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    training = result.setdefault("training", {})
    training["max_steps"] = max_steps
    training["checkpoint_every"] = 0
    benchmark = result.setdefault("benchmark", {})
    benchmark.update(
        {
            "enabled": True,
            "eval_every": eval_every,
            "preview_every": preview_every,
            "preview_warmup_steps": dense_first_steps,
            "preview_views": views,
            "evaluate_initialization": True,
        }
    )
    return result


def run_command(command: list[str], *, cwd: Path, dry_run: bool, stage: str) -> None:
    print(f"\n== {stage} ==", flush=True)
    print(" ".join(shell_display(value) for value in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def shell_display(value: str) -> str:
    if value and all(character.isalnum() or character in "-._/:=," for character in value):
        return value
    return repr(value)


def training_complete(run_dir: Path, max_steps: int) -> bool:
    summary_path = run_dir / "train_summary.json"
    eval_path = run_dir / "eval_history.csv"
    final_path = run_dir / "final_gaussians.pt"
    if not summary_path.exists() or not eval_path.exists() or not final_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        last_eval = last_csv_step(eval_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return int(summary["step"]) >= max_steps - 1 and last_eval >= max_steps


def last_csv_step(path: Path) -> int:
    last = -1
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            last = int(float(row["step"]))
    return last


def saved_schedule_matches(
    path: Path,
    *,
    max_steps: int,
    preview_every: int,
    dense_first_steps: int,
    eval_every: int,
    views: list[int],
) -> bool:
    if not path.exists():
        return False
    try:
        config = load_config(path)
        training = config["training"]
        benchmark = config["benchmark"]
        return (
            int(training["max_steps"]) == max_steps
            and int(training["checkpoint_every"]) == 0
            and int(benchmark["preview_every"]) == preview_every
            and int(benchmark.get("preview_warmup_steps", 0)) == dense_first_steps
            and int(benchmark["eval_every"]) == eval_every
            and [int(view) for view in benchmark["preview_views"]] == views
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def comparison_steps(max_steps: int) -> list[int]:
    candidates = (0, 1000, 3000, 8000, max_steps)
    return sorted({min(step, max_steps) for step in candidates})


def ensure_owned_output(output_root: Path, *, scene_root: Path) -> None:
    if output_root == scene_root or scene_root not in output_root.parents:
        raise ValueError(f"Output root must be a child of the scene root: {output_root}")


if __name__ == "__main__":
    main()
