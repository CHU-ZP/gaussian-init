from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from init.io import load_config, resolve_scene_path

if __package__:
    from scripts import pipeline_runner_utils as _runner_utils
else:
    import pipeline_runner_utils as _runner_utils

comparison_steps = _runner_utils.comparison_steps
last_csv_step = _runner_utils.last_csv_step
run_command = _runner_utils.run_command
shell_display = _runner_utils.shell_display
training_complete = _runner_utils.training_complete


CHECKPOINT_PATTERN = re.compile(r"(?:interrupted_)?step_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run aligned PINHOLE COLMAP SfM, build sparse Gaussians, train with the "
            "shared gsplat benchmark config, and compare against the VGGT pipeline."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/gsplat_compare_48.yaml"))
    parser.add_argument("--scene-root", type=Path, default=Path("data/tnt_truck_48"))
    parser.add_argument("--predictions", default="vggt/predictions.npz")
    parser.add_argument("--colmap-root", default="colmap")
    parser.add_argument("--scene-data", default="colmap/scene.npz")
    parser.add_argument("--init", default="init/colmap_sparse_gaussians.pt")
    parser.add_argument("--output-dir", default="comparisons/colmap_sparse")
    parser.add_argument("--compare-run", default="comparisons/region_init")
    parser.add_argument("--report-dir", default="comparisons/colmap_report")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--max-num-features", type=int, default=8192)
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    parser.add_argument("--restart-colmap", action="store_true")
    parser.add_argument("--restart-training", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    config_path = args.config.expanduser().resolve()
    scene_root = args.scene_root.expanduser().resolve()
    config = load_config(config_path)
    configured_steps = int(config.get("training", {}).get("max_steps", 30_000))
    max_steps = configured_steps if args.max_steps is None else int(args.max_steps)
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if int(config.get("training", {}).get("test_every", 8)) > 0:
        raise ValueError("The aligned full-view benchmark requires training.test_every <= 0")

    predictions_path = resolve_scene_path(scene_root, args.predictions).resolve()
    colmap_root = resolve_scene_path(scene_root, args.colmap_root).resolve()
    scene_data_path = resolve_scene_path(scene_root, args.scene_data).resolve()
    init_path = resolve_scene_path(scene_root, args.init).resolve()
    training_dir = resolve_scene_path(scene_root, args.output_dir).resolve()
    comparison_run = resolve_scene_path(scene_root, args.compare_run).resolve()
    report_dir = resolve_scene_path(scene_root, args.report_dir).resolve()
    manifest_path = colmap_root / "pipeline_manifest.json"
    commands: list[list[str]] = []
    started = time.time()

    if args.restart_training:
        remove_owned_directory(training_dir, scene_root=scene_root, dry_run=args.dry_run)
        remove_owned_directory(report_dir, scene_root=scene_root, dry_run=args.dry_run)

    reconstruction_command = [
        sys.executable,
        "-m",
        "preprocess.run_colmap",
        "--scene-root",
        str(scene_root),
        "--predictions",
        str(predictions_path),
        "--output-root",
        str(colmap_root),
        "--device",
        args.device,
        "--gpu-index",
        args.gpu_index,
        "--max-num-features",
        str(args.max_num_features),
    ]
    if args.restart_colmap:
        reconstruction_command.append("--restart")
    commands.append(reconstruction_command)
    run_command(
        reconstruction_command,
        cwd=repository_root,
        dry_run=args.dry_run,
        stage="aligned PINHOLE COLMAP reconstruction",
    )

    initialization_command = [
        sys.executable,
        "-m",
        "init.build_colmap_init",
        "--scene-root",
        str(scene_root),
        "--predictions",
        str(predictions_path),
        "--model",
        str(colmap_root / "sparse" / "0"),
        "--scene-output",
        str(scene_data_path),
        "--output",
        str(init_path),
        "--device",
        args.device,
        "--knn-chunk-size",
        str(args.knn_chunk_size),
    ]
    commands.append(initialization_command)
    run_command(
        initialization_command,
        cwd=repository_root,
        dry_run=args.dry_run,
        stage="COLMAP sparse Gaussian initialization",
    )

    if args.skip_training:
        print(f"COLMAP scene archive: {scene_data_path}")
        print(f"COLMAP Gaussian initialization: {init_path}")
        return
    if not args.dry_run:
        for path in (scene_data_path, init_path):
            if not path.exists():
                raise FileNotFoundError(f"COLMAP pipeline stage did not create {path}")

    resume_path = None if args.no_resume else latest_checkpoint(training_dir / "checkpoints")
    if training_complete(training_dir, max_steps=max_steps):
        print(f"[skip] COLMAP training already complete: {training_dir}", flush=True)
    else:
        if resume_path is not None and checkpoint_completed_steps(resume_path) >= max_steps:
            raise RuntimeError(
                "COLMAP checkpoint reached the target but benchmark artifacts are incomplete; "
                "rerun with --restart-training"
            )
        if training_dir.exists() and any(training_dir.iterdir()) and resume_path is None:
            if not args.dry_run:
                raise RuntimeError(
                    f"Non-empty COLMAP training output has no checkpoint: {training_dir}. "
                    "Use --restart-training."
                )
        training_command = [
            sys.executable,
            "-m",
            "gsplat_train.train",
            "--config",
            str(config_path),
            "--scene-root",
            str(scene_root),
            "--scene-data",
            str(scene_data_path),
            "--max-steps",
            str(max_steps),
            "--device",
            args.device,
            "--output-dir",
            str(training_dir),
        ]
        if resume_path is None:
            training_command.extend(("--init", str(init_path)))
        else:
            training_command.extend(("--resume", str(resume_path)))
        commands.append(training_command)
        run_command(
            training_command,
            cwd=repository_root,
            dry_run=args.dry_run,
            stage="COLMAP-initialized gsplat training",
        )

    report_written = False
    if not args.skip_report:
        if args.dry_run or (
            training_complete(training_dir, max_steps=max_steps)
            and (comparison_run / "train_history.csv").exists()
            and (comparison_run / "eval_history.csv").exists()
        ):
            report_command = [
                sys.executable,
                str(repository_root / "scripts" / "compare_gsplat_runs.py"),
                "--run",
                f"region_init={comparison_run}",
                "--run",
                f"colmap_sparse={training_dir}",
                "--output",
                str(report_dir),
                "--render-steps",
                ",".join(str(step) for step in comparison_steps(max_steps)),
            ]
            commands.append(report_command)
            run_command(
                report_command,
                cwd=repository_root,
                dry_run=args.dry_run,
                stage="VGGT versus COLMAP report",
            )
            report_written = True
        else:
            print(
                f"[skip] VGGT comparison run is unavailable or incomplete: {comparison_run}",
                flush=True,
            )

    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "max_steps": max_steps,
                    "vggt_alignment_archive": str(predictions_path),
                    "scene_data": str(scene_data_path),
                    "initialization": str(init_path),
                    "training_output": str(training_dir),
                    "comparison_run": str(comparison_run),
                    "report": str(report_dir) if report_written else None,
                    "elapsed_seconds": time.time() - started,
                    "commands": commands,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"COLMAP pipeline complete: {training_dir}")
        print(f"Pipeline manifest: {manifest_path}")


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates: list[tuple[int, int, Path]] = []
    for path in checkpoint_dir.glob("*.pt"):
        match = CHECKPOINT_PATTERN.search(path.name)
        if match is not None:
            candidates.append(
                (int(match.group(1)), int(path.name.startswith("interrupted_")), path)
            )
    return None if not candidates else max(candidates)[2]


def checkpoint_completed_steps(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unrecognized checkpoint filename: {path}")
    return int(match.group(1)) + 1


def remove_owned_directory(path: Path, *, scene_root: Path, dry_run: bool) -> None:
    resolved = path.resolve()
    root = scene_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to remove output outside scene root: {path}")
    if resolved.exists():
        if dry_run:
            print(f"[dry-run] remove {resolved}")
        else:
            shutil.rmtree(resolved)


if __name__ == "__main__":
    main()
