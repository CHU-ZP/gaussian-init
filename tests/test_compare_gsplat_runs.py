from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from scripts.compare_gsplat_runs import (
    load_run,
    summarize_runs,
    write_convergence_svg,
    write_render_comparison,
)
from scripts.render_gsplat_progress_gif import select_steps


def test_compare_histories_and_render_montage(tmp_path: Path) -> None:
    runs: dict[str, Path] = {}
    run_data = {}
    for method_index, method in enumerate(("ellipse", "points")):
        run = tmp_path / method
        runs[method] = run
        write_csv(
            run / "train_history.csv",
            [
                {
                    "step": step,
                    "optimization_seconds": step / 10.0,
                    "loss": 0.2 / step + method_index * 0.01,
                    "l1": 0.1 / step,
                    "ssim": 0.8 + step / 10000.0,
                    "psnr": 20.0 + step / 100.0,
                    "gaussians": 100 + step,
                    "view": step % 4,
                    "sh_degree": 0,
                }
                for step in (1, 500, 1000)
            ],
        )
        write_csv(
            run / "eval_history.csv",
            [
                {
                    "step": step,
                    "optimization_seconds": step / 10.0,
                    "views": 4,
                    "gaussians": 100 + step,
                    "l1": 0.1 / (step + 1),
                    "psnr": 20.0 + step / 100.0 + method_index,
                    "ssim": 0.8 + step / 10000.0,
                    "alpha_mean": 0.5,
                    "sh_degree": 0,
                }
                for step in (0, 500, 1000)
            ],
        )
        for step in (0, 1000):
            render_dir = run / "benchmark" / "renders" / f"step_{step:06d}"
            render_dir.mkdir(parents=True)
            for kind, color in (
                ("target", (128, 128, 128)),
                ("render", (30 + 50 * method_index, 60, 90)),
                ("error", (10, 20, 30)),
            ):
                Image.new("RGB", (32, 20), color).save(render_dir / f"view_000_{kind}.png")
        run_data[method] = load_run(run)

    summaries = summarize_runs(run_data, thresholds=[25.0])
    assert len(summaries) == 2
    assert summaries[0]["initial_psnr"] == 20.0
    assert summaries[1]["final_psnr"] == 31.0

    svg_path = tmp_path / "convergence.svg"
    write_convergence_svg(svg_path, run_data, loss_window=2)
    assert "48-view PSNR" in svg_path.read_text(encoding="utf-8")

    montage_path = tmp_path / "comparison.png"
    write_render_comparison(montage_path, runs, steps=[0, 1000], view_id=0)
    with Image.open(montage_path) as montage:
        assert montage.width > 32 * 2
        assert montage.height > 20 * 2


def test_dense_gif_step_schedule_keeps_warmup_then_sparse_frames() -> None:
    steps = list(range(301))

    selected = select_steps(steps, every=100, dense_first_steps=100)

    assert selected == [*range(101), 200, 300]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
