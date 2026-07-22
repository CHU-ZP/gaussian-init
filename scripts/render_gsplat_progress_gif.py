from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build side-by-side GIFs from saved gsplat benchmark renders."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Named training directory; pass once per method.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--views",
        default="12,36",
        help="Comma-separated benchmark view ids.",
    )
    parser.add_argument(
        "--view-labels",
        default=None,
        help="Optional comma-separated output labels corresponding to --views.",
    )
    parser.add_argument(
        "--step-every",
        type=int,
        default=1000,
        help="Keep one saved render every this many optimization steps.",
    )
    parser.add_argument(
        "--dense-first-steps",
        type=int,
        default=0,
        help="Keep every available frame through this step before applying --step-every.",
    )
    parser.add_argument("--cell-width", type=int, default=400)
    parser.add_argument("--duration-ms", type=int, default=180)
    parser.add_argument("--final-duration-ms", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step_every <= 0:
        raise ValueError("--step-every must be positive")
    if args.dense_first_steps < 0:
        raise ValueError("--dense-first-steps must be non-negative")
    if args.cell_width <= 0:
        raise ValueError("--cell-width must be positive")
    if args.duration_ms <= 0 or args.final_duration_ms <= 0:
        raise ValueError("GIF frame durations must be positive")

    runs = dict(parse_named_run(value) for value in args.run)
    if len(runs) < 2:
        raise ValueError("At least two distinct --run values are required")
    views = parse_views(args.views)
    view_labels = parse_view_labels(args.view_labels, views)
    histories = {name: load_eval_history(path) for name, path in runs.items()}
    steps = shared_render_steps(runs, views)
    selected_steps = select_steps(
        steps,
        every=args.step_every,
        dense_first_steps=args.dense_first_steps,
    )
    if not selected_steps:
        raise FileNotFoundError("No shared benchmark render steps were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for view_id, view_label in zip(views, view_labels, strict=True):
        output = args.output_dir / f"training_progress_{view_label}.gif"
        write_progress_gif(
            output,
            runs=runs,
            histories=histories,
            steps=selected_steps,
            view_id=view_id,
            view_label=view_label,
            cell_width=args.cell_width,
            duration_ms=args.duration_ms,
            final_duration_ms=args.final_duration_ms,
        )
        print(f"Wrote {output} ({len(selected_steps)} frames)")


def parse_named_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Run must use NAME=DIR syntax: {value}")
    name, path = value.split("=", 1)
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise ValueError(f"Run must use non-empty NAME=DIR syntax: {value}")
    return name, Path(path).expanduser().resolve()


def parse_views(value: str) -> list[int]:
    views = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not views:
        raise ValueError("--views must contain at least one view id")
    if any(view < 0 for view in views):
        raise ValueError("View ids must be non-negative")
    return views


def parse_view_labels(value: str | None, views: list[int]) -> list[str]:
    labels = (
        [f"view_{view:03d}" for view in views]
        if value is None
        else [item.strip() for item in value.split(",") if item.strip()]
    )
    if len(labels) != len(views):
        raise ValueError("--view-labels must contain one label per view")
    if len(labels) != len(set(labels)):
        raise ValueError("--view-labels must not contain duplicates")
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", label) is None for label in labels):
        raise ValueError("View labels may contain only letters, digits, underscores, and hyphens")
    return labels


def load_eval_history(run: Path) -> dict[int, dict[str, float]]:
    path = run / "eval_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation history: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(float(row["step"])): {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        }


def shared_render_steps(runs: dict[str, Path], views: list[int]) -> list[int]:
    shared: set[int] | None = None
    for run in runs.values():
        available = {
            int(path.name.removeprefix("step_"))
            for path in (run / "benchmark" / "renders").glob("step_*")
            if path.is_dir()
            and all((path / f"view_{view:03d}_render.png").exists() for view in views)
        }
        shared = available if shared is None else shared & available
    return sorted(shared or set())


def select_steps(steps: list[int], *, every: int, dense_first_steps: int = 0) -> list[int]:
    selected = [
        step for step in steps if step <= dense_first_steps or step == 0 or step % every == 0
    ]
    if steps and steps[-1] not in selected:
        selected.append(steps[-1])
    return selected


def write_progress_gif(
    output: Path,
    *,
    runs: dict[str, Path],
    histories: dict[str, dict[int, dict[str, float]]],
    steps: list[int],
    view_id: int,
    view_label: str | None = None,
    cell_width: int,
    duration_ms: int,
    final_duration_ms: int,
) -> None:
    first_run = next(iter(runs.values()))
    sample_path = render_path(first_run, steps[0], view_id, "render")
    with Image.open(sample_path) as sample:
        cell_height = round(sample.height * cell_width / sample.width)

    frames = [
        build_frame(
            runs=runs,
            histories=histories,
            step=step,
            final_step=steps[-1],
            view_id=view_id,
            view_label=view_label,
            cell_width=cell_width,
            cell_height=cell_height,
        ).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        for step in steps
    ]
    durations = [duration_ms] * len(frames)
    durations[-1] = final_duration_ms
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )


def build_frame(
    *,
    runs: dict[str, Path],
    histories: dict[str, dict[int, dict[str, float]]],
    step: int,
    final_step: int,
    view_id: int,
    view_label: str | None,
    cell_width: int,
    cell_height: int,
) -> Image.Image:
    title_height, label_height, metric_height, progress_height = 34, 24, 40, 8
    columns = 1 + len(runs)
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width,
            title_height + label_height + cell_height + metric_height + progress_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    display_view = view_label or f"view_{view_id:03d}"
    draw.text((10, 10), f"{display_view}     step {step:,}", fill="black")

    first_run = next(iter(runs.values()))
    columns_data: list[tuple[str, Path | None, dict[str, float] | None]] = [
        ("Target", target_path(first_run, step, view_id), None)
    ]
    columns_data.extend(
        (
            name,
            render_path(run, step, view_id, "render"),
            load_preview_metrics(run, step, view_id) or histories[name].get(step),
        )
        for name, run in runs.items()
    )
    image_y = title_height + label_height
    for column, (name, path, metrics) in enumerate(columns_data):
        x = column * cell_width
        draw_centered(draw, name, x, title_height, cell_width)
        image = load_resized(path, cell_width, cell_height)
        canvas.paste(image, (x, image_y))
        if metrics is not None:
            metric_text = (
                f"PSNR {metrics['psnr']:.2f} dB   "
                f"SSIM {metrics['ssim']:.3f}\n"
                f"N {int(metrics['gaussians']):,}"
            )
            draw.multiline_text(
                (x + 8, image_y + cell_height + 5),
                metric_text,
                fill="black",
                spacing=2,
            )

    progress_y = canvas.height - progress_height
    draw.rectangle((0, progress_y, canvas.width, canvas.height), fill="#e5e7eb")
    fraction = 1.0 if final_step <= 0 else step / final_step
    draw.rectangle(
        (0, progress_y, round(canvas.width * fraction), canvas.height),
        fill="#2563eb",
    )
    return canvas


def draw_centered(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int) -> None:
    bounds = draw.textbbox((0, 0), text)
    text_width = bounds[2] - bounds[0]
    draw.text((x + (width - text_width) / 2, y + 6), text, fill="black")


def render_path(run: Path, step: int, view_id: int, kind: str) -> Path:
    return run / "benchmark" / "renders" / f"step_{step:06d}" / f"view_{view_id:03d}_{kind}.png"


def target_path(run: Path, step: int, view_id: int) -> Path:
    current = render_path(run, step, view_id, "target")
    if current.exists():
        return current
    return render_path(run, 0, view_id, "target")


def load_preview_metrics(run: Path, step: int, view_id: int) -> dict[str, float] | None:
    path = run / "benchmark" / "renders" / f"step_{step:06d}" / "metrics.json"
    if not path.exists():
        return None
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        view = next(item for item in payload["per_view"] if int(item["view"]) == view_id)
        return {
            "psnr": float(view["psnr"]),
            "ssim": float(view["ssim"]),
            "l1": float(view["l1"]),
            "gaussians": float(summary["gaussians"]),
        }
    except (KeyError, OSError, TypeError, ValueError, StopIteration, json.JSONDecodeError):
        return None


def load_resized(path: Path | None, width: int, height: int) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("RGB", (width, height), "#e5e7eb")
    with Image.open(path) as image:
        return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


if __name__ == "__main__":
    main()
