from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gsplat benchmark histories and diagnostic renders."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Named training output directory; pass once per method.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--loss-window",
        type=int,
        default=200,
        help="Training-step window for the loss moving average.",
    )
    parser.add_argument("--thresholds", default="25,28,30")
    parser.add_argument("--render-steps", default="0,1000,3000,8000,30000")
    parser.add_argument("--render-views", default="0,12,24,36")
    parser.add_argument(
        "--view-labels",
        default=None,
        help="Optional comma-separated output labels corresponding to --render-views.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.loss_window <= 0:
        raise ValueError("--loss-window must be positive")
    runs = dict(parse_named_run(value) for value in args.run)
    if len(runs) < 2:
        raise ValueError("At least two distinct --run values are required")
    thresholds = parse_number_list(args.thresholds, float)
    render_steps = parse_number_list(args.render_steps, int)
    render_views = parse_number_list(args.render_views, int)
    view_labels = parse_view_labels(args.view_labels, render_views)

    run_data = {name: load_run(path) for name, path in runs.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = summarize_runs(run_data, thresholds=thresholds)
    write_summary(args.output, summaries, thresholds=thresholds)
    write_convergence_svg(
        args.output / "convergence.svg",
        run_data,
        loss_window=args.loss_window,
    )
    for view_id, view_label in zip(render_views, view_labels, strict=True):
        write_render_comparison(
            args.output / f"render_comparison_{view_label}.png",
            runs,
            steps=render_steps,
            view_id=view_id,
        )
    print(f"Comparison summary: {args.output / 'summary.csv'}")
    print(f"Convergence curves: {args.output / 'convergence.svg'}")
    print(f"Render comparisons: {args.output / 'render_comparison_*.png'}")


def parse_named_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Run must use NAME=DIR syntax: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name or not path.strip():
        raise ValueError(f"Run must use non-empty NAME=DIR syntax: {value}")
    return name, Path(path).expanduser().resolve()


def parse_number_list(value: str, conversion):
    return [conversion(item.strip()) for item in value.split(",") if item.strip()]


def parse_view_labels(value: str | None, views: list[int]) -> list[str]:
    labels = (
        [f"view_{view:03d}" for view in views]
        if value is None
        else [item.strip() for item in value.split(",") if item.strip()]
    )
    if len(labels) != len(views):
        raise ValueError("--view-labels must contain one label per render view")
    if len(labels) != len(set(labels)):
        raise ValueError("--view-labels must not contain duplicates")
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", label) is None for label in labels):
        raise ValueError("View labels may contain only letters, digits, underscores, and hyphens")
    return labels


def load_run(path: Path) -> dict[str, Any]:
    train_path = path / "train_history.csv"
    eval_path = path / "eval_history.csv"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Run {path} must contain train_history.csv and eval_history.csv")
    return {
        "path": path,
        "train": read_numeric_csv(train_path),
        "eval": read_numeric_csv(eval_path),
    }


def read_numeric_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"Metric history is empty: {path}")
    return rows


def summarize_runs(
    run_data: dict[str, dict[str, Any]], *, thresholds: list[float]
) -> list[dict[str, float | str | int | None]]:
    summaries: list[dict[str, float | str | int | None]] = []
    for name, data in run_data.items():
        evaluations = sorted(data["eval"], key=lambda row: row["step"])
        training = sorted(data["train"], key=lambda row: row["step"])
        initial = evaluations[0]
        final = evaluations[-1]
        best = max(evaluations, key=lambda row: row["psnr"])
        summary: dict[str, float | str | int | None] = {
            "method": name,
            "initial_psnr": initial["psnr"],
            "initial_ssim": initial["ssim"],
            "initial_l1": initial["l1"],
            "final_step": int(final["step"]),
            "final_psnr": final["psnr"],
            "final_ssim": final["ssim"],
            "final_l1": final["l1"],
            "best_psnr": best["psnr"],
            "best_psnr_step": int(best["step"]),
            "final_gaussians": int(final["gaussians"]),
            "optimization_seconds": training[-1]["optimization_seconds"],
        }
        for threshold in thresholds:
            reached = next((row for row in evaluations if row["psnr"] >= threshold), None)
            key = threshold_key(threshold)
            summary[f"psnr_{key}_step"] = None if reached is None else int(reached["step"])
            summary[f"psnr_{key}_seconds"] = (
                None if reached is None else reached["optimization_seconds"]
            )
        summaries.append(summary)
    return summaries


def write_summary(
    output_dir: Path,
    summaries: list[dict[str, float | str | int | None]],
    *,
    thresholds: list[float],
) -> None:
    fields = [
        "method",
        "initial_psnr",
        "initial_ssim",
        "initial_l1",
        "final_step",
        "final_psnr",
        "final_ssim",
        "final_l1",
        "best_psnr",
        "best_psnr_step",
        "final_gaussians",
        "optimization_seconds",
    ]
    for threshold in thresholds:
        key = threshold_key(threshold)
        fields.extend((f"psnr_{key}_step", f"psnr_{key}_seconds"))
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)


def threshold_key(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / min(index + 1, window))
    return result


def write_convergence_svg(
    path: Path,
    run_data: dict[str, dict[str, Any]],
    *,
    loss_window: int,
) -> None:
    panels = [
        ("48-view PSNR", "step", "psnr"),
        ("48-view SSIM", "step", "ssim"),
        ("48-view L1", "step", "l1"),
        ("Training loss (moving average)", "step", "loss_smooth"),
        ("Gaussian count", "step", "gaussians"),
    ]
    series_by_panel: list[dict[str, tuple[list[float], list[float]]]] = []
    for _, x_key, y_key in panels:
        panel_series: dict[str, tuple[list[float], list[float]]] = {}
        for name, data in run_data.items():
            rows = data["train"] if y_key in {"loss_smooth", "gaussians"} else data["eval"]
            x_values = [row[x_key] for row in rows]
            if y_key == "loss_smooth":
                y_values = rolling_mean([row["loss"] for row in rows], loss_window)
            else:
                y_values = [row[y_key] for row in rows]
            panel_series[name] = (x_values, y_values)
        series_by_panel.append(panel_series)

    width, height = 1500, 900
    panel_width, panel_height = 470, 360
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="38" font-size="24" font-family="sans-serif" '
        'font-weight="bold">gsplat initialization comparison</text>',
    ]
    panel_gap = 25
    for index, ((title, x_key, _), panel_series) in enumerate(
        zip(panels, series_by_panel, strict=True)
    ):
        row = 0 if index < 3 else 1
        column = index if row == 0 else index - 3
        columns_in_row = 3 if row == 0 else 2
        row_width = columns_in_row * panel_width + (columns_in_row - 1) * panel_gap
        left = (width - row_width) / 2 + column * (panel_width + panel_gap)
        top = 70 + row * 405
        elements.extend(
            svg_panel(
                title=title,
                x_label="optimization seconds" if x_key == "optimization_seconds" else "step",
                series=panel_series,
                left=left,
                top=top,
                width=panel_width,
                height=panel_height,
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def svg_panel(
    *,
    title: str,
    x_label: str,
    series: dict[str, tuple[list[float], list[float]]],
    left: int,
    top: int,
    width: int,
    height: int,
) -> list[str]:
    plot_left, plot_top = left + 62, top + 36
    plot_width, plot_height = width - 82, height - 88
    all_x = [value for values, _ in series.values() for value in values]
    all_y = [value for _, values in series.values() for value in values]
    x_min, x_max = finite_extent(all_x)
    y_min, y_max = finite_extent(all_y)
    y_margin = max((y_max - y_min) * 0.05, abs(y_max) * 1.0e-6, 1.0e-8)
    y_min -= y_margin
    y_max += y_margin

    def map_x(value: float) -> float:
        return plot_left + (value - x_min) / max(x_max - x_min, 1.0e-12) * plot_width

    def map_y(value: float) -> float:
        return plot_top + (y_max - value) / max(y_max - y_min, 1.0e-12) * plot_height

    output = [
        f'<text x="{left}" y="{top + 18}" font-size="17" font-family="sans-serif" '
        f'font-weight="bold">{html.escape(title)}</text>',
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" '
        f'height="{plot_height}" fill="#fafafa" stroke="#d1d5db"/>',
    ]
    for tick in range(5):
        fraction = tick / 4.0
        x = plot_left + fraction * plot_width
        y = plot_top + fraction * plot_height
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_max - fraction * (y_max - y_min)
        output.extend(
            [
                f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" '
                f'y2="{plot_top + plot_height}" stroke="#e5e7eb"/>',
                f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" '
                f'y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{x:.2f}" y="{plot_top + plot_height + 19}" text-anchor="middle" '
                f'font-size="11" font-family="sans-serif">{format_axis(x_value)}</text>',
                f'<text x="{plot_left - 8}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-size="11" font-family="sans-serif">{format_axis(y_value)}</text>',
            ]
        )
    output.append(
        f'<text x="{plot_left + plot_width / 2:.2f}" y="{top + height - 8}" '
        f'text-anchor="middle" font-size="12" font-family="sans-serif">{x_label}</text>'
    )
    for index, (name, (x_values, y_values)) in enumerate(series.items()):
        color = COLORS[index % len(COLORS)]
        stride = max(1, math.ceil(len(x_values) / 1500))
        pairs = list(zip(x_values[::stride], y_values[::stride], strict=True))
        if pairs and pairs[-1] != (x_values[-1], y_values[-1]):
            pairs.append((x_values[-1], y_values[-1]))
        points = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in pairs)
        output.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2" stroke-linejoin="round"/>'
        )
        legend_x = plot_left + 8 + (index % 2) * plot_width / 2
        legend_y = plot_top + 17 + (index // 2) * 17
        output.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 18}" '
                f'y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 23}" y="{legend_y}" font-size="11" '
                f'font-family="sans-serif">{html.escape(name)}</text>',
            ]
        )
    return output


def finite_extent(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    minimum, maximum = min(finite), max(finite)
    if minimum == maximum:
        margin = max(abs(minimum) * 0.05, 1.0)
        return minimum - margin, maximum + margin
    return minimum, maximum


def format_axis(value: float) -> str:
    absolute = abs(value)
    if absolute >= 10000:
        return f"{value / 1000:.0f}k"
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def write_render_comparison(
    path: Path,
    runs: dict[str, Path],
    *,
    steps: list[int],
    view_id: int,
) -> None:
    existing_steps = [
        step
        for step in steps
        if all(render_path(run, step, view_id, "render").exists() for run in runs.values())
    ]
    if not existing_steps:
        return
    sample = Image.open(
        render_path(next(iter(runs.values())), existing_steps[0], view_id, "render")
    )
    cell_width = min(sample.width, 360)
    cell_height = round(sample.height * cell_width / sample.width)
    label_height, row_label_width = 32, 88
    columns = 1 + 2 * len(runs)
    canvas = Image.new(
        "RGB",
        (row_label_width + columns * cell_width, label_height + len(existing_steps) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    headers = ["target"]
    for name in runs:
        headers.extend((f"{name} render", f"{name} error"))
    for column, title in enumerate(headers):
        draw.text(
            (row_label_width + column * cell_width + 6, 9),
            title,
            fill="black",
        )
    for row, step in enumerate(existing_steps):
        y = label_height + row * cell_height
        draw.text((8, y + 8), f"step {step}", fill="black")
        first_run = next(iter(runs.values()))
        target = load_resized(
            render_path(first_run, step, view_id, "target"), cell_width, cell_height
        )
        canvas.paste(target, (row_label_width, y))
        column = 1
        for run in runs.values():
            for kind in ("render", "error"):
                image = load_resized(render_path(run, step, view_id, kind), cell_width, cell_height)
                canvas.paste(image, (row_label_width + column * cell_width, y))
                column += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_path(run: Path, step: int, view_id: int, kind: str) -> Path:
    return run / "benchmark" / "renders" / f"step_{step:06d}" / f"view_{view_id:03d}_{kind}.png"


def load_resized(path: Path, width: int, height: int) -> Image.Image:
    if not path.exists():
        return Image.new("RGB", (width, height), "#e5e7eb")
    with Image.open(path) as image:
        return image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)


if __name__ == "__main__":
    main()
