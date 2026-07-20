from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def run_command(command: list[str], *, cwd: Path, dry_run: bool, stage: str) -> None:
    print(f"\n== {stage} ==", flush=True)
    print(" ".join(shell_display(value) for value in command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def shell_display(value: str) -> str:
    if value and all(character.isalnum() or character in "-._/:=," for character in value):
        return value
    return repr(value)


def training_complete(run_dir: Path, max_steps: int) -> bool:
    summary_path = run_dir / "train_summary.json"
    model_path = run_dir / "final_gaussians.pt"
    eval_path = run_dir / "eval_history.csv"
    if not summary_path.exists() or not model_path.exists() or not eval_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        final_eval_step = last_csv_step(eval_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return int(summary["step"]) >= max_steps - 1 and final_eval_step >= max_steps


def last_csv_step(path: Path) -> int:
    last = -1
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            last = int(float(row["step"]))
    return last


def comparison_steps(max_steps: int) -> list[int]:
    candidates = (0, 1000, 3000, 8000, max_steps)
    return sorted({min(step, max_steps) for step in candidates})
