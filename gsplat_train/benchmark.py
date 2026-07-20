from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

import numpy as np


TRAIN_HISTORY_FIELDS = (
    "step",
    "optimization_seconds",
    "loss",
    "l1",
    "ssim",
    "psnr",
    "gaussians",
    "view",
    "sh_degree",
)
EVAL_HISTORY_FIELDS = (
    "step",
    "optimization_seconds",
    "views",
    "gaussians",
    "l1",
    "psnr",
    "ssim",
    "alpha_mean",
    "sh_degree",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    enabled: bool = False
    history_every: int = 1
    eval_every: int = 500
    preview_every: int = 0
    preview_warmup_steps: int = 0
    eval_split: str = "all"
    preview_views: tuple[int, ...] = (0, 12, 24, 36)
    evaluate_initialization: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "BenchmarkConfig":
        if values is None:
            return cls()
        allowed = {
            "enabled",
            "history_every",
            "eval_every",
            "preview_every",
            "preview_warmup_steps",
            "eval_split",
            "preview_views",
            "evaluate_initialization",
        }
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"Unknown benchmark config keys: {unknown}")
        preview_values = values.get("preview_views", cls.preview_views)
        config = cls(
            enabled=bool(values.get("enabled", cls.enabled)),
            history_every=int(values.get("history_every", cls.history_every)),
            eval_every=int(values.get("eval_every", cls.eval_every)),
            preview_every=int(values.get("preview_every", cls.preview_every)),
            preview_warmup_steps=int(
                values.get("preview_warmup_steps", cls.preview_warmup_steps)
            ),
            eval_split=str(values.get("eval_split", cls.eval_split)),
            preview_views=tuple(int(value) for value in preview_values),
            evaluate_initialization=bool(
                values.get("evaluate_initialization", cls.evaluate_initialization)
            ),
        )
        if config.history_every <= 0:
            raise ValueError("benchmark.history_every must be positive")
        if config.eval_every <= 0:
            raise ValueError("benchmark.eval_every must be positive")
        if config.preview_every < 0:
            raise ValueError("benchmark.preview_every must be non-negative")
        if config.preview_warmup_steps < 0:
            raise ValueError("benchmark.preview_warmup_steps must be non-negative")
        if config.preview_warmup_steps > 0 and config.preview_every <= 0:
            raise ValueError(
                "benchmark.preview_every must be enabled when preview_warmup_steps is positive"
            )
        if config.preview_every > 0 and not config.preview_views:
            raise ValueError(
                "benchmark.preview_views must not be empty when preview_every is enabled"
            )
        if config.eval_split not in {"train", "test", "all"}:
            raise ValueError("benchmark.eval_split must be train, test, or all")
        if len(set(config.preview_views)) != len(config.preview_views):
            raise ValueError("benchmark.preview_views must not contain duplicates")
        if any(view < 0 for view in config.preview_views):
            raise ValueError("benchmark.preview_views must be non-negative")
        return config

    def should_save_preview(self, step: int) -> bool:
        if self.preview_every <= 0 or step < 0:
            return False
        return step <= self.preview_warmup_steps or step % self.preview_every == 0

    def evaluation_indices(
        self,
        *,
        view_count: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> np.ndarray:
        if self.eval_split == "train":
            indices = np.asarray(train_indices, dtype=np.int64)
        elif self.eval_split == "test":
            indices = np.asarray(test_indices, dtype=np.int64)
        else:
            indices = np.arange(view_count, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"benchmark.eval_split={self.eval_split!r} contains no views")
        invalid_previews = sorted(set(self.preview_views).difference(indices.tolist()))
        if invalid_previews:
            raise ValueError(
                "benchmark.preview_views must belong to the selected evaluation split; "
                f"invalid values: {invalid_previews}"
            )
        return indices


class CsvHistory:
    """Append-only CSV output with explicit fresh-run and resume behavior."""

    def __init__(
        self,
        path: str | Path,
        fields: tuple[str, ...],
        *,
        fresh: bool,
    ) -> None:
        self.path = Path(path)
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            self.path.unlink(missing_ok=True)
        self._validate_existing_header()
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        self._handle: TextIO = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(fields))
        if needs_header:
            self._writer.writeheader()
            self._handle.flush()

    def write(self, values: Mapping[str, Any]) -> None:
        missing = sorted(set(self.fields).difference(values))
        unknown = sorted(set(values).difference(self.fields))
        if missing or unknown:
            raise ValueError(f"CSV row mismatch: missing={missing}, unknown={unknown}")
        self._writer.writerow({field: values[field] for field in self.fields})

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def _validate_existing_header(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        if header != list(self.fields):
            raise ValueError(f"Existing CSV header in {self.path} does not match expected fields")


def last_csv_float(path: str | Path, field: str, *, default: float = 0.0) -> float:
    input_path = Path(path)
    if not input_path.exists() or input_path.stat().st_size == 0:
        return default
    last_value = default
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(field):
                last_value = float(row[field])
    return last_value


def truncate_csv_after_step(path: str | Path, *, max_step: int) -> None:
    """Discard metric rows newer than a checkpoint before appending a resumed run."""
    input_path = Path(path)
    if not input_path.exists() or input_path.stat().st_size == 0:
        return
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if fields is None or "step" not in fields:
            raise ValueError(f"CSV history has no step column: {input_path}")
        rows = [row for row in reader if int(float(row["step"])) <= max_step]
    temporary = input_path.with_suffix(f"{input_path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(input_path)
