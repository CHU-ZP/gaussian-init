from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .model import GaussianModel, load_torch_state


CHECKPOINT_FORMAT = "vggt_pca_gsplat_training_v1"


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    model: GaussianModel,
    optimizers: Mapping[str, torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    strategy_state: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    scene_scale: float,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "max_sh_degree": model.max_sh_degree,
        "splats": model.raw_state_cpu(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "strategy_state": _to_cpu(strategy_state),
        "config": dict(config),
        "scene_scale": float(scene_scale),
        "train_indices": np.asarray(train_indices, dtype=np.int64),
        "test_indices": np.asarray(test_indices, dtype=np.int64),
        "rng": capture_rng_state(),
    }
    torch.save(payload, output)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = load_torch_state(path)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported training checkpoint format in {path}")
    required = {"step", "splats", "optimizers", "scene_scale", "train_indices", "test_indices"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Training checkpoint is missing keys: {missing}")
    return checkpoint


def restore_optimizer_states(
    optimizers: Mapping[str, torch.optim.Optimizer], states: Mapping[str, Any]
) -> None:
    if set(optimizers) != set(states):
        raise ValueError(
            "Checkpoint optimizer keys do not match model parameters: "
            f"current={sorted(optimizers)}, checkpoint={sorted(states)}"
        )
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(states[name])


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def move_strategy_state_to_device(
    state: Mapping[str, Any] | None, device: str | torch.device
) -> dict[str, Any] | None:
    if state is None:
        return None
    return {name: _to_device(value, device) for name, value in state.items()}


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _to_device(value: Any, device: str | torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value
