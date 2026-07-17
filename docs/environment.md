# Environment Management

This repository uses `uv` as the single environment manager.

Do not run `pip install -r external/vggt/requirements.txt` directly in the
project environment. VGGT pins Torch, and the project must control the Python,
Torch, CUDA wheel index, VGGT, and local package versions together.

## Policy

- Use Python 3.10 for local and remote machines.
- Use PyTorch 2.3.1 and torchvision 0.18.1 for VGGT compatibility.
- Use the PyTorch CUDA 12.1 wheel index (`cu121`) as the default CUDA 12.x build.
- Install VGGT as an editable local source from `external/vggt`.
- Install this repository through `uv sync`.
- Keep `external/vggt` out of the main repository history.
- Install the pinned precompiled gsplat wheel through the project `train` extra.

The Python version is recorded in:

```text
.python-version
```

The dependency and CUDA index policy is recorded in:

```text
pyproject.toml
```

## Why CUDA 12.1

VGGT's original requirements pin:

```text
torch==2.3.1
torchvision==0.18.1
```

For PyTorch 2.3.1, the official CUDA 12.x wheel target is CUDA 12.1:

```text
https://download.pytorch.org/whl/cu121
```

This is usually the most compatible choice for remote servers that report a
newer CUDA 12.x driver through `nvidia-smi`, because PyTorch wheels include
their own CUDA runtime libraries and rely on the NVIDIA driver being new enough.

If the remote server specifically requires CUDA 12.4 or newer PyTorch wheels,
that should be handled as a deliberate Torch/VGGT compatibility upgrade rather
than a silent environment change.

## First Setup

From the repository root:

```bash
mkdir -p external
test -d external/vggt || git clone https://github.com/facebookresearch/vggt.git external/vggt
uv sync --extra dev --extra train
uv run python scripts/verify_env.py
```

`uv sync` will create the project virtual environment automatically, using
Python 3.10 from `.python-version`.

## Everyday Commands

Run tools through uv:

```bash
uv run python -m pytest -q
uv run python -m init.build_init --help
uv run python -m preprocess.run_vggt --help
uv run python scripts/verify_env.py
```

Activate the virtual environment only when you want an interactive shell:

```bash
source .venv/bin/activate
```

## Installing gsplat

Verify Torch first:

```bash
uv run python scripts/verify_env.py
```

Install the pinned Python 3.10 / PyTorch 2.3 / CUDA 12.1 gsplat wheel into
the same uv-managed environment:

```bash
uv sync --extra dev --extra train
```

The wheel source and exact build are recorded in `pyproject.toml` and
`uv.lock`. A CUDA 12.x-capable NVIDIA driver is still required, but this path
does not normally require a local CUDA compiler.

## Switching CUDA Wheel Index

The active CUDA wheel index is configured in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

For the current VGGT-pinned Torch 2.3.1 stack, keep `cu121`.

To move to `cu124`, `cu126`, or `cu128`, update the Torch and torchvision
versions together with the index URL, then regenerate the lock file:

```bash
uv lock --upgrade
uv sync --extra dev --extra train
```

Do not change only the CUDA index while keeping old Torch pins unless the
official PyTorch previous-version matrix confirms that exact pair exists.
