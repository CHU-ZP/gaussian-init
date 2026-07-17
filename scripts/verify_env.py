from __future__ import annotations

import importlib.util
import sys

import torch
import torchvision


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Expected Python 3.10 from .python-version.")

    if torch.version.cuda is None:
        raise SystemExit("Expected a CUDA-enabled PyTorch build, got CPU-only torch.")

    if importlib.util.find_spec("vggt") is None:
        raise SystemExit("VGGT is not importable. Clone external/vggt and run uv sync.")

    print("vggt: importable")
    if importlib.util.find_spec("gsplat") is None:
        print("gsplat: not installed (run uv sync --extra train for optimization)")
    else:
        import gsplat

        print(f"gsplat: {getattr(gsplat, '__version__', 'unknown')}")


if __name__ == "__main__":
    main()
