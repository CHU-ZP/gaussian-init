# External Dependencies

Third-party source trees are placed here for local development, but they are not
vendored into this repository.

Expected layout:

```text
external/
└── vggt/
```

Clone VGGT from the repository root:

```bash
git clone https://github.com/facebookresearch/vggt.git external/vggt
```

Do not run `pip install -r external/vggt/requirements.txt` directly in the
project environment. Use the root `pyproject.toml` and `uv sync` so Python,
Torch, CUDA wheels, VGGT, gsplat, and this repository stay compatible.
