from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import viser

from init.gaussian_params import sh_dc_to_rgb


def load_gaussian_splats(path: str | Path) -> dict[str, np.ndarray]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = ("means", "covariances", "sh_dc", "opacities")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Missing Gaussian fields: {', '.join(missing)}")

    centers = state["means"].detach().cpu().numpy().astype(np.float32)
    covariances = state["covariances"].detach().cpu().numpy().astype(np.float32)
    rgbs = np.clip(
        sh_dc_to_rgb(state["sh_dc"].detach().cpu().numpy()),
        0.0,
        1.0,
    ).astype(np.float32)
    opacities = state["opacities"].detach().cpu().numpy().astype(np.float32).reshape(-1, 1)

    count = centers.shape[0]
    expected_shapes = {
        "means": (count, 3),
        "covariances": (count, 3, 3),
        "sh_dc": (count, 3),
        "opacities": (count, 1),
    }
    arrays = {
        "means": centers,
        "covariances": covariances,
        "sh_dc": rgbs,
        "opacities": opacities,
    }
    for key, expected_shape in expected_shapes.items():
        if arrays[key].shape != expected_shape:
            raise ValueError(f"{key} must have shape {expected_shape}, got {arrays[key].shape}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains non-finite values")

    return {
        "centers": centers,
        "covariances": covariances,
        "rgbs": rgbs,
        "opacities": np.clip(opacities, 0.0, 1.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View initialized Gaussians in a web browser.")
    parser.add_argument("--input", type=Path, required=True, help="Input fused_gaussians.pt path.")
    parser.add_argument("--host", default="127.0.0.1", help="Viewer bind address.")
    parser.add_argument("--port", type=int, default=8080, help="Viewer port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splats = load_gaussian_splats(args.input)
    if splats["centers"].shape[0] == 0:
        raise SystemExit("The input contains no Gaussians to visualize.")

    server = viser.ViserServer(host=args.host, port=args.port)
    scene_center = np.mean(splats["centers"], axis=0)
    server.scene.add_gaussian_splats(
        "/gaussians",
        centers=splats["centers"],
        covariances=splats["covariances"],
        rgbs=splats["rgbs"],
        opacities=splats["opacities"],
        position=-scene_center,
    )

    print(f"Loaded {splats['centers'].shape[0]} Gaussians from {args.input}")
    print(f"Open http://127.0.0.1:{server.get_port()} in a browser")
    print("Press Ctrl+C to stop the viewer")
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
