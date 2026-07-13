from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from init.gaussian_params import rgb_to_sh_dc
from scripts.view_gaussians import load_gaussian_splats


def test_load_gaussian_splats_for_viser(tmp_path: Path) -> None:
    path = tmp_path / "gaussians.pt"
    rgbs = np.asarray([[0.2, 0.4, 0.6], [1.0, 0.0, 0.5]], dtype=np.float32)
    torch.save(
        {
            "means": torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]),
            "covariances": torch.eye(3).repeat(2, 1, 1),
            "sh_dc": torch.from_numpy(rgb_to_sh_dc(rgbs)),
            "opacities": torch.tensor([0.1, 0.8]),
            "metadata": {"stage": "test"},
        },
        path,
    )

    splats = load_gaussian_splats(path)

    assert splats["centers"].shape == (2, 3)
    assert splats["covariances"].shape == (2, 3, 3)
    assert splats["opacities"].shape == (2, 1)
    assert np.allclose(splats["rgbs"], rgbs)
