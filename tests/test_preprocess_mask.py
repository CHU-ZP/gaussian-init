from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from preprocess.run_vggt import build_preprocessed_valid_masks, resize_valid_masks


def test_pad_preprocessing_tracks_content_mask(tmp_path: Path) -> None:
    path = tmp_path / "wide.png"
    Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8)).save(path)
    mask = build_preprocessed_valid_masks(
        [path],
        mode="pad",
        expected_shape=(518, 518),
    )
    expected_height = round(50 * (518 / 100) / 14) * 14
    assert mask.shape == (1, 518, 518)
    assert int(mask.sum()) == expected_height * 518
    assert not mask[0, 0, 0]
    assert mask[0, 259, 259]

    resized = resize_valid_masks(mask, target_shape=(252, 252))
    assert resized.shape == (1, 252, 252)
    assert not resized[0, 0, 0]
    assert resized[0, 126, 126]
