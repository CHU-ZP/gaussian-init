from __future__ import annotations

import numpy as np
import pytest
import torch

from gsplat_train.dataset import load_scene_data
from init.build_init import validate_prediction_precision_contract
from init.io import load_dense_predictions
from preprocess.run_vggt import run_prediction_heads, validate_camera_parameters
from scripts.view_vggt import load_vggt_view_data, normalize_confidence


class _FakeVGGT:
    def __init__(self) -> None:
        self.depth_chunk_size: int | None = None
        self.point_chunk_size: int | None = None

    def camera_head(self, _tokens):
        return [torch.ones((1, 2, 9), dtype=torch.float32)]

    def depth_head(self, _tokens, images, _patch_start_idx, *, frames_chunk_size: int):
        self.depth_chunk_size = frames_chunk_size
        shape = (*images.shape[:2], *images.shape[-2:], 1)
        return torch.ones(shape), torch.ones(shape[:-1])

    def point_head(self, _tokens, images, _patch_start_idx, *, frames_chunk_size: int):
        self.point_chunk_size = frames_chunk_size
        shape = (*images.shape[:2], *images.shape[-2:], 3)
        return torch.ones(shape), torch.ones(shape[:-1])


def test_prediction_heads_use_requested_frame_chunks() -> None:
    model = _FakeVGGT()
    outputs = run_prediction_heads(
        model,
        [torch.ones((1, 2, 4))],
        torch.ones((1, 2, 3, 14, 14)),
        0,
        device=torch.device("cpu"),
        use_point_map=True,
        frames_chunk_size=1,
    )

    assert model.depth_chunk_size == 1
    assert model.point_chunk_size == 1
    assert all(output is not None for output in outputs)


def test_camera_validation_rejects_non_rotation() -> None:
    extrinsics = np.zeros((1, 3, 4), dtype=np.float32)
    extrinsics[0, :3, :3] = np.eye(3, dtype=np.float32)
    intrinsics = np.asarray([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]])
    validate_camera_parameters(extrinsics, intrinsics)

    extrinsics[0, 1, 1] = 0.98
    with pytest.raises(ValueError, match="must run in float32"):
        validate_camera_parameters(extrinsics, intrinsics)


def test_dense_predictions_do_not_require_confidence(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        world_points=np.zeros((1, 2, 3, 3), dtype=np.float32),
    )

    predictions = load_dense_predictions(predictions_path)

    assert "confidence" not in predictions


def test_unbounded_vggt_confidence_normalizes_for_visualization() -> None:
    normalized = normalize_confidence(np.asarray([1.0, 1.2, 1.6], dtype=np.float32))
    assert normalized[0] == 0.0
    assert normalized[-1] == 1.0
    assert np.all(np.diff(normalized) > 0.0)


def test_incompatible_runner_predictions_are_rejected() -> None:
    runner_output = {
        "model_id": np.asarray("facebook/VGGT-1B"),
        "world_points_source": np.asarray("depth_unprojection"),
        "processed_images": np.zeros((1, 2, 2, 3), dtype=np.float32),
    }
    with pytest.raises(RuntimeError, match="precision contract is incompatible"):
        validate_prediction_precision_contract(runner_output)

    runner_output["precision_contract"] = np.asarray("vggt_aggregator_amp_heads_float32_v1")
    runner_output["head_dtype"] = np.asarray("float32")
    validate_prediction_precision_contract(runner_output)


def test_scene_data_exposes_both_camera_directions(tmp_path) -> None:
    scene_root = tmp_path / "scene"
    (scene_root / "vggt").mkdir(parents=True)
    c2w = np.eye(4, dtype=np.float32)[None]
    c2w[0, 0, 3] = 2.0
    np.savez_compressed(
        scene_root / "vggt" / "predictions.npz",
        world_points=np.zeros((1, 2, 2, 3), dtype=np.float32),
        confidence=np.ones((1, 2, 2), dtype=np.float32),
        processed_images=np.zeros((1, 2, 2, 3), dtype=np.float32),
        processed_valid_mask=np.ones((1, 2, 2), dtype=bool),
        intrinsics=np.eye(3, dtype=np.float32)[None],
        extrinsics=c2w,
    )

    scene = load_scene_data(
        {
            "scene": {
                "root": str(scene_root),
                "predictions_path": "vggt/predictions.npz",
            }
        }
    )

    assert scene.extrinsics_c2w is not None
    assert scene.extrinsics_w2c is not None
    assert scene.extrinsics_c2w[0, 0, 3] == 2.0
    assert scene.extrinsics_w2c[0, 0, 3] == -2.0


def test_vggt_viewer_uses_rgb_valid_mask_and_confidence_percentile(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.npz"
    points = np.zeros((1, 2, 3, 3), dtype=np.float32)
    points[0, ..., 0] = np.arange(6, dtype=np.float32).reshape(2, 3)
    confidence = np.arange(1, 7, dtype=np.float32).reshape(1, 2, 3)
    images = np.zeros((1, 2, 3, 3), dtype=np.float32)
    images[0, ..., 0] = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
    valid_mask = np.ones((1, 2, 3), dtype=bool)
    valid_mask[0, 1, 2] = False
    np.savez_compressed(
        predictions_path,
        world_points=points,
        confidence=confidence,
        processed_images=images,
        processed_valid_mask=valid_mask,
    )

    view_data = load_vggt_view_data(
        predictions_path,
        confidence_percentile=50.0,
        stride=1,
        max_points=None,
    )

    assert view_data["confidence_threshold"] == 3.0
    assert np.asarray(view_data["points"]).shape == (3, 3)
    assert np.asarray(view_data["colors"])[:, 0].tolist() == [51, 76, 102]
