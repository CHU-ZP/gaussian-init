from __future__ import annotations

from pathlib import Path

import numpy as np

from gsplat_train.dataset import load_scene_data
from init.build_colmap_init import extract_reconstruction_arrays, normalize_to_camera_gauge
from init.knn import compute_knn_isotropic_scales
from preprocess.run_colmap import export_aligned_images


class _FakeCamera:
    model_name = "PINHOLE"
    width = 6
    height = 4

    @staticmethod
    def calibration_matrix() -> np.ndarray:
        return np.asarray([[5.0, 0.0, 3.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]])


class _FakePose:
    def __init__(self, tx: float) -> None:
        self.tx = tx

    def matrix(self) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0, 0.0, self.tx], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )


class _FakeImage:
    def __init__(self, name: str, tx: float) -> None:
        self.name = name
        self.camera_id = 1
        self.has_pose = True
        self._pose = _FakePose(tx)

    def cam_from_world(self) -> _FakePose:
        return self._pose


class _FakePoint:
    def __init__(self, xyz: tuple[float, float, float], color: tuple[int, int, int]) -> None:
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.color = np.asarray(color, dtype=np.uint8)


class _FakeReconstruction:
    def __init__(self) -> None:
        self.cameras = {1: _FakeCamera()}
        self.images = {
            8: _FakeImage("view_001.png", -1.0),
            3: _FakeImage("view_000.png", 0.0),
        }
        self.points3D = {
            7: _FakePoint((0.0, 0.0, 0.0), (255, 0, 0)),
            2: _FakePoint((1.0, 0.0, 0.0), (0, 255, 0)),
            9: _FakePoint((0.0, 1.0, 0.0), (0, 0, 255)),
            4: _FakePoint((0.0, 0.0, 1.0), (255, 255, 255)),
        }


def test_aligned_image_export_uses_colmap_mask_naming(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.npz"
    images = np.zeros((2, 4, 6, 3), dtype=np.float32)
    images[0, ..., 0] = 1.0
    mask = np.ones((2, 4, 6), dtype=bool)
    mask[1, 0, 0] = False
    np.savez_compressed(predictions, processed_images=images, processed_valid_mask=mask)

    stats = export_aligned_images(
        predictions,
        images_dir=tmp_path / "images",
        masks_dir=tmp_path / "masks",
    )

    assert stats["image_names"] == ["view_000.png", "view_001.png"]
    assert stats["uses_masks"] is True
    assert (tmp_path / "images" / "view_000.png").exists()
    assert (tmp_path / "masks" / "view_001.png.png").exists()


def test_reconstruction_conversion_preserves_expected_view_order() -> None:
    arrays = extract_reconstruction_arrays(
        _FakeReconstruction(),
        expected_image_names=["view_000.png", "view_001.png"],
        expected_shape=(4, 6),
        require_all_views=True,
    )

    c2w = np.asarray(arrays["extrinsics_c2w"])
    assert c2w.shape == (2, 4, 4)
    assert np.allclose(c2w[:, 0, 3], [0.0, 1.0])
    assert np.allclose(np.asarray(arrays["colors"])[0], [0.0, 1.0, 0.0])
    assert arrays["registered_images"] == 2


def test_colmap_gauge_normalization_matches_target_camera_radius() -> None:
    source_c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    source_c2w[:, 0, 3] = [-2.0, 2.0]
    target_c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    target_c2w[:, 0, 3] = [8.0, 12.0]
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    normalized_points, normalized_c2w, stats = normalize_to_camera_gauge(
        points,
        source_c2w,
        target_c2w,
    )

    assert np.allclose(normalized_points, [[10.0, 0.0, 0.0]])
    assert np.allclose(normalized_c2w[:, 0, 3], [8.0, 12.0])
    assert stats["uniform_scale"] == 1.0


def test_exact_knn_scale_uses_rms_neighbor_distance() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float32,
    )

    scales = compute_knn_isotropic_scales(
        points,
        neighbors=3,
        device="cpu",
        chunk_size=2,
    )

    assert np.isclose(scales[0], np.sqrt((1.0**2 + 2.0**2 + 3.0**2) / 3.0))


def test_camera_only_archive_loads_for_gsplat(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    archive_path = scene_root / "colmap" / "scene.npz"
    archive_path.parent.mkdir(parents=True)
    images = np.zeros((2, 4, 6, 3), dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    intrinsics[:, 0, 0] = 5.0
    intrinsics[:, 1, 1] = 5.0
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    c2w[1, 0, 3] = 1.0
    np.savez_compressed(
        archive_path,
        processed_images=images,
        processed_valid_mask=np.ones((2, 4, 6), dtype=bool),
        intrinsics=intrinsics,
        extrinsics_c2w=c2w,
        reprojection_error_px=np.asarray(0.25, dtype=np.float32),
    )

    scene = load_scene_data(
        {
            "scene": {"root": str(scene_root), "predictions_path": "colmap/scene.npz"},
            "training": {"max_reprojection_error_px": 0.5},
        },
        validate_projection=True,
    )

    assert len(scene) == 2
    assert scene.reprojection_error_px == 0.25
    assert np.allclose(scene.extrinsics_w2c[1, 0, 3], -1.0)
