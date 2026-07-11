from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.prepare_tnt_truck import (
    extract_archive_images,
    materialize_dataset,
    safe_archive_path,
    select_evenly,
)


def test_extract_select_and_materialize_truck_images(tmp_path: Path) -> None:
    archive_path = tmp_path / "Truck.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in ("Truck/10.jpg", "Truck/2.jpg", "Truck/1.jpg", "Truck/20.jpg"):
            archive.writestr(name, name.encode())
        archive.writestr("Truck/readme.txt", b"metadata")

    extracted = extract_archive_images(archive_path, tmp_path / "extracted")
    assert [path.name for path in extracted] == ["1.jpg", "2.jpg", "10.jpg", "20.jpg"]
    selected = select_evenly(extracted, 3)
    assert [path.name for path in selected] == ["1.jpg", "10.jpg", "20.jpg"]

    output_root = tmp_path / "scene"
    materialize_dataset(
        selected,
        output_root=output_root,
        copy_images=True,
        force=False,
        archive_path=archive_path,
        total_archive_images=len(extracted),
    )
    assert [path.name for path in sorted((output_root / "images").iterdir())] == [
        "000000.jpg",
        "000001.jpg",
        "000002.jpg",
    ]
    manifest = json.loads((output_root / "dataset_manifest.json").read_text())
    assert manifest["scene"] == "Truck"
    assert manifest["archive_image_count"] == 4
    assert manifest["selected_image_count"] == 3
    assert manifest["materialization"] == "copy"

    (output_root / "vggt").mkdir()
    (output_root / "vggt" / "stale.npz").write_bytes(b"stale")
    materialize_dataset(
        selected[:2],
        output_root=output_root,
        copy_images=True,
        force=True,
        archive_path=archive_path,
        total_archive_images=len(extracted),
    )
    assert not (output_root / "vggt").exists()
    assert len(list((output_root / "images").iterdir())) == 2


def test_archive_path_rejects_traversal() -> None:
    with pytest.raises(RuntimeError, match="Unsafe path"):
        safe_archive_path("../escape.jpg")
