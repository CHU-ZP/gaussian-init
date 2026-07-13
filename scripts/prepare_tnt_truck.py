from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType

SCENE = "Truck"
LICENSE_URL = "https://www.tanksandtemples.org/license/"
DOWNLOAD_PAGE_URL = "https://www.tanksandtemples.org/download/"
OFFICIAL_DOWNLOADER_URL = (
    "https://raw.githubusercontent.com/IntelVCL/TanksAndTemples/"
    "2a0d1b25df9352003274fbe0979a17064d768d13/"
    "python_toolbox/download_t2_dataset.py"
)
OFFICIAL_DOWNLOADER_SHA256 = "369a01ee3cd016e29cc4bf31558126aabb78d401541a1747e333cb445e30bcef"
OFFICIAL_IMAGE_CHECKSUM_URL = (
    "https://storage.googleapis.com/t2-downloads/image_sets/image_sets_md5.chk"
)
TRUCK_RESOURCE_KEY = "0-uYzL1Ga_EW1Ck0o-msT7Sg"
TRUCK_ARCHIVE_MD5 = "0ceab344da71b7173d26c972b7c37773"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official Tanks and Temples Truck image set and prepare "
            "a scene directory for this repository."
        )
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help=f"Confirm that you have read and accept {LICENSE_URL}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "tnt_truck",
        help="Prepared scene root (default: data/tnt_truck).",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "downloads" / "tanks_and_temples",
        help="Cache for the official downloader and Truck.zip.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=12,
        help="Number of uniformly spaced frames to prepare; 0 keeps every frame.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy selected images instead of creating relative symbolic links.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace prepared images and remove stale vggt/init outputs.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Delete and re-download Truck.zip if its MD5 does not match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_license:
        raise SystemExit(
            "Refusing to download until --accept-license is provided. "
            f"Read the dataset terms at {LICENSE_URL}"
        )
    if args.num_images < 0:
        raise SystemExit("--num-images must be non-negative")

    download_root = args.download_root.resolve()
    output_root = args.output.resolve()
    downloader_path = download_root / "tools" / "download_t2_dataset.py"
    fetch_official_downloader(downloader_path)
    downloader = load_official_downloader(downloader_path)

    image_sets_root = download_root / "image_sets"
    image_sets_root.mkdir(parents=True, exist_ok=True)
    checksums = load_official_image_checksums(image_sets_root)
    archive_path = download_official_truck_archive(
        downloader,
        download_root=download_root,
        expected_md5=checksums.get(SCENE, TRUCK_ARCHIVE_MD5),
        force_download=args.force_download,
    )

    extracted_root = download_root / "extracted" / SCENE
    source_images = extract_archive_images(archive_path, extracted_root)
    selected_images = select_evenly(source_images, args.num_images)
    materialize_dataset(
        selected_images,
        output_root=output_root,
        copy_images=args.copy,
        force=args.force,
        archive_path=archive_path,
        total_archive_images=len(source_images),
    )
    print_next_steps(output_root, len(selected_images))


def fetch_official_downloader(path: Path) -> None:
    if path.exists():
        verify_sha256(path, OFFICIAL_DOWNLOADER_SHA256)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        OFFICIAL_DOWNLOADER_URL,
        headers={"User-Agent": "vggt-pca-gsplat-dataset-preparer/1.0"},
    )
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != OFFICIAL_DOWNLOADER_SHA256:
        raise RuntimeError(
            "Official downloader SHA256 mismatch before execution: "
            f"expected {OFFICIAL_DOWNLOADER_SHA256}, got {digest}"
        )
    path.write_bytes(payload)


def load_official_downloader(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("tanks_and_temples_official_downloader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official downloader: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_official_image_checksums(image_sets_root: Path) -> dict[str, str]:
    checksum_path = image_sets_root / "image_sets_md5.chk"
    if not checksum_path.exists() or checksum_path.read_bytes().lstrip().startswith(b"<"):
        request = urllib.request.Request(
            OFFICIAL_IMAGE_CHECKSUM_URL,
            headers={"User-Agent": "vggt-pca-gsplat-dataset-preparer/1.0"},
        )
        with urllib.request.urlopen(request) as response:
            checksum_path.write_bytes(response.read())
    checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) < 2:
            continue
        filename = Path(fields[-1]).name
        checksums[Path(filename).stem] = fields[0]
    return checksums


def download_official_truck_archive(
    downloader: ModuleType,
    *,
    download_root: Path,
    expected_md5: str | None,
    force_download: bool,
) -> Path:
    archive_path = download_root / "image_sets" / f"{SCENE}.zip"
    if (
        archive_path.exists()
        and expected_md5 is not None
        and file_md5(archive_path) != expected_md5
    ):
        if not force_download:
            raise RuntimeError(
                f"Existing archive failed MD5 verification: {archive_path}. "
                "Use --force-download to replace it."
            )
        archive_path.unlink()

    if not archive_path.exists():
        downloader.unpack = False
        downloader.download_file_from_google_drive = download_google_drive_file
        prefix = str(download_root) + os.sep
        downloader.download_image_sets(
            prefix,
            SCENE,
            {SCENE: expected_md5 or ""},
            expected_md5 is not None,
        )

    if not archive_path.exists():
        raise RuntimeError(f"Official downloader did not create {archive_path}")
    actual_md5 = file_md5(archive_path)
    if expected_md5 is not None and actual_md5 != expected_md5:
        raise RuntimeError(
            f"Downloaded Truck.zip failed MD5 verification: expected {expected_md5}, "
            f"got {actual_md5}"
        )
    return archive_path


def download_google_drive_file(file_id: str, destination: str) -> None:
    """Compatibility replacement for the official downloader's legacy Drive request."""
    import requests

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".part")
    params = {
        "id": file_id,
        "export": "download",
        "confirm": "t",
    }
    if file_id == "0B-ePgl6HF260NEw3OGN4ckF0dnM":
        params["resourcekey"] = TRUCK_RESOURCE_KEY

    try:
        with requests.get(
            "https://drive.usercontent.google.com/download",
            params=params,
            stream=True,
            timeout=(30, 120),
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise RuntimeError("Google Drive returned HTML instead of the requested archive")
            expected_size = int(response.headers.get("content-length", "0"))
            downloaded = 0
            with temporary_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    print(f"\r{downloaded / 1_000_000:7.1f} MB downloaded", end="", flush=True)
            print()
            if expected_size and downloaded != expected_size:
                raise RuntimeError(
                    f"Incomplete Google Drive download: expected {expected_size} bytes, "
                    f"received {downloaded}"
                )
        temporary_path.replace(destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def extract_archive_images(archive_path: Path, extraction_root: Path) -> list[Path]:
    extraction_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        corrupted_member = archive.testzip()
        if corrupted_member is not None:
            raise RuntimeError(f"CRC verification failed for {corrupted_member} in {archive_path}")
        image_members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and Path(member.filename).suffix.lower() in IMAGE_SUFFIXES
        ]
        if not image_members:
            raise RuntimeError(f"No supported images found in {archive_path}")
        for member in image_members:
            relative = safe_archive_path(member.filename)
            target = extraction_root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != member.file_size:
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            extracted.append(target)
    return sorted(extracted, key=natural_sort_key)


def safe_archive_path(filename: str) -> PurePosixPath:
    relative = PurePosixPath(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe path in Truck.zip: {filename}")
    meaningful_parts = tuple(part for part in relative.parts if part not in {"", "."})
    if not meaningful_parts:
        raise RuntimeError(f"Invalid path in Truck.zip: {filename}")
    return PurePosixPath(*meaningful_parts)


def select_evenly(images: list[Path], count: int) -> list[Path]:
    if not images:
        raise ValueError("Cannot select from an empty image list")
    if count == 0 or count >= len(images):
        return list(images)
    if count == 1:
        return [images[len(images) // 2]]
    indices = [round(index * (len(images) - 1) / (count - 1)) for index in range(count)]
    return [images[index] for index in indices]


def materialize_dataset(
    images: list[Path],
    *,
    output_root: Path,
    copy_images: bool,
    force: bool,
    archive_path: Path,
    total_archive_images: int,
) -> None:
    images_dir = output_root / "images"
    derived_paths = [output_root / "vggt", output_root / "init"]
    existing_paths = [path for path in [images_dir, *derived_paths] if path.exists()]
    if existing_paths:
        if not force:
            raise RuntimeError(
                "Prepared or derived scene data already exists: "
                f"{', '.join(str(path) for path in existing_paths)}. "
                "Use --force to rebuild the dataset and invalidate derived outputs."
            )
        for path in existing_paths:
            if path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_images: list[dict[str, str]] = []
    for index, source in enumerate(images):
        destination = images_dir / f"{index:06d}{source.suffix.lower()}"
        if copy_images:
            shutil.copy2(source, destination)
        else:
            relative_source = os.path.relpath(source, start=destination.parent)
            destination.symlink_to(relative_source)
        manifest_images.append(
            {
                "prepared": str(destination.relative_to(output_root)),
                "source": str(source),
            }
        )

    manifest = {
        "dataset": "Tanks and Temples",
        "scene": SCENE,
        "download_page": DOWNLOAD_PAGE_URL,
        "license": LICENSE_URL,
        "official_downloader": OFFICIAL_DOWNLOADER_URL,
        "archive": str(archive_path),
        "archive_md5": file_md5(archive_path),
        "archive_image_count": total_archive_images,
        "selected_image_count": len(images),
        "materialization": "copy" if copy_images else "relative_symlink",
        "images": manifest_images,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_next_steps(output_root: Path, image_count: int) -> None:
    display_root = os.path.relpath(output_root, start=REPOSITORY_ROOT)
    shell_root = shlex.quote(display_root)
    print(f"Prepared {image_count} uniformly spaced Truck images in {display_root}/images")
    print("\nRun VGGT:")
    print(
        "uv run python -m preprocess.run_vggt "
        f"--images {shell_root}/images "
        f"--output {shell_root}/vggt/predictions.npz "
        "--device cuda --preprocess-mode crop --max-resolution 336 "
        "--head-frames-chunk-size 1"
    )
    print("\nBuild Gaussian initialization:")
    print(
        "uv run python -m init.build_init "
        "--config configs/log_ellipse.yaml "
        f"--scene-root {shell_root}"
    )


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"Cached official downloader SHA256 mismatch: expected {expected}, got {digest}. "
            f"Delete {path} and retry."
        )


def file_md5(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def natural_sort_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", str(path))
    )


if __name__ == "__main__":
    main()
