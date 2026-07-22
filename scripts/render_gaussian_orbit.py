from __future__ import annotations

import argparse
import logging
import math
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import viser
from viser import transforms as tf

if __package__:
    from scripts.view_gaussians import load_gaussian_splats
else:
    from view_gaussians import load_gaussian_splats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a turntable animation of initialized Gaussians with Viser."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input Gaussian .pt file.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional predictions.npz used to recover a scene up direction and starting view.",
    )
    parser.add_argument(
        "--focus-box",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="Normalized image box used to retain a foreground subject for visualization.",
    )
    parser.add_argument(
        "--look-at-box",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help=(
            "Normalized image box used only to estimate the orbit center. "
            "Unlike --focus-box, it does not discard any Gaussians."
        ),
    )
    parser.add_argument(
        "--focus-view",
        type=int,
        default=0,
        help="Prediction view used by --focus-box.",
    )
    parser.add_argument(
        "--focus-depth-percentile",
        type=float,
        default=75.0,
        help="Depth percentile inside --focus-box used to estimate foreground extent.",
    )
    parser.add_argument(
        "--focus-depth-scale",
        type=float,
        default=1.7,
        help="Multiplier applied to the focus-box foreground depth.",
    )
    parser.add_argument(
        "--trim-lowest-up-percent",
        type=float,
        default=0.0,
        help="Display-only removal of the lowest subject points along the camera-derived up axis.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Output height in pixels.")
    parser.add_argument("--frames", type=int, default=120, help="Frames in one full orbit.")
    parser.add_argument("--fps", type=int, default=15, help="Output frame rate.")
    parser.add_argument(
        "--elevation-degrees",
        type=float,
        default=16.0,
        help="Camera elevation above the orbit plane.",
    )
    parser.add_argument(
        "--fov-degrees",
        type=float,
        default=48.0,
        help="Vertical field of view.",
    )
    parser.add_argument(
        "--framing-percentile",
        type=float,
        default=90.0,
        help="Percentile of center distances used to frame the scene.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.12,
        help="Multiplicative camera-distance padding.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Explicit orbit radius. By default it is estimated from the scene.",
    )
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop-radius",
        type=float,
        default=None,
        help="Display only Gaussians within this distance of the robust scene center.",
    )
    crop_group.add_argument(
        "--crop-percentile",
        type=float,
        default=None,
        help="Display the nearest percentage of Gaussians around the robust scene center.",
    )
    parser.add_argument(
        "--opacity-scale",
        type=float,
        default=1.0,
        help="Display-only opacity multiplier.",
    )
    parser.add_argument(
        "--gaussian-scale",
        type=float,
        default=1.0,
        help="Display-only multiplier for all Gaussian covariance scales.",
    )
    parser.add_argument(
        "--background",
        default="#111820",
        help="Background color as #RRGGBB.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Viser port.")
    parser.add_argument(
        "--browser",
        type=Path,
        default=None,
        help="Chrome/Chromium executable. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--no-launch-browser",
        action="store_true",
        help="Wait for a manually opened browser instead of launching a headless client.",
    )
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for a Viser browser client.",
    )
    parser.add_argument(
        "--scene-load-seconds",
        type=float,
        default=2.0,
        help="Delay after the browser connects before capturing frames.",
    )
    parser.add_argument(
        "--render-timeout",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for each browser-rendered frame.",
    )
    parser.add_argument(
        "--show-browser-log",
        action="store_true",
        help="Print Chrome/Chromium diagnostics while rendering.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must use the .mp4 extension")
    if args.width < 2 or args.height < 2:
        raise ValueError("--width and --height must be at least 2")
    if args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must be even for yuv420p output")
    if args.frames < 2 or args.fps < 1:
        raise ValueError("--frames must be at least 2 and --fps must be positive")
    if not 0.0 < args.fov_degrees < 150.0:
        raise ValueError("--fov-degrees must lie in (0, 150)")
    if not 0.0 < args.framing_percentile <= 100.0:
        raise ValueError("--framing-percentile must lie in (0, 100]")
    if not 0.0 < args.focus_depth_percentile <= 100.0:
        raise ValueError("--focus-depth-percentile must lie in (0, 100]")
    if args.focus_view < 0 or args.focus_depth_scale <= 0.0:
        raise ValueError("focus view/depth arguments are invalid")
    if not 0.0 <= args.trim_lowest_up_percent < 100.0:
        raise ValueError("--trim-lowest-up-percent must lie in [0, 100)")
    if args.trim_lowest_up_percent > 0.0 and args.focus_box is None:
        raise ValueError("--trim-lowest-up-percent requires --focus-box")
    for argument, box in (("--focus-box", args.focus_box), ("--look-at-box", args.look_at_box)):
        if box is None:
            continue
        x0, y0, x1, y1 = box
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"{argument} must satisfy 0 <= X0 < X1 <= 1 and 0 <= Y0 < Y1 <= 1")
    if args.padding <= 0.0 or args.opacity_scale <= 0.0 or args.gaussian_scale <= 0.0:
        raise ValueError("--padding, --opacity-scale, and --gaussian-scale must be positive")
    if args.radius is not None and args.radius <= 0.0:
        raise ValueError("--radius must be positive")
    if args.crop_radius is not None and args.crop_radius <= 0.0:
        raise ValueError("--crop-radius must be positive")
    if args.crop_percentile is not None and not 0.0 < args.crop_percentile <= 100.0:
        raise ValueError("--crop-percentile must lie in (0, 100]")
    if args.client_timeout <= 0.0 or args.scene_load_seconds < 0.0 or args.render_timeout <= 0.0:
        raise ValueError("client timing values are invalid")


def parse_hex_color(value: str) -> np.ndarray:
    text = value.strip()
    if len(text) != 7 or not text.startswith("#"):
        raise ValueError("--background must have the form #RRGGBB")
    try:
        channels = [int(text[index : index + 2], 16) for index in (1, 3, 5)]
    except ValueError as exc:
        raise ValueError("--background must have the form #RRGGBB") from exc
    return np.asarray(channels, dtype=np.uint8)


def normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1.0e-8:
        raise ValueError(f"Cannot normalize degenerate {name}")
    return np.asarray(vector, dtype=np.float64) / length


def find_predictions(input_path: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    if input_path.parent.name == "init":
        candidate = input_path.parent.parent / "vggt" / "predictions.npz"
        if candidate.is_file():
            return candidate
    return None


def load_camera_reference(
    path: Path | None,
    *,
    scene_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fallback_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    fallback_radial = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if path is None:
        return fallback_up, fallback_radial
    if not path.is_file():
        raise FileNotFoundError(f"Camera archive does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        if "extrinsics_c2w" in archive:
            c2w = np.asarray(archive["extrinsics_c2w"], dtype=np.float64)
        elif "extrinsics" in archive:
            c2w = np.asarray(archive["extrinsics"], dtype=np.float64)
        elif "extrinsics_w2c" in archive:
            c2w = np.linalg.inv(np.asarray(archive["extrinsics_w2c"], dtype=np.float64))
        else:
            raise ValueError(f"Camera archive has no extrinsics: {path}")
    if c2w.ndim != 3 or c2w.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Unexpected camera shape {c2w.shape} in {path}")
    if not np.isfinite(c2w).all():
        raise ValueError(f"Camera archive contains non-finite values: {path}")

    rotations = c2w[:, :3, :3]
    positions = c2w[:, :3, 3]
    up = normalize(np.mean(-rotations[:, :, 1], axis=0), name="mean camera up")
    radial = positions[0] - scene_center
    radial = radial - float(radial @ up) * up
    if np.linalg.norm(radial) <= 1.0e-8:
        radial = rotations[0, :, 2]
        radial = radial - float(radial @ up) * up
    radial = normalize(radial, name="starting radial direction")
    return up, radial


def focus_gaussians(
    splats: dict[str, np.ndarray],
    *,
    predictions_path: Path,
    focus_box: tuple[float, float, float, float] | list[float],
    view_index: int,
    depth_percentile: float,
    depth_scale: float,
    trim_lowest_up_percent: float,
) -> tuple[dict[str, np.ndarray], float]:
    with np.load(predictions_path, allow_pickle=False) as archive:
        if "intrinsics" not in archive or "depth" not in archive:
            raise ValueError("--focus-box requires intrinsics and depth in predictions.npz")
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
        depth = np.asarray(archive["depth"], dtype=np.float64)
        if "extrinsics_w2c" in archive:
            w2c = np.asarray(archive["extrinsics_w2c"], dtype=np.float64)
        elif "extrinsics_c2w" in archive:
            w2c = np.linalg.inv(np.asarray(archive["extrinsics_c2w"], dtype=np.float64))
        elif "extrinsics" in archive:
            w2c = np.linalg.inv(np.asarray(archive["extrinsics"], dtype=np.float64))
        else:
            raise ValueError("--focus-box requires camera extrinsics in predictions.npz")

    views = intrinsics.shape[0]
    if view_index >= views:
        raise ValueError(f"--focus-view {view_index} is outside the available {views} views")
    if depth.shape[0] != views or w2c.shape[0] != views:
        raise ValueError("Camera, depth, and intrinsics view counts do not match")

    height, width = depth.shape[1:3]
    x0, y0, x1, y1 = (float(value) for value in focus_box)
    pixel_x0 = max(0, min(width - 1, int(math.floor(x0 * width))))
    pixel_y0 = max(0, min(height - 1, int(math.floor(y0 * height))))
    pixel_x1 = max(pixel_x0 + 1, min(width, int(math.ceil(x1 * width))))
    pixel_y1 = max(pixel_y0 + 1, min(height, int(math.ceil(y1 * height))))
    box_depth = depth[view_index, pixel_y0:pixel_y1, pixel_x0:pixel_x1]
    valid_depth = box_depth[np.isfinite(box_depth) & (box_depth > 0.0)]
    if valid_depth.size == 0:
        raise ValueError("The focus box contains no finite positive depth values")
    far_depth = float(np.percentile(valid_depth, depth_percentile) * depth_scale)

    centers = np.asarray(splats["centers"], dtype=np.float64)
    homogeneous = np.concatenate(
        [centers, np.ones((centers.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera_points = (w2c[view_index] @ homogeneous.T).T[:, :3]
    projected = (intrinsics[view_index] @ camera_points.T).T
    projected_xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1.0e-12)
    normalized_x = projected_xy[:, 0] / width
    normalized_y = projected_xy[:, 1] / height
    keep = (
        np.isfinite(camera_points).all(axis=1)
        & np.isfinite(projected_xy).all(axis=1)
        & (camera_points[:, 2] > 0.0)
        & (camera_points[:, 2] <= far_depth)
        & (normalized_x >= x0)
        & (normalized_x <= x1)
        & (normalized_y >= y0)
        & (normalized_y <= y1)
    )
    if trim_lowest_up_percent > 0.0:
        c2w = np.linalg.inv(w2c)
        up = normalize(np.mean(-c2w[:, :3, 1], axis=0), name="mean camera up")
        heights = centers @ up
        height_cutoff = float(np.percentile(heights[keep], trim_lowest_up_percent))
        keep &= heights >= height_cutoff
    if np.count_nonzero(keep) < 3:
        raise ValueError("The focus box retained fewer than three Gaussians")
    return {key: value[keep] for key, value in splats.items()}, far_depth


def camera_orientation(
    position: np.ndarray,
    *,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    forward = normalize(target - position, name="camera forward direction")
    projected_up = up - float(up @ forward) * forward
    if np.linalg.norm(projected_up) <= 1.0e-8:
        fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        projected_up = fallback - float(fallback @ forward) * forward
    camera_down = -normalize(projected_up, name="camera up direction")
    camera_right = normalize(np.cross(camera_down, forward), name="camera right direction")
    rotation = np.stack([camera_right, camera_down, forward], axis=1)
    return tf.SO3.from_matrix(rotation).wxyz.astype(np.float64)


def estimate_orbit_radius(
    centers: np.ndarray,
    covariances: np.ndarray,
    *,
    percentile: float,
    vertical_fov: float,
    padding: float,
) -> tuple[float, float]:
    distances = np.linalg.norm(centers, axis=1)
    support_radius = float(np.percentile(distances, percentile))
    eigenvalues = np.linalg.eigvalsh(covariances)
    gaussian_radius = float(np.percentile(np.sqrt(np.maximum(eigenvalues[:, -1], 0.0)), percentile))
    extent = support_radius + 2.0 * gaussian_radius
    if not np.isfinite(extent) or extent <= 1.0e-8:
        raise ValueError("Cannot estimate a finite positive scene extent")
    orbit_radius = padding * extent / math.sin(vertical_fov * 0.5)
    return float(orbit_radius), float(extent)


def find_browser(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Browser executable does not exist: {explicit_path}")
        return explicit_path
    for name in ("google-chrome", "chromium", "chromium-browser"):
        resolved = shutil.which(name)
        if resolved is not None:
            return Path(resolved)
    raise RuntimeError(
        "Could not find Chrome or Chromium. Pass --browser or use --no-launch-browser."
    )


def start_encoder(output: Path, *, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write the MP4 animation")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def finish_encoder(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is not None:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr.strip()}")


def request_render(
    client: viser.ClientHandle,
    *,
    height: int,
    width: int,
    wxyz: np.ndarray,
    position: np.ndarray,
    fov: float,
    timeout: float,
) -> np.ndarray:
    results: queue.Queue[np.ndarray | BaseException] = queue.Queue(maxsize=1)

    def _render() -> None:
        try:
            results.put(
                client.get_render(
                    height,
                    width,
                    wxyz=wxyz,
                    position=position,
                    fov=fov,
                    transport_format="jpeg",
                )
            )
        except BaseException as exc:
            results.put(exc)

    threading.Thread(target=_render, daemon=True).start()
    try:
        result = results.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError(f"Viser did not return a frame within {timeout:g} seconds") from exc
    if isinstance(result, BaseException):
        raise result
    return result


def composite_render_background(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected rendered image shape: {image.shape}")
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(
            image[:, :, :3].astype(np.float32) * alpha
            + background.reshape(1, 1, 3).astype(np.float32) * (1.0 - alpha),
            0.0,
            255.0,
        ).astype(np.uint8)

    rgb = np.asarray(image[:, :, :3], dtype=np.uint8).copy()
    if np.all(background == 255):
        return rgb

    import cv2

    near_white = np.min(rgb, axis=2) >= 238
    _, labels = cv2.connectedComponents(near_white.astype(np.uint8), connectivity=4)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border_labels = border_labels[border_labels != 0]
    if border_labels.size > 0:
        rgb[np.isin(labels, border_labels)] = background
    return rgb


def main() -> None:
    args = parse_args()
    splats = load_gaussian_splats(args.input)
    if splats["centers"].shape[0] == 0:
        raise ValueError("The input contains no Gaussians")

    predictions_path = find_predictions(args.input, args.predictions)
    look_at_splats: dict[str, np.ndarray] | None = None
    if args.look_at_box is not None:
        if predictions_path is None:
            raise ValueError(
                "--look-at-box requires --predictions or an auto-detected predictions.npz"
            )
        look_at_splats, look_at_far_depth = focus_gaussians(
            splats,
            predictions_path=predictions_path,
            focus_box=args.look_at_box,
            view_index=args.focus_view,
            depth_percentile=args.focus_depth_percentile,
            depth_scale=args.focus_depth_scale,
            trim_lowest_up_percent=0.0,
        )
        print(
            f"Look-at box estimated the subject center from "
            f"{look_at_splats['centers'].shape[0]:,} Gaussians "
            f"with camera-space depth <= {look_at_far_depth:.4g}; "
            f"all {splats['centers'].shape[0]:,} Gaussians remain visible"
        )
    if args.focus_box is not None:
        if predictions_path is None:
            raise ValueError(
                "--focus-box requires --predictions or an auto-detected predictions.npz"
            )
        original_count = int(splats["centers"].shape[0])
        splats, focus_far_depth = focus_gaussians(
            splats,
            predictions_path=predictions_path,
            focus_box=args.focus_box,
            view_index=args.focus_view,
            depth_percentile=args.focus_depth_percentile,
            depth_scale=args.focus_depth_scale,
            trim_lowest_up_percent=args.trim_lowest_up_percent,
        )
        print(
            f"Focus box retained {splats['centers'].shape[0]:,}/{original_count:,} Gaussians "
            f"with camera-space depth <= {focus_far_depth:.4g}"
        )

    center_source = splats if look_at_splats is None else look_at_splats
    scene_center = np.median(center_source["centers"], axis=0).astype(np.float64)
    centers = (splats["centers"] - scene_center).astype(np.float32)
    selection = np.ones(centers.shape[0], dtype=bool)
    center_distances = np.linalg.norm(centers, axis=1)
    crop_radius = args.crop_radius
    if args.crop_percentile is not None:
        crop_radius = float(np.percentile(center_distances, args.crop_percentile))
    if crop_radius is not None:
        selection = center_distances <= crop_radius
        if np.count_nonzero(selection) < 3:
            raise ValueError("The requested crop retains fewer than three Gaussians")
        centers = centers[selection]

    covariances = splats["covariances"][selection]
    rgbs = splats["rgbs"][selection]
    opacities = np.clip(splats["opacities"][selection] * args.opacity_scale, 0.0, 1.0)
    up, starting_radial = load_camera_reference(
        predictions_path,
        scene_center=scene_center,
    )
    tangent = normalize(np.cross(up, starting_radial), name="orbit tangent")

    fov = math.radians(args.fov_degrees)
    estimated_radius, scene_extent = estimate_orbit_radius(
        centers,
        covariances,
        percentile=args.framing_percentile,
        vertical_fov=fov,
        padding=args.padding,
    )
    orbit_radius = estimated_radius if args.radius is None else float(args.radius)
    elevation = math.radians(args.elevation_degrees)

    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction(up)
    background = parse_hex_color(args.background)
    server.scene.add_gaussian_splats(
        "/gaussians",
        centers=centers,
        covariances=covariances,
        rgbs=rgbs,
        opacities=opacities,
        scale=args.gaussian_scale,
    )

    initial_position = orbit_radius * (
        math.cos(elevation) * starting_radial + math.sin(elevation) * up
    )
    server.initial_camera.position = initial_position
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.up = up

    clients: queue.Queue[viser.ClientHandle] = queue.Queue()

    @server.on_client_connect
    def _on_client_connect(client: viser.ClientHandle) -> None:
        clients.put(client)

    browser_process: subprocess.Popen[bytes] | None = None
    temporary_profile: tempfile.TemporaryDirectory[str] | None = None
    try:
        url_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        url = f"http://{url_host}:{server.get_port()}"
        if args.no_launch_browser:
            print(f"Open {url} in a browser to begin capture")
        else:
            browser = find_browser(args.browser)
            temporary_profile = tempfile.TemporaryDirectory(prefix="gaussian-orbit-chrome-")
            browser_command = [
                str(browser),
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--hide-scrollbars",
                "--ignore-gpu-blocklist",
                "--enable-unsafe-swiftshader",
                f"--window-size={args.width},{args.height + 120}",
                f"--user-data-dir={temporary_profile.name}",
                f"--app={url}",
            ]
            browser_process = subprocess.Popen(
                browser_command,
                stdout=subprocess.DEVNULL,
                stderr=None if args.show_browser_log else subprocess.DEVNULL,
            )

        try:
            client = clients.get(timeout=args.client_timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                f"No Viser client connected within {args.client_timeout:g} seconds"
            ) from exc
        print(f"Connected Viser client {client.client_id}; waiting for scene upload")
        time.sleep(args.scene_load_seconds)

        print(
            f"Rendering {args.frames} frames at {args.width}x{args.height}, "
            f"radius={orbit_radius:.4g}, extent(p{args.framing_percentile:g})={scene_extent:.4g}"
        )
        if crop_radius is not None:
            print(
                f"Focus crop: radius={crop_radius:.4g}, "
                f"retained={centers.shape[0]}/{selection.shape[0]} Gaussians"
            )
        if predictions_path is not None:
            print(f"Camera reference: {predictions_path}")

        encoder = start_encoder(
            args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        try:
            assert encoder.stdin is not None
            target = np.zeros(3, dtype=np.float64)
            for frame_index in range(args.frames):
                angle = 2.0 * math.pi * frame_index / args.frames
                radial = math.cos(angle) * starting_radial + math.sin(angle) * tangent
                position = orbit_radius * (math.cos(elevation) * radial + math.sin(elevation) * up)
                wxyz = camera_orientation(position, target=target, up=up)
                image = request_render(
                    client,
                    height=args.height,
                    width=args.width,
                    wxyz=wxyz,
                    position=position,
                    fov=fov,
                    timeout=args.render_timeout,
                )
                if image.shape not in (
                    (args.height, args.width, 3),
                    (args.height, args.width, 4),
                ):
                    raise RuntimeError(
                        f"Unexpected Viser frame shape {image.shape}; "
                        f"expected RGB or RGBA at {(args.height, args.width)}"
                    )
                composited = composite_render_background(image, background)
                encoder.stdin.write(np.ascontiguousarray(composited, dtype=np.uint8).tobytes())
                if frame_index == 0 or (frame_index + 1) % 10 == 0:
                    print(f"  frame {frame_index + 1:04d}/{args.frames:04d}")
        except Exception:
            encoder.kill()
            encoder.wait()
            raise
        else:
            finish_encoder(encoder)
    finally:
        if browser_process is not None:
            browser_process.terminate()
            try:
                browser_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                browser_process.kill()
                browser_process.wait()
        if temporary_profile is not None:
            temporary_profile.cleanup()
        server.stop()

    print(f"Wrote orbit animation: {args.output.resolve()}")


if __name__ == "__main__":
    main()
