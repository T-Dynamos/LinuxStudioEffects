"""Example 08: Fastest streaming Video Depth Anything Small — side-by-side Input Video | Depth Map.

Supports:
  - Live webcam input -> v4l2loopback virtual camera (and optional ffplay preview window)
  - Input video file -> optional side-by-side MP4 output and/or ffplay preview window
  - Choice of NPU (`--device npu`) or CPU (`--device cpu`) at fastest ViT-S resolutions
    (default 140x140 for maximum FPS; pass --input-size 196 or 252 for finer spatial detail)
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Iterator

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    CameraConfig,
    ComputeDevice,
    V4L2Camera,
    VirtualCameraConfig,
    VirtualCameraOutput,
    is_device_available,
    pick_default_camera,
    pick_default_loopback,
    resize_rgb,
)
from linux_studio_effects.video.models.video_depth_anything import (
    VideoDepthAnythingModel,
    depth_to_colormap_rgb,
    get_video_depth_config,
    load_from_hf,
)

# Minimal 3x5 bitmap font for subtle corner labels ("IN" / "DEPTH") with zero external font deps
_FONT_3X5: dict[str, tuple[str, ...]] = {
    " ": ("000", "000", "000", "000", "000"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "N": ("101", "111", "111", "111", "101"),
    "P": ("111", "101", "111", "100", "100"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "C": ("111", "100", "100", "100", "111"),
    "-": ("000", "000", "111", "000", "000"),
    "V": ("101", "101", "101", "101", "010"),
}


def _draw_badge(canvas: np.ndarray, text: str, x0: int, y0: int, scale: int = 2) -> None:
    char_w = 4 * scale
    pad_x, pad_y = 4, 3
    bw = len(text) * char_w + 2 * pad_x
    bh = 5 * scale + 2 * pad_y
    h, w, _ = canvas.shape
    y1, y2 = max(0, y0), min(h, y0 + bh)
    x1, x2 = max(0, x0), min(w, x0 + bw)
    canvas[y1:y2, x1:x2] = (canvas[y1:y2, x1:x2].astype(np.uint16) // 3).astype(np.uint8)
    for idx, ch in enumerate(text.upper()):
        glyph = _FONT_3X5.get(ch)
        if glyph is None:
            continue
        gx = x0 + pad_x + idx * char_w
        gy = y0 + pad_y
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == "1":
                    py1, py2 = gy + r * scale, gy + (r + 1) * scale
                    px1, px2 = gx + c * scale, gx + (c + 1) * scale
                    if 0 <= py1 and py2 <= h and 0 <= px1 and px2 <= w:
                        canvas[py1:py2, px1:px2] = 255


def _iter_camera_frames(width: int, height: int, fps: float) -> Iterator[np.ndarray]:
    with V4L2Camera(CameraConfig(width=width, height=height, pixel_format="MJPG", fps=fps)) as cam:
        while True:
            frame = cam.read_frame(decode=True, timeout_s=2.0)
            if frame.data is not None:
                yield frame.data


def _iter_video_file_frames(video_path: str, width: int, height: int) -> Iterator[np.ndarray]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = width * height * 3
    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.terminate()
        proc.wait(timeout=3.0)


def _iter_synthetic_frames(width: int, height: int) -> Iterator[np.ndarray]:
    y, x = np.mgrid[0:height, 0:width]
    t = 0
    while True:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = ((x + t * 8) % 256).astype(np.uint8)
        rgb[:, :, 1] = (y * 255 // max(1, height)).astype(np.uint8)
        rgb[:, :, 2] = 128
        cx = int(width * (0.3 + 0.4 * (0.5 + 0.5 * np.sin(t * 0.2))))
        cy = height // 2
        rgb[max(0, cy - 45) : min(height, cy + 45), max(0, cx - 45) : min(width, cx + 45)] = [240, 210, 170]
        yield rgb
        t += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast streaming Video Depth Anything Small — side-by-side Input Video | Depth Map."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "npu"),
        default="auto",
        help="Execution backend ('auto' picks fastest Small configuration, or force 'cpu'/'npu')",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=140,
        help="Model square input (multiple of 14: 140=~10fps max speed, 196=~4fps balanced, 252=high detail)",
    )
    parser.add_argument("--input-video", type=str, default=None, help="Optional video file path instead of live webcam")
    parser.add_argument("--output-mp4", type=str, default=None, help="Optional MP4 file path to save side-by-side output")
    parser.add_argument("--panel-width", type=int, default=320, help="Width of each side panel (total width = 2x + 2)")
    parser.add_argument("--panel-height", type=int, default=240, help="Height of each side panel")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop cleanly after N seconds (0 = until Ctrl+C)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop cleanly after N frames (0 = unlimited)")
    parser.add_argument("--no-loopback", action="store_true", help="Disable writing to /dev/video4 loopback device")
    parser.add_argument("--preview", action="store_true", help="Open live ffplay window automatically")
    parser.add_argument("--synthetic", action="store_true", help="Use moving synthetic scene if no camera needed")
    args = parser.parse_args()

    backend = "cpu" if args.device == "auto" else args.device
    if backend == "npu" and not is_device_available(ComputeDevice.NPU):
        print("NPU is not available on this host.", file=sys.stderr)
        return 1

    cfg = get_video_depth_config("small", input_size=args.input_size)
    print(f"Loading Video-Depth-Anything Small ({args.input_size}x{args.input_size}) on {backend.upper()}...")
    weights = load_from_hf(cfg, local_files_only=False)
    model = VideoDepthAnythingModel(cfg, weights, backend=backend)

    pw, ph = args.panel_width, args.panel_height
    if backend == "npu":
        # Warm up IRON kernel handles upfront so live frame 0 starts without compilation delay.
        model.infer(np.zeros((ph, pw, 3), dtype=np.uint8))
        model.reset_temporal_state()
    divider = 2
    out_w, out_h = pw * 2 + divider, ph
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, pw : pw + divider] = 240  # crisp center divider

    loopback_node = None if args.no_loopback else pick_default_loopback()
    vcam = None
    if loopback_node is not None:
        vcam = VirtualCameraOutput(VirtualCameraConfig(device=loopback_node.path, width=out_w, height=out_h, fps=15.0))
        vcam.__enter__()

    ffplay_proc = None
    if args.preview and shutil.which("ffplay"):
        ffplay_proc = subprocess.Popen(
            [
                "ffplay",
                "-hide_banner",
                "-loglevel",
                "error",
                "-window_title",
                f"Video Depth Anything Small ({backend.upper()} {args.input_size}x{args.input_size})",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{out_w}x{out_h}",
                "-i",
                "pipe:0",
            ],
            stdin=subprocess.PIPE,
        )

    ffmpeg_out = None
    if args.output_mp4 and shutil.which("ffmpeg"):
        ffmpeg_out = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-video_size",
                f"{out_w}x{out_h}",
                "-framerate",
                "10",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                args.output_mp4,
            ],
            stdin=subprocess.PIPE,
        )

    if args.input_video:
        frames_iter = _iter_video_file_frames(args.input_video, pw, ph)
        src_desc = args.input_video
    elif not args.synthetic and pick_default_camera() is not None:
        frames_iter = _iter_camera_frames(640, 480, 30.0)
        src_desc = pick_default_camera().path  # type: ignore[union-attr]
    else:
        frames_iter = _iter_synthetic_frames(pw, ph)
        src_desc = "synthetic scene"

    if vcam is not None:
        print(f"Streaming side-by-side [{src_desc} -> {loopback_node.path} ({out_w}x{out_h})]")
        print(f"Preview with: ffplay {loopback_node.path}")
    else:
        print(f"Running side-by-side [{src_desc}] ({out_w}x{out_h})")
    print("Press Ctrl+C to stop.")

    count = 0
    infer_ms_sum = 0.0
    t_start = time.perf_counter()
    t_last_log = t_start
    tag = "NPU-VDA" if backend == "npu" else "CPU-VDA"

    try:
        for rgb in frames_iter:
            res = model.infer(rgb)
            left = resize_rgb(rgb, pw, ph)
            right = depth_to_colormap_rgb(res.depth, out_width=pw, out_height=ph)

            canvas[:, :pw] = left
            canvas[:, pw + divider :] = right
            _draw_badge(canvas, "IN", 6, 6)
            _draw_badge(canvas, tag, pw + divider + 6, 6)

            if vcam is not None:
                vcam.write_frame(canvas)
            if ffplay_proc is not None and ffplay_proc.stdin is not None:
                ffplay_proc.stdin.write(canvas.tobytes())
            if ffmpeg_out is not None and ffmpeg_out.stdin is not None:
                ffmpeg_out.stdin.write(canvas.tobytes())

            count += 1
            if count > 1:
                infer_ms_sum += res.total_ms
            now = time.perf_counter()
            if now - t_last_log >= 1.0:
                steady_n = max(1, count - 1)
                avg_ms = (infer_ms_sum / steady_n) if count > 1 else res.total_ms
                print(
                    f"[side-by-side] frame={count:4d} | model={avg_ms:5.1f} ms ({1000.0/avg_ms:4.2f} fps) "
                    f"[bb={res.backbone_ms:4.1f}ms, t_dpt={res.neck_head_ms:4.1f}ms]",
                    flush=True,
                )
                t_last_log = now

            if args.max_frames > 0 and count >= args.max_frames:
                break
            if args.duration > 0.0 and (now - t_start) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        model.close()
        if vcam is not None:
            vcam.__exit__(None, None, None)
        if ffplay_proc is not None:
            if ffplay_proc.stdin is not None:
                ffplay_proc.stdin.close()
            ffplay_proc.wait(timeout=3.0)
        if ffmpeg_out is not None:
            if ffmpeg_out.stdin is not None:
                ffmpeg_out.stdin.close()
            ffmpeg_out.wait(timeout=5.0)

    elapsed = max(time.perf_counter() - t_start, 1e-6)
    steady_n = max(1, count - 1)
    avg_ms = (infer_ms_sum / steady_n) if count > 1 else 0.0
    print(f"\n[done] {count} frames in {elapsed:.2f}s | avg steady-state model={avg_ms:.1f} ms ({1000.0/max(avg_ms, 1e-3):.2f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
