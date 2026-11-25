"""Example 03: Benchmark raw V4L2 capture FPS and decoded RGB FPS."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import CameraConfig, benchmark_camera, format_benchmark_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark physical webcam capture & decode FPS.")
    parser.add_argument("--device", default="", help="Camera node (default: auto-detect)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--format", default="MJPG", help="V4L2 pixel format (MJPG or YUYV)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    cfg = CameraConfig(
        device=args.device,
        width=args.width,
        height=args.height,
        pixel_format=args.format.upper(),
        fps=args.fps,
    )
    print(format_benchmark_result(benchmark_camera(cfg, duration_s=args.duration, decode=False)))
    print()
    print(format_benchmark_result(benchmark_camera(cfg, duration_s=args.duration, decode=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
