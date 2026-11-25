"""Run any discovered effect on synthetic frames or through a live virtual camera."""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    DeviceUnavailableError,
    FrameContext,
    PipelineConfig,
    VideoPipeline,
    available_effect_backends,
    create_effect,
    pick_default_camera,
    query_video_device,
)


def _size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT, e.g. 1280x720") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive")
    return width, height


def _synthetic_run(effect, width: int, height: int, fps: float, frames: int) -> int:
    """Exercise an effect without camera or loopback hardware."""
    effect.setup(width, height, fps)
    times: list[float] = []
    try:
        yy, xx = np.indices((height, width))
        for index in range(frames):
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (xx + index * 7) % 256
            frame[:, :, 1] = (yy + index * 11) % 256
            frame[:, :, 2] = ((xx + yy) // 2 + index * 13) % 256
            ctx = FrameContext(index + 1, index / fps, width, height, fps)
            start = time.perf_counter()
            output = effect.process(frame, ctx)
            times.append((time.perf_counter() - start) * 1000)
            if output.shape != frame.shape or output.dtype != np.uint8:
                raise RuntimeError(f"Effect returned invalid frame: shape={output.shape}, dtype={output.dtype}")
    finally:
        effect.teardown()
    print(
        f"{effect.name} ({effect.active_device.value}): processed {frames} synthetic frames "
        f"at {width}x{height}; average {sum(times) / len(times):.2f} ms/frame"
    )
    return 0


def _run_privileged(command: list[str]) -> None:
    """Run kernel-device management as root, prompting through sudo if needed."""
    cmd = command if os.geteuid() == 0 else ["sudo", *command]
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command is missing: {cmd[0]}") from exc
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _ensure_loopback_device(device: str):
    """Create a missing v4l2loopback node and return its matching cleanup callback."""
    device_path = pathlib.Path(device)
    if device_path.exists():
        return None

    number = device_path.name.removeprefix("video")
    if (
        device_path.parent != pathlib.Path("/dev")
        or not device_path.name.startswith("video")
        or not number.isdigit()
    ):
        raise RuntimeError(f"Cannot auto-create {device}: expected a /dev/videoN path")

    if pathlib.Path("/sys/module/v4l2loopback").exists():
        ctl = shutil.which("v4l2loopback-ctl")
        if not ctl:
            raise RuntimeError(
                "v4l2loopback is loaded but the requested device is missing; "
                "install v4l2loopback-ctl to create and remove devices dynamically"
            )
        _run_privileged([ctl, "add", "--name", "LinuxStudioEffects", "--exclusive-caps", "1", device])

        def cleanup() -> None:
            _run_privileged([ctl, "delete", device])

    else:
        modprobe = shutil.which("modprobe") or "/usr/bin/modprobe"
        _run_privileged(
            [
                modprobe,
                "v4l2loopback",
                f"video_nr={number}",
                "card_label=LinuxStudioEffects",
                "exclusive_caps=1",
            ]
        )

        def cleanup() -> None:
            _run_privileged([modprobe, "-r", "v4l2loopback"])

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if device_path.exists():
            print(f"Created temporary v4l2loopback device {device}", flush=True)
            return cleanup
        time.sleep(0.05)

    cleanup()
    raise RuntimeError(f"v4l2loopback setup finished, but {device} did not appear")


def main() -> int:
    catalog = available_effect_backends()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List effects and implemented backends")
    parser.add_argument("--effect", choices=sorted(catalog), default="mirror")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "npu"), default="auto")
    parser.add_argument("--in-size", type=_size, help="Camera input size; default is its largest advertised mode")
    parser.add_argument("--out-size", type=_size, help="Virtual camera output size; default matches input")
    parser.add_argument("--fps", type=float, help="Frame rate; default is the largest mode's advertised maximum")
    parser.add_argument("--frames", type=int, default=0, help="Process synthetic frames without camera hardware")
    parser.add_argument("--duration", type=float, default=0.0, help="Live run duration; 0 runs until Ctrl+C")
    parser.add_argument("--variant", choices=("small", "base", "large"), default="small", help="Depth model variant")
    parser.add_argument("--input-size", type=int, default=140, help="Depth model input size (multiple of 14)")
    parser.add_argument("--real-weights", action="store_true", help="Load real model weights instead of synthetic weights")
    parser.add_argument("--input-device", default="", help="Camera device; default auto-detect")
    parser.add_argument(
        "--output-device",
        default="/dev/video4",
        help="v4l2loopback output device; a missing /dev/videoN is created and cleaned up automatically",
    )
    args = parser.parse_args()

    if args.list:
        for name, backends in catalog.items():
            print(f"{name}: {', '.join(backends) if backends else '(no backend files)'}")
        return 0
    if args.frames < 0 or (args.fps is not None and args.fps <= 0) or args.duration < 0:
        parser.error("--frames and --duration must be >= 0, and --fps must be positive")

    model_options: dict[str, object] = {}
    if args.effect in {"video_depth", "portrait_blur"}:
        if args.input_size <= 0 or args.input_size % 14:
            parser.error("--input-size must be a positive multiple of 14 for depth effects")
        model_options = {
            "variant": args.variant,
            "input_size": args.input_size,
            "use_real_weights": args.real_weights,
        }
    try:
        effect = create_effect(args.effect, device=args.device, **model_options)
    except (DeviceUnavailableError, ImportError) as exc:
        parser.error(str(exc))

    if args.frames:
        synthetic_size = args.in_size or (640, 480)
        return _synthetic_run(effect, *synthetic_size, args.fps or 30.0, args.frames)

    camera_info = query_video_device(args.input_device) if args.input_device else pick_default_camera()
    camera = camera_info.path if camera_info else ""
    loopback = args.output_device
    if not camera:
        parser.error("No accessible camera found; use --frames N to run on synthetic frames")

    selected_mode = None
    if args.in_size is None and camera_info and camera_info.modes:
        selected_mode = max(
            camera_info.modes,
            key=lambda mode: (
                mode.width * mode.height,
                mode.pixel_format == "MJPG",
                max(mode.fps_options, default=0.0),
            ),
        )
    if args.in_size:
        input_width, input_height = args.in_size
        input_format = "MJPG"
        mode_fps = 30.0
    elif selected_mode is not None:
        input_width, input_height = selected_mode.width, selected_mode.height
        input_format = selected_mode.pixel_format
        mode_fps = max(selected_mode.fps_options, default=30.0)
    else:
        input_width, input_height = 1920, 1080
        input_format = "MJPG"
        mode_fps = 30.0
    fps = args.fps or mode_fps
    output_width, output_height = args.out_size or (input_width, input_height)

    config = PipelineConfig(
        input_device=camera,
        input_width=input_width,
        input_height=input_height,
        input_format=input_format,
        output_device=loopback,
        output_width=output_width,
        output_height=output_height,
        input_fps=fps,
        output_fps=fps,
    )
    pipeline = VideoPipeline(config, effect=effect)

    def report(stats) -> None:
        print(
            f"[{effect.name}:{effect.active_device.value}] in={stats.input_fps:.1f} fps "
            f"out={stats.output_fps:.1f} fps process+write={stats.avg_process_ms:.2f} ms",
            flush=True,
        )

    print(
        f"Using {camera} at {input_width}x{input_height} {input_format} -> "
        f"{loopback} at {output_width}x{output_height} ({fps:g} fps)",
        flush=True,
    )
    try:
        cleanup_loopback = _ensure_loopback_device(loopback)
    except RuntimeError as exc:
        parser.error(str(exc))
    try:
        result = pipeline.run(duration_s=args.duration or None, on_stats=report)
    finally:
        if cleanup_loopback is not None:
            try:
                cleanup_loopback()
                print(f"Removed temporary v4l2loopback device {loopback}", flush=True)
            except RuntimeError as exc:
                print(f"WARNING: could not remove temporary loopback device: {exc}", file=sys.stderr)
    print(f"Stopped after {result.frames_out} output frames ({result.output_fps:.2f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
