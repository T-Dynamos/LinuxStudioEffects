"""Benchmark each discovered effect/backend against frames from a live camera."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    CameraConfig,
    DeviceUnavailableError,
    FrameContext,
    V4L2Camera,
    available_effect_backends,
    create_effect,
    is_device_available,
    pick_default_camera,
    query_video_device,
)
from linux_studio_effects.video.pixels import NATIVE_PIXEL_FORMATS

_MODEL_EFFECTS = frozenset(("portrait_blur",))


def _size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT, for example 1280x720") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive")
    return width, height


def _best_mode(info, effect_name: str, pixel_format: str | None, size: tuple[int, int] | None):
    formats = NATIVE_PIXEL_FORMATS
    modes = [
        mode
        for mode in info.modes
        if (mode.pixel_format == "MJPG" or mode.pixel_format in formats)
        and (pixel_format is None or mode.pixel_format == "MJPG" or mode.pixel_format == pixel_format)
        and (size is None or (mode.width, mode.height) == size)
    ]
    return max(
        modes,
        key=lambda mode: (
            mode.width * mode.height,
            max(mode.fps_options, default=0.0),
            mode.pixel_format == "MJPG",
        ),
        default=None,
    )


def _benchmark_one(
    effect_name: str,
    backend: str,
    camera: str,
    mode,
    *,
    pixel_format: str | None,
    duration_s: float,
    warmup_frames: int,
    variant: str,
    input_size: int,
    real_weights: bool,
) -> None:
    effect = None
    try:
        process_format = pixel_format or ("YUYV" if mode.pixel_format == "MJPG" else mode.pixel_format)
        options: dict[str, object] = {}
        if effect_name in _MODEL_EFFECTS:
            options.update(
                variant=variant,
                input_size=input_size,
                use_real_weights=real_weights,
            )
        effect = create_effect(effect_name, device=backend, **options)
        fps = max(mode.fps_options, default=30.0)
        frame_times_ms: list[float] = []
        frames = dropped = 0
        with V4L2Camera(
            CameraConfig(
                device=camera,
                width=mode.width,
                height=mode.height,
                pixel_format=mode.pixel_format,
                fps=fps,
            )
        ) as capture:
            negotiated = capture.negotiated
            effect.setup(negotiated.width, negotiated.height, negotiated.fps)

            def run_one(index: int) -> None:
                nonlocal dropped
                captured = capture.read_frame(native=True, output_format=process_format, timeout_s=3.0)
                dropped += captured.dropped_since_last
                if captured.data is None:
                    return
                context = FrameContext(
                    frame_index=index,
                    timestamp_s=captured.driver_timestamp_s,
                    width=captured.width,
                    height=captured.height,
                    fps=negotiated.fps,
                    pixel_format=process_format,
                )
                start = time.perf_counter()
                output = effect.process(captured.data, context)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if output.shape != captured.data.shape or output.dtype != captured.data.dtype:
                    raise RuntimeError(
                        f"Effect changed native frame shape/dtype: {captured.data.shape}/{captured.data.dtype} "
                        f"-> {output.shape}/{output.dtype}"
                    )
                frame_times_ms.append(elapsed_ms)

            for index in range(warmup_frames):
                run_one(index + 1)
            frame_times_ms.clear()
            dropped = 0
            frames = 0
            start_wall = time.perf_counter()
            deadline = start_wall + duration_s
            while time.perf_counter() < deadline:
                before = len(frame_times_ms)
                run_one(warmup_frames + frames + 1)
                if len(frame_times_ms) > before:
                    frames += 1

            elapsed_s = max(time.perf_counter() - start_wall, 1e-9)
            if frame_times_ms:
                avg_ms = sum(frame_times_ms) / len(frame_times_ms)
                ordered = sorted(frame_times_ms)
                p95_ms = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
                print(
                    f"{effect_name:16s} {backend:4s} {negotiated.width}x{negotiated.height} "
                    f"{negotiated.pixel_format:4s}"
                    f"{f'->{process_format}' if negotiated.pixel_format != process_format else '     '} "
                    f"camera={frames / elapsed_s:5.2f} fps "
                    f"effect={avg_ms:8.2f} ms avg / {p95_ms:8.2f} ms p95 "
                    f"({1000.0 / avg_ms:5.2f} effect fps) dropped={dropped}",
                    flush=True,
                )
            else:
                print(f"{effect_name:16s} {backend:4s} no frames processed", flush=True)
    except Exception as exc:
        print(f"{effect_name:16s} {backend:4s} FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if effect is not None:
            try:
                effect.teardown()
            except Exception as exc:
                print(f"{effect_name:16s} {backend:4s} teardown warning: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-device", default="", help="Camera node; default is the best accessible camera")
    parser.add_argument("--size", type=_size, help="Force a supported camera size, e.g. 1280x720")
    parser.add_argument(
        "--pixel-format",
        choices=NATIVE_PIXEL_FORMATS,
        help="Force the processing format; MJPG camera frames are decoded to this format with PyAV",
    )
    parser.add_argument("--duration", type=float, default=3.0, help="Timed seconds for each effect/backend")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Unmeasured frames before each run")
    parser.add_argument("--variant", choices=("small", "base", "large"), default="small")
    parser.add_argument("--input-size", type=int, default=140, help="Depth model input size; must be a multiple of 14")
    parser.add_argument("--real-weights", action="store_true", help="Load real depth weights instead of synthetic weights")
    parser.add_argument("--list", action="store_true", help="List effect/backend pairs without opening a camera")
    args = parser.parse_args()

    catalog = available_effect_backends()
    if args.list:
        for effect_name, backends in catalog.items():
            for backend in backends:
                print(f"{effect_name}:{backend}")
        return 0
    if args.duration <= 0 or args.warmup_frames < 0:
        parser.error("--duration must be positive and --warmup-frames cannot be negative")
    if args.input_size <= 0 or args.input_size % 14:
        parser.error("--input-size must be a positive multiple of 14")

    info = query_video_device(args.input_device) if args.input_device else pick_default_camera()
    if info is None:
        parser.error("No accessible V4L2 camera found")

    print(f"Benchmark camera: {info.path} ({info.card})")
    print(f"Timed duration: {args.duration:g}s per backend; warm-up: {args.warmup_frames} frames")
    print("Depth weights: " + ("real" if args.real_weights else "synthetic"))
    for effect_name, backends in catalog.items():
        mode = _best_mode(info, effect_name, args.pixel_format, args.size)
        if mode is None:
            expected_formats = ", ".join(NATIVE_PIXEL_FORMATS)
            for backend in backends:
                print(
                    f"{effect_name:16s} {backend:4s} SKIPPED: no raw mode in ({expected_formats}) "
                    "or MJPG mode for PyAV decoding"
                )
            continue
        for backend in backends:
            if not is_device_available(backend):
                print(f"{effect_name:16s} {backend:4s} SKIPPED: backend unavailable")
                continue
            _benchmark_one(
                effect_name,
                backend,
                info.path,
                mode,
                pixel_format=args.pixel_format,
                duration_s=args.duration,
                warmup_frames=args.warmup_frames,
                variant=args.variant,
                input_size=args.input_size,
                real_weights=args.real_weights,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
