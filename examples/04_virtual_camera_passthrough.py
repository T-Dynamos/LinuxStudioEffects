"""Example 04: Interactive/CLI camera -> v4l2loopback passthrough until Ctrl+C."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    PipelineConfig,
    PipelineStats,
    VideoDeviceInfo,
    VideoMode,
    VideoPipeline,
    pick_default_camera,
    pick_default_loopback,
    query_video_device,
)
from linux_studio_effects.video.pixels import NATIVE_PIXEL_FORMATS


def _parse_size(text: str) -> tuple[int, int]:
    parts = text.lower().strip().split("x", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected WxH (e.g. 1280x720), got {text!r}")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"Resolution must be positive, got {text!r}")
    return w, h


def _sorted_modes(dev: VideoDeviceInfo) -> list[tuple[str, int, int, float]]:
    """Return unique (format, width, height, max_fps) modes sorted sensibly."""
    entries: list[tuple[str, int, int, float]] = []
    for m in dev.modes:
        if m.pixel_format not in NATIVE_PIXEL_FORMATS:
            continue
        fps = max(m.fps_options) if m.fps_options else 30.0
        entries.append((m.pixel_format, m.width, m.height, fps))
    fmt_rank = {"YUYV": 0, "UYVY": 1, "RGB3": 2, "BGR3": 3}
    return sorted(entries, key=lambda e: (fmt_rank.get(e[0], 2), e[1] * e[2], e[0]))


def _prompt_input_mode(dev: VideoDeviceInfo) -> tuple[int, int, str, float]:
    """Prompt user to pick one of the camera's advertised modes (or enter custom)."""
    modes = _sorted_modes(dev)
    if not modes:
        return 1280, 720, "YUYV", 30.0

    default_idx = next(
        (i for i, (fmt, w, h, _) in enumerate(modes) if fmt == "YUYV" and w == 1280 and h == 720),
        next((i for i, (fmt, _, _, _) in enumerate(modes) if fmt == "YUYV"), 0),
    )

    print(f"\nInput camera: {dev.path} ({dev.card})")
    print("Available input modes:")
    for i, (fmt, w, h, fps) in enumerate(modes, start=1):
        tag = " (default)" if (i - 1) == default_idx else ""
        print(f"  [{i:2d}] {w}x{h} {fmt} @ {fps:g} fps{tag}")

    while True:
        raw = input(f"Select input mode [1-{len(modes)}, Enter={default_idx + 1}]: ").strip()
        if not raw:
            fmt, w, h, fps = modes[default_idx]
            return w, h, fmt, fps
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(modes):
                fmt, w, h, fps = modes[idx]
                return w, h, fmt, fps
        if "x" in raw.lower():
            try:
                w, h = _parse_size(raw)
                return w, h, modes[default_idx][0], 30.0
            except argparse.ArgumentTypeError:
                pass
        print("  Invalid choice. Enter a number from the list or WxH (e.g. 1280x720).")


def _prompt_output_config(in_w: int, in_h: int, in_fps: float) -> tuple[int, int, float]:
    """Prompt user for output resolution and FPS."""
    presets: list[tuple[int, int, str]] = [
        (in_w, in_h, f"Same as input ({in_w}x{in_h})"),
    ]
    for w, h in ((640, 480), (1280, 720), (1920, 1080)):
        if (w, h) != (in_w, in_h):
            presets.append((w, h, f"{w}x{h}"))

    print("\nOutput resolution presets:")
    for i, (_, _, label) in enumerate(presets, start=1):
        tag = " (default)" if i == 1 else ""
        print(f"  [{i}] {label}{tag}")

    out_w, out_h = in_w, in_h
    while True:
        raw = input(f"Select output resolution [1-{len(presets)} or WxH, Enter=1]: ").strip()
        if not raw:
            out_w, out_h = presets[0][0], presets[0][1]
            break
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(presets):
                out_w, out_h = presets[idx][0], presets[idx][1]
                break
        if "x" in raw.lower():
            try:
                out_w, out_h = _parse_size(raw)
                break
            except argparse.ArgumentTypeError:
                pass
        print("  Invalid choice. Enter a preset number or WxH (e.g. 640x480).")

    while True:
        fps_raw = input(f"Output FPS [Enter={in_fps:g}]: ").strip()
        if not fps_raw:
            return out_w, out_h, in_fps
        try:
            fps_val = float(fps_raw)
            if fps_val > 0:
                return out_w, out_h, fps_val
        except ValueError:
            pass
        print("  Invalid FPS. Enter a positive number (e.g. 15 or 30).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream physical camera into a v4l2loopback virtual webcam until Ctrl+C."
    )
    parser.add_argument("--in-device", default="", help="Input camera (default: auto-detect)")
    parser.add_argument("--in-size", type=_parse_size, default=None, help="Input WxH (e.g. 1280x720)")
    parser.add_argument("--in-format", default=None, choices=NATIVE_PIXEL_FORMATS, help="Native input FourCC")
    parser.add_argument("--in-fps", type=float, default=None, help="Input FPS")
    parser.add_argument("--out-device", default="", help="Output v4l2loopback node (default: auto-detect)")
    parser.add_argument("--out-size", type=_parse_size, default=None, help="Output WxH (e.g. 640x480)")
    parser.add_argument("--out-format", default=None, choices=NATIVE_PIXEL_FORMATS, help="Output FourCC; must match input")
    parser.add_argument("--out-fps", type=float, default=None, help="Output FPS")
    parser.add_argument("--resize-mode", choices=("cover", "stretch"), default="cover")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = until Ctrl+C)")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts and use defaults/CLI arguments directly",
    )
    args = parser.parse_args()

    loopback = query_video_device(args.out_device) if args.out_device else pick_default_loopback()
    if loopback is None or not loopback.can_output:
        print(
            "No accessible v4l2loopback device found. Load it first with:\n"
            "  sudo modprobe v4l2loopback devices=1 card_label='LinuxStudioEffects' exclusive_caps=1",
            file=sys.stderr,
        )
        return 1

    in_dev = query_video_device(args.in_device) if args.in_device else pick_default_camera()
    if in_dev is None or not in_dev.can_capture:
        print("No accessible V4L2 input camera found.", file=sys.stderr)
        return 1

    interactive = sys.stdin.isatty() and not args.non_interactive
    needs_in_prompt = interactive and args.in_size is None and args.in_format is None
    needs_out_prompt = interactive and args.out_size is None

    try:
        if needs_in_prompt:
            in_w, in_h, in_fmt, in_fps = _prompt_input_mode(in_dev)
        else:
            preferred_fmt = (args.in_format or "YUYV").upper()
            available = [mode for mode in _sorted_modes(in_dev) if mode[0] == preferred_fmt]
            if args.in_size:
                available = [mode for mode in available if (mode[1], mode[2]) == args.in_size]
            if not available:
                print(
                    f"No {preferred_fmt} camera mode matches {args.in_size or 'an available resolution'}; "
                    "input/output pixel conversion is disabled.",
                    file=sys.stderr,
                )
                return 1
            selected = max(available, key=lambda mode: (mode[1] * mode[2], mode[3]))
            in_fmt, default_w, default_h, default_fps = selected
            in_w, in_h = args.in_size or (default_w, default_h)
            in_fps = args.in_fps if args.in_fps is not None else default_fps

        if args.in_format is not None and args.in_format.upper() != in_fmt:
            print("--in-format must match the selected camera mode", file=sys.stderr)
            return 1
        if args.in_fps is not None:
            in_fps = args.in_fps

        out_fmt = (args.out_format or in_fmt).upper()
        if out_fmt != in_fmt:
            print(
                f"Input format {in_fmt} differs from output format {out_fmt}; conversion is disabled.",
                file=sys.stderr,
            )
            return 1

        if needs_out_prompt:
            out_w, out_h, out_fps = _prompt_output_config(in_w, in_h, in_fps)
        else:
            out_w, out_h = args.out_size or (in_w, in_h)
            out_fps = args.out_fps if args.out_fps is not None else in_fps

        if args.out_fps is not None:
            out_fps = args.out_fps
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 0

    cfg = PipelineConfig(
        input_device=in_dev.path,
        input_width=in_w,
        input_height=in_h,
        input_format=in_fmt,
        input_fps=in_fps,
        output_device=loopback.path,
        output_width=out_w,
        output_height=out_h,
        output_format=out_fmt,
        output_fps=out_fps,
        resize_mode=args.resize_mode,
    )
    pipe = VideoPipeline(cfg)

    print(
        f"\nStarting virtual camera passthrough:\n"
        f"  Input : {in_dev.path} ({in_w}x{in_h} {in_fmt} @ {in_fps:g} fps)\n"
        f"  Output: {loopback.path} ({out_w}x{out_h} {cfg.output_format} @ {out_fps:g} fps)\n"
        f"Preview with: ffplay {loopback.path}\n"
        f"Press Ctrl+C to stop.\n",
        flush=True,
    )

    def _on_tick(st: PipelineStats) -> None:
        print(
            f"[live] {st.input_config.device} ({st.input_config.width}x{st.input_config.height} "
            f"{st.input_config.pixel_format} @ {st.input_fps:.1f} fps) -> "
            f"{st.output_config.device} ({st.output_config.width}x{st.output_config.height} "
            f"{st.output_config.pixel_format} @ {st.output_fps:.1f} fps) | "
            f"proc+write={st.avg_process_ms:.2f} ms | CPU={st.cpu_percent:.1f}%",
            flush=True,
        )

    final = pipe.run(duration_s=args.duration if args.duration > 0 else None, on_stats=_on_tick)
    print(
        f"\n[stopped] in={final.frames_in} ({final.input_fps:.2f} fps), "
        f"out={final.frames_out} ({final.output_fps:.2f} fps), "
        f"dropped_in={final.dropped_in}, avg_proc={final.avg_process_ms:.2f} ms, "
        f"CPU={final.cpu_percent:.1f}% (100% = 1 full core)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
