"""Example 04: Configurable camera -> v4l2loopback virtual webcam passthrough with CPU/FPS monitor."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import PipelineConfig, PipelineStats, VideoPipeline, pick_default_loopback


def _parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x", 1)
    return int(w), int(h)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream physical camera into a v4l2loopback virtual webcam and monitor CPU/FPS."
    )
    parser.add_argument("--in-device", default="", help="Input camera (default: auto-detect)")
    parser.add_argument("--in-size", type=_parse_size, default=(1280, 720), help="Input WxH (default: 1280x720)")
    parser.add_argument("--in-format", default="MJPG", help="Input FourCC (default: MJPG)")
    parser.add_argument("--in-fps", type=float, default=30.0, help="Input FPS (default: 30)")
    parser.add_argument("--out-device", default="", help="Output v4l2loopback node (default: auto-detect)")
    parser.add_argument("--out-size", type=_parse_size, default=(1280, 720), help="Output WxH (default: 1280x720)")
    parser.add_argument("--out-format", default="YUYV", help="Output FourCC: YUYV, RGB3, BGR3")
    parser.add_argument("--out-fps", type=float, default=30.0, help="Output FPS (default: 30)")
    parser.add_argument("--resize-mode", choices=("cover", "stretch"), default="cover")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to run (0 = until Ctrl+C)")
    args = parser.parse_args()

    if not args.out_device and pick_default_loopback() is None:
        print(
            "No v4l2loopback device found. Load it first with:\n"
            "  sudo modprobe v4l2loopback devices=1 card_label='LinuxStudioEffects' exclusive_caps=1",
            file=sys.stderr,
        )
        return 1

    cfg = PipelineConfig(
        input_device=args.in_device,
        input_width=args.in_size[0],
        input_height=args.in_size[1],
        input_format=args.in_format.upper(),
        input_fps=args.in_fps,
        output_device=args.out_device,
        output_width=args.out_size[0],
        output_height=args.out_size[1],
        output_format=args.out_format.upper(),
        output_fps=args.out_fps,
        resize_mode=args.resize_mode,
    )
    pipe = VideoPipeline(cfg)

    def _on_tick(st: PipelineStats) -> None:
        print(
            f"[live] {st.input_config.device} ({st.input_config.width}x{st.input_config.height} "
            f"{st.input_config.pixel_format} @ {st.input_fps:.1f} fps) -> "
            f"{st.output_config.device} ({st.output_config.width}x{st.output_config.height} "
            f"{st.output_config.pixel_format} @ {st.output_fps:.1f} fps) | "
            f"proc+write={st.avg_process_ms:.2f} ms | CPU={st.cpu_percent:.1f}%",
            flush=True,
        )

    try:
        final = pipe.run(duration_s=args.duration if args.duration > 0 else None, on_stats=_on_tick)
    except KeyboardInterrupt:
        pipe.stop()
        return 0

    print(
        f"\n[done] in={final.frames_in} ({final.input_fps:.2f} fps), "
        f"out={final.frames_out} ({final.output_fps:.2f} fps), "
        f"dropped_in={final.dropped_in}, avg_proc={final.avg_process_ms:.2f} ms, "
        f"CPU={final.cpu_percent:.1f}% (100% = 1 full core)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
