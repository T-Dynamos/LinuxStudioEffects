"""Example 06: Stream live webcam through NPU GreyscaleEffect into v4l2loopback."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    GreyscaleEffect,
    PipelineConfig,
    PipelineStats,
    VideoPipeline,
    pick_default_loopback,
)


def _parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x", 1)
    return int(w), int(h)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run NPU-accelerated GreyscaleEffect on live webcam -> v4l2loopback until Ctrl+C."
    )
    parser.add_argument("--in-size", type=_parse_size, default=(1280, 720))
    parser.add_argument("--out-size", type=_parse_size, default=(1280, 720))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = until Ctrl+C)")
    args = parser.parse_args()

    loopback = pick_default_loopback()
    if loopback is None:
        print(
            "No v4l2loopback device found. Load it first with:\n"
            "  sudo modprobe v4l2loopback devices=1 card_label='LinuxStudioEffects' exclusive_caps=1",
            file=sys.stderr,
        )
        return 1

    eff = GreyscaleEffect()
    cfg = PipelineConfig(
        input_width=args.in_size[0],
        input_height=args.in_size[1],
        input_fps=args.fps,
        output_device=loopback.path,
        output_width=args.out_size[0],
        output_height=args.out_size[1],
        output_fps=args.fps,
    )
    pipe = VideoPipeline(cfg, effect=eff)

    print(
        f"Initializing NPU GreyscaleEffect (first run compiles/caches IRON kernel) -> {loopback.path}\n"
        f"Preview with: ffplay {loopback.path}\nPress Ctrl+C to stop.\n",
        flush=True,
    )

    def _on_tick(st: PipelineStats) -> None:
        print(
            f"[live-npu] in={st.input_fps:.1f} fps -> out={st.output_fps:.1f} fps | "
            f"npu+write={st.avg_process_ms:.2f} ms | CPU={st.cpu_percent:.1f}%",
            flush=True,
        )

    final = pipe.run(duration_s=args.duration if args.duration > 0 else None, on_stats=_on_tick)
    print(
        f"\n[stopped] {final.frames_out} frames out ({final.output_fps:.2f} fps), "
        f"avg_proc={final.avg_process_ms:.2f} ms, CPU={final.cpu_percent:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
