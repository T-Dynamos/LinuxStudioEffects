"""Example 05: Device-agnostic frame-in -> frame-out -> virtual camera pipeline."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    BUILTIN_EFFECTS,
    ComputeDevice,
    FrameContext,
    FrameEffect,
    FrameGraph,
    PipelineConfig,
    PipelineStats,
    VideoPipeline,
    pick_default_loopback,
)
from linux_studio_effects.video.effects.mirror.cpu import MirrorEffect
from linux_studio_effects.video.effects.spotlight.cpu import SpotlightEffect


class WarmTintEffect(FrameEffect):
    """Example custom FrameEffect showing setup/process/teardown with device selection."""

    supported_devices = (ComputeDevice.CPU,)

    def __init__(self, warmth: float = 0.12, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.warmth = warmth
        self._lut_r: np.ndarray | None = None
        self._lut_b: np.ndarray | None = None

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        idx = np.arange(256, dtype=np.float32)
        self._lut_r = np.clip(np.round(idx * (1.0 + self.warmth)), 0, 255).astype(np.uint8)
        self._lut_b = np.clip(np.round(idx * (1.0 - self.warmth)), 0, 255).astype(np.uint8)

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        assert self._lut_r is not None and self._lut_b is not None
        out = frame.copy()
        out[:, :, 0] = self._lut_r[frame[:, :, 0]]
        out[:, :, 2] = self._lut_b[frame[:, :, 2]]
        return out

    def teardown(self) -> None:
        self._lut_r = None
        self._lut_b = None
        super().teardown()


def simple_border_overlay(frame: np.ndarray) -> np.ndarray:
    """Example plain function (frame -> frame) usable directly in EffectChain or VideoPipeline."""
    out = frame.copy()
    out[:4, :, :] = [0, 220, 140]
    out[-4:, :, :] = [0, 220, 140]
    return out


def _parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x", 1)
    return int(w), int(h)


def _build_effect(preset: str, device: str) -> FrameEffect:
    if preset == "studio_demo":
        graph = FrameGraph(name="StudioDemoGraph")
        graph.add("mirror", MirrorEffect(device=device))
        graph.add("spotlight", SpotlightEffect(edge_dim=0.45, device=device), source="mirror")
        graph.add("warm_tint", WarmTintEffect(warmth=0.10, device=device), source="spotlight")
        graph.add("border", simple_border_overlay, source="warm_tint")
        graph.set_output("border")
        return graph
    return BUILTIN_EFFECTS[preset]()


def main() -> int:
    choices = (*BUILTIN_EFFECTS.keys(), "studio_demo")
    parser = argparse.ArgumentParser(
        description="Run a composable frame graph into v4l2loopback until Ctrl+C."
    )
    parser.add_argument("--effect", choices=choices, default="studio_demo", help="Effect preset (default: studio_demo)")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto", help="Compute backend for CPU-only demo effects")
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

    eff = _build_effect(args.effect, args.device)
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
        f"Running effect={eff.name} (backend={eff.active_device.value}) "
        f"into {loopback.path} ({args.out_size[0]}x{args.out_size[1]} @ {args.fps:g} fps)\n"
        f"Preview with: ffplay {loopback.path}\nPress Ctrl+C to stop.\n",
        flush=True,
    )

    def _on_tick(st: PipelineStats) -> None:
        print(
            f"[live] out={st.output_fps:.1f} fps | effect+write={st.avg_process_ms:.2f} ms | CPU={st.cpu_percent:.1f}%",
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
