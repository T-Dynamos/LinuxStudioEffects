"""Configurable camera input -> frame processor -> virtual camera output pipeline."""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .camera import CameraConfig, V4L2Camera
from .effects import BUILTIN_EFFECTS, EffectChain, FrameContext, FrameEffect, as_frame_effect
from .virtual_cam import VirtualCameraConfig, VirtualCameraOutput, resize_rgb

FrameProcessor = FrameEffect | Callable[[np.ndarray], np.ndarray]
StatsCallback = Callable[["PipelineStats"], None]


@dataclass(frozen=True)
class PipelineConfig:
    """End-to-end camera input and virtual camera output stream configuration."""

    input_device: str = ""
    input_width: int = 1280
    input_height: int = 720
    input_format: str = "MJPG"
    input_fps: float = 30.0
    output_device: str = ""
    output_width: int = 1280
    output_height: int = 720
    output_format: str = "YUYV"
    output_fps: float = 30.0
    resize_mode: str = "cover"


@dataclass(frozen=True)
class PipelineStats:
    """Throughput, latency, and CPU utilization summary for a pipeline run."""

    input_config: CameraConfig
    output_config: VirtualCameraConfig
    frames_in: int
    frames_out: int
    dropped_in: int
    elapsed_s: float
    input_fps: float
    output_fps: float
    avg_process_ms: float
    cpu_percent: float


class VideoPipeline:
    """Streams frames from a physical V4L2 camera into a v4l2loopback virtual camera."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        frame_processor: FrameProcessor | None = None,
        effect: FrameEffect | list[FrameProcessor] | tuple[FrameProcessor, ...] | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        chosen = effect if effect is not None else frame_processor
        if chosen is None:
            self._effect: FrameEffect | None = None
        elif isinstance(chosen, (list, tuple)):
            self._effect = EffectChain(chosen)
        else:
            self._effect = as_frame_effect(chosen)
        self._stop_event = threading.Event()
        self._negotiated_in: CameraConfig | None = None
        self._negotiated_out: VirtualCameraConfig | None = None

    @property
    def effect(self) -> FrameEffect | None:
        return self._effect

    @property
    def negotiated_input(self) -> CameraConfig | None:
        return self._negotiated_in

    @property
    def negotiated_output(self) -> VirtualCameraConfig | None:
        return self._negotiated_out

    def stop(self) -> None:
        """Signal a running pipeline loop to exit cleanly after the current frame."""
        self._stop_event.set()

    def run(
        self,
        *,
        duration_s: float | None = None,
        max_frames: int | None = None,
        stats_interval_s: float = 1.0,
        on_stats: StatsCallback | None = None,
    ) -> PipelineStats:
        """Run the capture -> process -> resize -> virtual-cam output loop."""
        self._stop_event.clear()
        cfg = self._config

        in_cfg = CameraConfig(
            device=cfg.input_device,
            width=cfg.input_width,
            height=cfg.input_height,
            pixel_format=cfg.input_format,
            fps=cfg.input_fps,
        )
        out_cfg = VirtualCameraConfig(
            device=cfg.output_device,
            width=cfg.output_width,
            height=cfg.output_height,
            pixel_format=cfg.output_format,
            fps=cfg.output_fps,
        )

        with V4L2Camera(in_cfg) as cam, VirtualCameraOutput(out_cfg) as vcam:
            self._negotiated_in = cam.negotiated
            self._negotiated_out = vcam.negotiated

            if self._effect is not None:
                self._effect.setup(cam.negotiated.width, cam.negotiated.height, cam.negotiated.fps)

            wall_start = time.monotonic()
            cpu_start = time.process_time()
            last_report_wall = wall_start
            last_report_cpu = cpu_start
            last_report_in = 0
            last_report_out = 0

            frames_in = 0
            frames_out = 0
            dropped_in = 0
            proc_ms_samples: list[float] = []

            target_out_period = (1.0 / cfg.output_fps) if cfg.output_fps > 0 else 0.0
            next_out_deadline = wall_start

            try:
                while not self._stop_event.is_set():
                    now = time.monotonic()
                    if duration_s is not None and (now - wall_start) >= duration_s:
                        break
                    if max_frames is not None and frames_out >= max_frames:
                        break

                    # Determine whether this captured frame should be emitted to output.
                    should_emit = target_out_period <= 0.0 or now >= next_out_deadline - 0.002
                    frame = cam.read_frame(decode=should_emit)
                    frames_in += 1
                    dropped_in += frame.dropped_since_last
                    frame_now = frame.capture_monotonic_s

                    if should_emit and frame.data is not None:
                        t0 = time.perf_counter()
                        if self._effect is not None and self._effect.enabled:
                            ctx = FrameContext(
                                frame_index=frames_in,
                                timestamp_s=frame.driver_timestamp_s,
                                width=frame.width,
                                height=frame.height,
                                fps=cam.negotiated.fps,
                            )
                            out_rgb = self._effect.process(frame.data, ctx)
                        else:
                            out_rgb = frame.data
                        if out_rgb.shape[1] != vcam.negotiated.width or out_rgb.shape[0] != vcam.negotiated.height:
                            out_rgb = resize_rgb(
                                out_rgb,
                                vcam.negotiated.width,
                                vcam.negotiated.height,
                                mode=cfg.resize_mode,
                            )
                        vcam.write_frame(out_rgb)
                        proc_ms_samples.append((time.perf_counter() - t0) * 1000.0)
                        frames_out += 1

                        if target_out_period > 0.0:
                            next_out_deadline += target_out_period
                            if next_out_deadline < frame_now:
                                next_out_deadline = frame_now + target_out_period

                    if on_stats is not None and stats_interval_s > 0 and (frame_now - last_report_wall) >= stats_interval_s:
                        dt = max(frame_now - last_report_wall, 1e-9)
                        cpu_now = time.process_time()
                        d_cpu = max(cpu_now - last_report_cpu, 0.0)
                        on_stats(
                            PipelineStats(
                                input_config=cam.negotiated,
                                output_config=vcam.negotiated,
                                frames_in=frames_in - last_report_in,
                                frames_out=frames_out - last_report_out,
                                dropped_in=dropped_in,
                                elapsed_s=dt,
                                input_fps=(frames_in - last_report_in) / dt,
                                output_fps=(frames_out - last_report_out) / dt,
                                avg_process_ms=statistics.fmean(proc_ms_samples[-30:]) if proc_ms_samples else 0.0,
                                cpu_percent=(d_cpu / dt) * 100.0,
                            )
                        )
                        last_report_wall = frame_now
                        last_report_cpu = cpu_now
                        last_report_in = frames_in
                        last_report_out = frames_out
            except KeyboardInterrupt:
                self._stop_event.set()
            finally:
                if self._effect is not None:
                    self._effect.teardown()

            total_wall = max(time.monotonic() - wall_start, 1e-9)
            total_cpu = max(time.process_time() - cpu_start, 0.0)
            return PipelineStats(
                input_config=cam.negotiated,
                output_config=vcam.negotiated,
                frames_in=frames_in,
                frames_out=frames_out,
                dropped_in=dropped_in,
                elapsed_s=total_wall,
                input_fps=frames_in / total_wall,
                output_fps=frames_out / total_wall,
                avg_process_ms=statistics.fmean(proc_ms_samples) if proc_ms_samples else 0.0,
                cpu_percent=(total_cpu / total_wall) * 100.0,
            )


def _parse_size(text: str) -> tuple[int, int]:
    parts = text.lower().split("x", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected WxH size, got {text!r}")
    return int(parts[0]), int(parts[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run configurable V4L2 camera -> v4l2loopback pipeline.")
    parser.add_argument("--in-device", default="", help="Input camera path (default: auto-detect)")
    parser.add_argument("--in-size", type=_parse_size, default=(1280, 720), help="Input WxH (default: 1280x720)")
    parser.add_argument("--in-format", default="MJPG", help="Input FourCC (default: MJPG)")
    parser.add_argument("--in-fps", type=float, default=30.0, help="Input FPS (default: 30)")
    parser.add_argument("--out-device", default="", help="Output v4l2loopback path (default: auto-detect)")
    parser.add_argument("--out-size", type=_parse_size, default=(1280, 720), help="Output WxH (default: 1280x720)")
    parser.add_argument("--out-format", default="YUYV", help="Output FourCC: YUYV, RGB3, BGR3 (default: YUYV)")
    parser.add_argument("--out-fps", type=float, default=30.0, help="Output FPS (default: 30)")
    parser.add_argument("--resize-mode", choices=("cover", "stretch"), default="cover")
    parser.add_argument(
        "--effect",
        choices=tuple(BUILTIN_EFFECTS.keys()),
        default="none",
        help="Built-in CPU frame effect to apply (default: none)",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = until Ctrl+C)")
    args = parser.parse_args(argv)

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
    pipe = VideoPipeline(cfg, effect=BUILTIN_EFFECTS[args.effect]())

    def _print_tick(st: PipelineStats) -> None:
        print(
            f"[pipeline] in={st.input_fps:.1f} fps ({st.input_config.width}x{st.input_config.height} {st.input_config.pixel_format}) "
            f"-> out={st.output_fps:.1f} fps ({st.output_config.width}x{st.output_config.height} {st.output_config.pixel_format} @ {st.output_config.device}) "
            f"| proc+write={st.avg_process_ms:.2f} ms | CPU={st.cpu_percent:.1f}%",
            flush=True,
        )

    try:
        summary = pipe.run(
            duration_s=args.duration if args.duration > 0 else None,
            on_stats=_print_tick,
        )
    except KeyboardInterrupt:
        pipe.stop()
        return 0

    print(
        f"\n[summary] {summary.frames_in} frames in ({summary.input_fps:.2f} fps), "
        f"{summary.frames_out} frames out ({summary.output_fps:.2f} fps), "
        f"dropped_in={summary.dropped_in}, avg_proc={summary.avg_process_ms:.2f} ms, "
        f"CPU={summary.cpu_percent:.1f}% (100% = 1 full core)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
